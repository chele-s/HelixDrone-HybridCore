import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'Release'))

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import deque
import time

from python_src.envs.drone_env import QuadrotorEnv, EnvConfig, TaskType, VectorizedQuadrotorEnv
from python_src.agents import TD3Agent, DDPGAgent, PrioritizedReplayBuffer, ReplayBuffer, create_agent


@dataclass
class TrainConfig:
    agent_type: str = 'td3'
    total_timesteps: int = 500_000
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1
    
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    hidden_dim: int = 256
    
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    
    exploration_noise: float = 0.1
    exploration_noise_decay: float = 0.9999
    exploration_noise_min: float = 0.01
    
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    
    eval_freq: int = 5000
    eval_episodes: int = 10
    save_freq: int = 25000
    log_freq: int = 1000
    
    seed: int = 42
    device: str = 'auto'
    checkpoint_dir: str = 'checkpoints'
    
    num_envs: int = 1


class RollingStats:
    def __init__(self, window: int = 100):
        self.window = window
        self.rewards = deque(maxlen=window)
        self.lengths = deque(maxlen=window)
        self.successes = deque(maxlen=window)
    
    def add(self, reward: float, length: int, success: bool = False):
        self.rewards.append(reward)
        self.lengths.append(length)
        self.successes.append(float(success))
    
    @property
    def mean_reward(self) -> float:
        return np.mean(self.rewards) if self.rewards else 0.0
    
    @property
    def mean_length(self) -> float:
        return np.mean(self.lengths) if self.lengths else 0.0
    
    @property
    def success_rate(self) -> float:
        return np.mean(self.successes) if self.successes else 0.0


class Trainer:
    def __init__(self, config: TrainConfig, env_config: Optional[EnvConfig] = None):
        self.config = config
        self.env_config = env_config or EnvConfig()
        
        self._setup_seed()
        self._setup_device()
        self._setup_env()
        self._setup_agent()
        self._setup_buffer()
        self._setup_logging()
    
    def _setup_seed(self):
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
    
    def _setup_device(self):
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
    
    def _setup_env(self):
        if self.config.num_envs > 1:
            self.env = VectorizedQuadrotorEnv(
                num_envs=self.config.num_envs,
                config=self.env_config,
                task=TaskType.HOVER
            )
            self.is_vectorized = True
        else:
            self.env = QuadrotorEnv(config=self.env_config, task=TaskType.HOVER)
            self.is_vectorized = False
        
        self.eval_env = QuadrotorEnv(config=self.env_config, task=TaskType.HOVER)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
    
    def _setup_agent(self):
        self.agent = create_agent(
            agent_type=self.config.agent_type,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
            hidden_dim=self.config.hidden_dim,
            lr_actor=self.config.lr_actor,
            lr_critic=self.config.lr_critic,
            gamma=self.config.gamma,
            tau=self.config.tau,
            policy_noise=self.config.policy_noise,
            noise_clip=self.config.noise_clip,
            policy_delay=self.config.policy_delay
        )
        
        self.exploration_noise = self.config.exploration_noise
    
    def _setup_buffer(self):
        if self.config.use_per:
            self.buffer = PrioritizedReplayBuffer(
                capacity=self.config.buffer_size,
                device=self.device,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                alpha=self.config.per_alpha,
                beta_start=self.config.per_beta_start,
                beta_frames=self.config.total_timesteps
            )
        else:
            self.buffer = ReplayBuffer(
                capacity=self.config.buffer_size,
                device=self.device,
                state_dim=self.state_dim,
                action_dim=self.action_dim
            )
    
    def _setup_logging(self):
        self.stats = RollingStats(window=100)
        self.best_eval_reward = float('-inf')
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.timesteps = 0
        self.episodes = 0
        self.start_time = None
    
    def train(self) -> Dict[str, List[float]]:
        self.start_time = time.time()
        history = {'rewards': [], 'eval_rewards': [], 'actor_loss': [], 'critic_loss': []}
        
        obs, _ = self.env.reset(seed=self.config.seed)
        episode_reward = 0.0 if not self.is_vectorized else np.zeros(self.config.num_envs)
        episode_length = 0 if not self.is_vectorized else np.zeros(self.config.num_envs, dtype=int)
        
        while self.timesteps < self.config.total_timesteps:
            if self.timesteps < self.config.learning_starts:
                if self.is_vectorized:
                    action = np.array([self.env.action_space.sample() for _ in range(self.config.num_envs)])
                else:
                    action = self.env.action_space.sample()
            else:
                if self.is_vectorized:
                    action = np.array([
                        self._get_action_with_noise(obs[i]) for i in range(self.config.num_envs)
                    ])
                else:
                    action = self._get_action_with_noise(obs)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            if self.is_vectorized:
                for i in range(self.config.num_envs):
                    self.buffer.push(obs[i], action[i], reward[i], next_obs[i], terminated[i])
                    episode_reward[i] += reward[i]
                    episode_length[i] += 1
                    
                    if terminated[i] or truncated[i]:
                        self.stats.add(episode_reward[i], episode_length[i])
                        history['rewards'].append(episode_reward[i])
                        self.episodes += 1
                        episode_reward[i] = 0.0
                        episode_length[i] = 0
                
                self.timesteps += self.config.num_envs
            else:
                done = terminated or truncated
                self.buffer.push(obs, action, reward, next_obs, terminated)
                episode_reward += reward
                episode_length += 1
                self.timesteps += 1
                
                if done:
                    self.stats.add(episode_reward, episode_length)
                    history['rewards'].append(episode_reward)
                    self.episodes += 1
                    obs, _ = self.env.reset()
                    episode_reward = 0.0
                    episode_length = 0
                    self.agent.reset_noise()
                    continue
            
            obs = next_obs
            
            if self.timesteps >= self.config.learning_starts:
                if self.timesteps % self.config.train_freq == 0:
                    for _ in range(self.config.gradient_steps):
                        metrics = self.agent.update(self.buffer, self.config.batch_size)
                        if metrics:
                            history['actor_loss'].append(metrics.get('actor_loss', 0))
                            history['critic_loss'].append(metrics.get('critic_loss', 0))
                
                self.exploration_noise = max(
                    self.config.exploration_noise_min,
                    self.exploration_noise * self.config.exploration_noise_decay
                )
            
            if self.timesteps % self.config.log_freq == 0:
                self._log_progress()
            
            if self.timesteps % self.config.eval_freq == 0:
                eval_reward = self._evaluate()
                history['eval_rewards'].append(eval_reward)
                
                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.agent.save(self.checkpoint_dir / 'best')
            
            if self.timesteps % self.config.save_freq == 0:
                self.agent.save(self.checkpoint_dir / f'step_{self.timesteps}')
        
        self.agent.save(self.checkpoint_dir / 'final')
        return history
    
    def _get_action_with_noise(self, obs: np.ndarray) -> np.ndarray:
        action = self.agent.get_action(obs, add_noise=False)
        noise = np.random.randn(self.action_dim) * self.exploration_noise
        action = np.clip(action + noise, -1.0, 1.0)
        return action
    
    def _evaluate(self) -> float:
        total_reward = 0.0
        
        for _ in range(self.config.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            
            while not done:
                action = self.agent.get_action(obs, add_noise=False)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                total_reward += reward
                done = terminated or truncated
        
        mean_reward = total_reward / self.config.eval_episodes
        print(f"[EVAL] Steps: {self.timesteps} | Mean Reward: {mean_reward:.2f}")
        return mean_reward
    
    def _log_progress(self):
        elapsed = time.time() - self.start_time
        fps = self.timesteps / elapsed if elapsed > 0 else 0
        
        print(
            f"Steps: {self.timesteps:>7} | "
            f"Episodes: {self.episodes:>5} | "
            f"Reward: {self.stats.mean_reward:>8.2f} | "
            f"Length: {self.stats.mean_length:>6.1f} | "
            f"Noise: {self.exploration_noise:.3f} | "
            f"FPS: {fps:.0f}"
        )


def main():
    train_config = TrainConfig(
        agent_type='td3',
        total_timesteps=500_000,
        batch_size=512,
        buffer_size=1_000_000,
        learning_starts=10000,
        train_freq=20,
        gradient_steps=10,
        
        lr_actor=3e-4,
        lr_critic=3e-4,
        hidden_dim=512,
        
        policy_noise=0.15,
        noise_clip=0.4,
        policy_delay=2,
        
        exploration_noise=0.25,
        exploration_noise_decay=0.99998,
        exploration_noise_min=0.05,
        
        use_per=True,
        per_alpha=0.7,
        per_beta_start=0.5,
        
        eval_freq=10000,
        save_freq=50000,
        log_freq=1000,
        
        seed=42,
        num_envs=1
    )
    
    env_config = EnvConfig(
        dt=0.02,
        max_steps=1000,
        domain_randomization=False,
        wind_enabled=False,
        motor_dynamics=True,
        
        reward_position=-2.0,
        reward_velocity=-0.2,
        reward_angular=-0.1,
        reward_action=-0.005,
        reward_action_rate=-0.01,
        reward_alive=3.0,
        reward_crash=-100.0,
        reward_success=100.0
    )
    
    trainer = Trainer(train_config, env_config)
    
    print("=" * 60)
    print("HelixDrone TD3 Training")
    print("=" * 60)
    print(f"Device: {trainer.device}")
    print(f"State dim: {trainer.state_dim}, Action dim: {trainer.action_dim}")
    print(f"Agent: {train_config.agent_type.upper()}")
    print(f"Buffer: {'PER' if train_config.use_per else 'Uniform'}")
    print(f"Total timesteps: {train_config.total_timesteps:,}")
    print("=" * 60)
    
    history = trainer.train()
    
    print("=" * 60)
    print("Training Complete")
    print(f"Best eval reward: {trainer.best_eval_reward:.2f}")
    print(f"Final model saved to: {trainer.checkpoint_dir / 'final'}")
    print("=" * 60)


if __name__ == "__main__":
    main()