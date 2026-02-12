import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'Release'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'build'))

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import deque
import time

from python_src.envs.drone_env import QuadrotorEnv, EnvConfig, TaskType, VectorizedQuadrotorEnv
from python_src.agents import TD3Agent, DDPGAgent, TD3LSTMAgent, CppPrioritizedReplayBuffer as PrioritizedReplayBuffer, ReplayBuffer, CppSequenceReplayBuffer as SequenceReplayBuffer, CppSequencePrioritizedReplayBuffer as SequencePrioritizedReplayBuffer, create_agent
from python_src.utils.helix_math import RunningMeanStd


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
    tau: float = 0.001
    hidden_dim: int = 256
    
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    
    exploration_noise_start: float = 0.1
    exploration_noise_end: float = 0.01
    exploration_warmup_steps: int = 25000
    exploration_noise_decay: float = 0.9999
    exploration_noise_min: float = 0.01
    
    use_per: bool = True
    per_alpha: float = 0.4
    per_beta_start: float = 0.5
    
    use_obs_normalization: bool = True
    obs_norm_clip: float = 10.0
    obs_norm_update_freq: int = 100
    
    eval_freq: int = 5000
    eval_episodes: int = 10
    save_freq: int = 25000
    log_freq: int = 1000
    
    seed: int = 42
    device: str = 'auto'
    checkpoint_dir: str = 'checkpoints'
    resume_from: Optional[str] = None
    
    num_envs: int = 1
    
    lstm_hidden: int = 128
    lstm_layers: int = 2
    sequence_length: int = 16
    burn_in_length: int = 0


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


class AdaptiveExplorationNoise:
    def __init__(
        self,
        start_noise: float = 0.1,
        end_noise: float = 0.01,
        warmup_steps: int = 25000,
        decay_rate: float = 0.9999,
        min_noise: float = 0.01,
        crash_penalty_factor: float = 1.5
    ):
        self.start_noise = start_noise
        self.end_noise = end_noise
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.min_noise = min_noise
        self.crash_penalty_factor = crash_penalty_factor
        
        self.current_noise = start_noise
        self.steps = 0
        self.recent_crashes = deque(maxlen=100)
        self.warmup_complete = False
    
    def update(self, crashed: bool = False) -> float:
        self.steps += 1
        self.recent_crashes.append(float(crashed))
        
        if self.steps < self.warmup_steps:
            progress = self.steps / self.warmup_steps
            self.current_noise = self.start_noise + (self.end_noise - self.start_noise) * progress
        else:
            self.warmup_complete = True
            self.current_noise = max(
                self.min_noise,
                self.current_noise * self.decay_rate
            )
        
        crash_rate = np.mean(self.recent_crashes) if self.recent_crashes else 0.0
        if crash_rate > 0.5 and self.steps > 1000:
            boost = 1.0 + (crash_rate - 0.5) * self.crash_penalty_factor
            self.current_noise = min(self.start_noise, self.current_noise * boost)
        
        return self.current_noise
    
    @property
    def noise(self) -> float:
        return self.current_noise


class Trainer:
    def __init__(self, config: TrainConfig, env_config: Optional[EnvConfig] = None):
        self.config = config
        self.env_config = env_config or EnvConfig()
        
        self._setup_seed()
        self._setup_device()
        self._setup_env()
        self._setup_agent()
        self._setup_buffer()
        self._setup_normalization()
        self._setup_exploration()
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
        
        # Create deterministic eval environment (no randomization)
        from dataclasses import replace
        eval_env_config = replace(
            self.env_config,
            domain_randomization=False,
            wind_enabled=False,
            curriculum_enabled=False
        )
        self.eval_env = QuadrotorEnv(config=eval_env_config, task=TaskType.HOVER)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        self.use_lstm = self.config.agent_type.lower() == 'td3_lstm'
        if self.use_lstm:
            self.base_obs_dim = 20
    
    def _setup_agent(self):
        if self.use_lstm:
            self.agent = TD3LSTMAgent(
                obs_dim=self.base_obs_dim,
                action_dim=self.action_dim,
                device=self.device,
                hidden_dim=self.config.hidden_dim,
                lstm_hidden=self.config.lstm_hidden,
                lstm_layers=self.config.lstm_layers,
                sequence_length=self.config.sequence_length,
                burn_in_length=self.config.burn_in_length,
                lr_actor=self.config.lr_actor,
                lr_critic=self.config.lr_critic,
                gamma=self.config.gamma,
                tau=self.config.tau,
                policy_noise=self.config.policy_noise,
                noise_clip=self.config.noise_clip,
                policy_delay=self.config.policy_delay
            )
        else:
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
    
    def _setup_buffer(self):
        if self.use_lstm:
            if self.config.use_per:
                self.buffer = SequencePrioritizedReplayBuffer(
                    capacity=self.config.buffer_size,
                    device=self.device,
                    obs_dim=self.base_obs_dim,
                    action_dim=self.action_dim,
                    sequence_length=self.config.sequence_length,
                    burn_in_length=self.config.burn_in_length,
                    alpha=self.config.per_alpha,
                    beta_start=self.config.per_beta_start,
                    beta_frames=self.config.total_timesteps
                )
            else:
                self.buffer = SequenceReplayBuffer(
                    capacity=self.config.buffer_size,
                    device=self.device,
                    obs_dim=self.base_obs_dim,
                    action_dim=self.action_dim,
                    sequence_length=self.config.sequence_length,
                    burn_in_length=self.config.burn_in_length
                )
        else:
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
    
    def _setup_normalization(self):
        if self.config.use_obs_normalization and not self.use_lstm:
            self.obs_normalizer = RunningMeanStd(shape=(self.state_dim,))
        else:
            self.obs_normalizer = None
    
    def _setup_exploration(self):
        self.exploration = AdaptiveExplorationNoise(
            start_noise=self.config.exploration_noise_start,
            end_noise=self.config.exploration_noise_end,
            warmup_steps=self.config.exploration_warmup_steps,
            decay_rate=self.config.exploration_noise_decay,
            min_noise=self.config.exploration_noise_min
        )
    
    def _setup_logging(self):
        self.stats = RollingStats(window=100)
        self.best_eval_reward = float('-inf')
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.timesteps = 0
        self.episodes = 0
        self.start_time = None
        self._last_log_time = None
        self._last_log_steps = 0
        
        if self.config.resume_from:
            self._resume_from_checkpoint(self.config.resume_from)
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"[RESUME] Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"[RESUME] Loading from {checkpoint_path}")
        self.agent.load(checkpoint_path)
        
        normalizer_path = checkpoint_path / 'obs_normalizer.npz'
        if normalizer_path.exists() and self.obs_normalizer is not None:
            data = np.load(normalizer_path)
            self.obs_normalizer.mean = data['mean']
            self.obs_normalizer.var = data['var']
            self.obs_normalizer.count = data['count']
            print(f"[RESUME] Loaded observation normalizer")
        
        state_path = checkpoint_path / 'training_state.npz'
        if state_path.exists():
            state = np.load(state_path)
            self.timesteps = int(state['timesteps'])
            self.episodes = int(state['episodes'])
            self.best_eval_reward = float(state['best_eval_reward'])
            print(f"[RESUME] Restored state: steps={self.timesteps}, episodes={self.episodes}")
        else:
            ckpt_name = checkpoint_path.name
            if ckpt_name.startswith('step_'):
                self.timesteps = int(ckpt_name.split('_')[1])
                print(f"[RESUME] Inferred timesteps from checkpoint name: {self.timesteps}")
        
        print(f"[RESUME] Ready to continue training from step {self.timesteps}")
    
    def _normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if self.obs_normalizer is None:
            return obs
        
        if update and self.timesteps % self.config.obs_norm_update_freq == 0:
            if obs.ndim == 1:
                self.obs_normalizer.update(obs.reshape(1, -1))
            else:
                self.obs_normalizer.update(obs)
        
        return self.obs_normalizer.normalize(obs, clip=self.config.obs_norm_clip)
    
    def train(self) -> Dict[str, List[float]]:
        self.start_time = time.time()
        history = {'rewards': [], 'eval_rewards': [], 'actor_loss': [], 'critic_loss': []}
        
        if self.config.resume_from is not None:
            self._resume_from_checkpoint(self.config.resume_from)
            self.target_timesteps = self.timesteps + self.config.total_timesteps
            print(f"[RESUME] Will train for {self.config.total_timesteps:,} more steps (target: {self.target_timesteps:,})")
        else:
            self.target_timesteps = self.config.total_timesteps
        
        if self.obs_normalizer is not None and self.timesteps == 0:
            print("[WARMUP] Collecting observations to initialize normalizer...")
            warmup_obs_list = []
            for _ in range(10):
                obs_raw, _ = self.env.reset()
                warmup_obs_list.append(obs_raw.reshape(1, -1) if obs_raw.ndim == 1 else obs_raw)
                for _ in range(50):
                    if self.is_vectorized:
                        action = np.array([self.env.action_space.sample() for _ in range(self.config.num_envs)])
                    else:
                        action = self.env.action_space.sample()
                    obs_raw, _, done, trunc, _ = self.env.step(action)
                    warmup_obs_list.append(obs_raw.reshape(1, -1) if obs_raw.ndim == 1 else obs_raw)
                    if (done if not self.is_vectorized else done.any()):
                        obs_raw, _ = self.env.reset()
            
            warmup_obs = np.concatenate(warmup_obs_list, axis=0)
            self.obs_normalizer.update(warmup_obs)
            print(f"[WARMUP] Normalizer initialized with {len(warmup_obs)} observations")
            print(f"[WARMUP] Mean range: [{self.obs_normalizer.mean.min():.3f}, {self.obs_normalizer.mean.max():.3f}]")
            print(f"[WARMUP] Var range: [{self.obs_normalizer.var.min():.3f}, {self.obs_normalizer.var.max():.3f}]")
        
        obs_raw, _ = self.env.reset(seed=self.config.seed)
        obs = self._normalize_obs(obs_raw, update=True)
        
        if self.use_lstm:
            obs_base = self.env._get_base_obs() if hasattr(self.env, '_get_base_obs') else obs_raw[:self.base_obs_dim]
            self.agent.reset_hidden_states()
        
        episode_reward = 0.0 if not self.is_vectorized else np.zeros(self.config.num_envs)
        episode_length = 0 if not self.is_vectorized else np.zeros(self.config.num_envs, dtype=int)
        episode_crashes = np.zeros(self.config.num_envs, dtype=bool) if self.is_vectorized else False
        
        while self.timesteps < self.target_timesteps:
            if self.timesteps < self.config.learning_starts:
                if self.is_vectorized:
                    collective = np.random.randn(self.config.num_envs, 1).astype(np.float32) * 0.15
                    differential = np.random.randn(self.config.num_envs, self.action_dim).astype(np.float32) * 0.08
                    action = np.clip(collective + differential, -1.0, 1.0)
                else:
                    collective = np.float32(np.random.randn() * 0.15)
                    differential = np.random.randn(self.action_dim).astype(np.float32) * 0.08
                    action = np.clip(collective + differential, -1.0, 1.0)
            else:
                noise_scale = self.exploration.noise
                if self.is_vectorized:
                    if self.use_lstm:
                        obs_base_batch = obs_raw[:, :self.base_obs_dim]
                        action = self.agent.get_actions_batch(obs_base_batch, add_noise=False)
                        collective = np.random.randn(self.config.num_envs, 1).astype(np.float32) * noise_scale
                        differential = np.random.randn(self.config.num_envs, self.action_dim).astype(np.float32) * (noise_scale * 0.4)
                        action = np.clip(action + collective + differential, -1.0, 1.0)
                    else:
                        action = self.agent.get_actions_batch(obs, add_noise=True, noise_scale=noise_scale)
                else:
                    if self.use_lstm:
                        obs_base = obs_raw[:self.base_obs_dim]
                        action = self.agent.get_action(obs_base, add_noise=False)
                        collective = np.float32(np.random.randn() * noise_scale)
                        differential = np.random.randn(self.action_dim).astype(np.float32) * (noise_scale * 0.4)
                        action = np.clip(action + collective + differential, -1.0, 1.0)
                    else:
                        action = self._get_action_with_noise(obs, noise_scale)
            
            next_obs_raw, reward, terminated, truncated, info = self.env.step(action)
            next_obs = self._normalize_obs(next_obs_raw, update=True)
            
            if self.is_vectorized:
                if self.use_lstm:
                    obs_base_batch = obs_raw[:, :self.base_obs_dim]
                    next_obs_base_batch = next_obs_raw[:, :self.base_obs_dim]
                    self.buffer.push_batch(
                        states=obs_base_batch,
                        actions=action,
                        rewards=reward,
                        next_states=next_obs_base_batch,
                        dones=terminated
                    )
                else:
                    self.buffer.push_batch(
                        states=obs if obs.ndim == 2 else obs.reshape(1, -1),
                        actions=action if action.ndim == 2 else action.reshape(1, -1),
                        rewards=reward if isinstance(reward, np.ndarray) else np.array([reward]),
                        next_states=next_obs if next_obs.ndim == 2 else next_obs.reshape(1, -1),
                        dones=terminated if isinstance(terminated, np.ndarray) else np.array([terminated])
                    )
                
                done_envs = []
                for i in range(self.config.num_envs):
                    episode_reward[i] += reward[i]
                    episode_length[i] += 1
                    
                    if terminated[i] or truncated[i]:
                        crashed = terminated[i] and episode_length[i] < 100
                        self.exploration.update(crashed=crashed)
                        self.stats.add(episode_reward[i], episode_length[i])
                        history['rewards'].append(episode_reward[i])
                        self.episodes += 1
                        done_envs.append(i)
                        episode_reward[i] = 0.0
                        episode_length[i] = 0
                
                if self.use_lstm and done_envs:
                    self.agent.reset_hidden_states(np.array(done_envs))
                
                self.timesteps += self.config.num_envs
            else:
                done = terminated or truncated
                
                if self.use_lstm:
                    obs_base = obs_raw[:self.base_obs_dim]
                    next_obs_base = next_obs_raw[:self.base_obs_dim]
                    self.buffer.push(obs_base, action, reward, next_obs_base, terminated)
                else:
                    self.buffer.push(obs, action, reward, next_obs, terminated)
                    
                episode_reward += reward
                episode_length += 1
                self.timesteps += 1
                
                if done:
                    crashed = terminated and episode_length < 100
                    self.exploration.update(crashed=crashed)
                    self.stats.add(episode_reward, episode_length)
                    history['rewards'].append(episode_reward)
                    self.episodes += 1
                    obs_raw, _ = self.env.reset()
                    obs = self._normalize_obs(obs_raw, update=True)
                    if self.use_lstm:
                        self.agent.reset_hidden_states()
                    episode_reward = 0.0
                    episode_length = 0
                    self.agent.reset_noise()
                    continue
            
            obs_raw = next_obs_raw
            obs = next_obs
            
            if self.timesteps >= self.config.learning_starts:
                if self.timesteps % self.config.train_freq == 0:
                    for _ in range(self.config.gradient_steps):
                        critic_warmup = (self.timesteps - self.config.learning_starts) < 5000
                        metrics = self.agent.update(
                            self.buffer, self.config.batch_size,
                            critic_only=critic_warmup
                        )
                        if metrics:
                            history['actor_loss'].append(metrics.get('actor_loss', 0))
                            history['critic_loss'].append(metrics.get('critic_loss', 0))
            
            if self.timesteps % self.config.log_freq == 0:
                self._log_progress()
            
            if self.timesteps % self.config.eval_freq == 0:
                eval_reward = self._evaluate()
                history['eval_rewards'].append(eval_reward)
                
                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.agent.save(self.checkpoint_dir / 'best')
            
            if self.timesteps % self.config.save_freq == 0:
                ckpt_path = self.checkpoint_dir / f'step_{self.timesteps}'
                self.agent.save(ckpt_path)
                self._save_training_state(ckpt_path)
        
        final_path = self.checkpoint_dir / 'final'
        self.agent.save(final_path)
        self._save_training_state(final_path)
        return history
    
    def _get_action_with_noise(self, obs: np.ndarray, noise_scale: float) -> np.ndarray:
        action = self.agent.get_action(obs, add_noise=False)
        noise = np.random.randn(self.action_dim) * noise_scale
        action = np.clip(action + noise, -1.0, 1.0)
        return action
    
    def _evaluate(self) -> float:
        total_reward = 0.0
        total_length = 0
        
        for _ in range(self.config.eval_episodes):
            obs_raw, _ = self.eval_env.reset()
            obs = self._normalize_obs(obs_raw, update=False)
            done = False
            ep_length = 0
            
            if self.use_lstm:
                self.agent.reset_hidden_states()
            
            while not done:
                if self.use_lstm:
                    obs_base = obs_raw[:self.base_obs_dim]
                    action = self.agent.get_action(obs_base, add_noise=False)
                else:
                    action = self.agent.get_action(obs, add_noise=False)
                
                obs_raw, reward, terminated, truncated, _ = self.eval_env.step(action)
                obs = self._normalize_obs(obs_raw, update=False)
                total_reward += reward
                ep_length += 1
                done = terminated or truncated
            
            total_length += ep_length
        
        mean_reward = total_reward / self.config.eval_episodes
        mean_length = total_length / self.config.eval_episodes
        print(f"[EVAL] Steps: {self.timesteps} | Mean Reward: {mean_reward:.2f} | Mean Length: {mean_length:.0f}")
        return mean_reward
    
    def _log_progress(self):
        now = time.time()
        if self._last_log_time is None:
            self._last_log_time = now
            self._last_log_steps = self.timesteps
            fps = 0
        else:
            dt = now - self._last_log_time
            ds = self.timesteps - self._last_log_steps
            fps = ds / dt if dt > 0 else 0
            self._last_log_time = now
            self._last_log_steps = self.timesteps
        
        print(
            f"Steps: {self.timesteps:>7} | "
            f"Episodes: {self.episodes:>5} | "
            f"Reward: {self.stats.mean_reward:>8.2f} | "
            f"Length: {self.stats.mean_length:>6.1f} | "
            f"Noise: {self.exploration.noise:.3f} | "
            f"FPS: {fps:.0f}"
        )
        
        if self.timesteps % 50000 == 0 and self.timesteps > 0:
            self._log_detailed_metrics()
    
    def _log_detailed_metrics(self):
        lengths = list(self.stats.lengths)
        rewards = list(self.stats.rewards)
        
        if len(lengths) < 10:
            return
        
        short_eps = sum(1 for l in lengths if l < 50)
        medium_eps = sum(1 for l in lengths if 50 <= l < 200)
        long_eps = sum(1 for l in lengths if l >= 200)
        total = len(lengths)
        
        crash_rate = short_eps / total * 100 if total > 0 else 0
        stable_rate = long_eps / total * 100 if total > 0 else 0
        
        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths) if lengths else 0
        min_reward = min(rewards) if rewards else 0
        max_reward = max(rewards) if rewards else 0
        
        print("\n" + "=" * 70)
        print(f"[DETAILED METRICS @ Step {self.timesteps}]")
        print("-" * 70)
        print(f"  Episodes:   Short(<50): {short_eps:>3} ({crash_rate:.0f}%)  |  "
              f"Medium(50-200): {medium_eps:>3}  |  Long(>200): {long_eps:>3} ({stable_rate:.0f}%)")
        print(f"  Length:     Avg: {avg_length:.1f}  |  Max: {max_length}")
        print(f"  Reward:     Min: {min_reward:.1f}  |  Max: {max_reward:.1f}  |  Avg: {self.stats.mean_reward:.1f}")
        print(f"  Exploration: Noise={self.exploration.noise:.3f}")
        print("=" * 70 + "\n")
    
    def _save_normalizer(self):
        if self.obs_normalizer is not None:
            np.savez(
                self.checkpoint_dir / 'obs_normalizer.npz',
                mean=self.obs_normalizer.mean,
                var=self.obs_normalizer.var,
                count=self.obs_normalizer.count
            )
    
    def _save_training_state(self, ckpt_path: Path):
        self._save_normalizer()
        if self.obs_normalizer is not None:
            np.savez(
                ckpt_path / 'obs_normalizer.npz',
                mean=self.obs_normalizer.mean,
                var=self.obs_normalizer.var,
                count=self.obs_normalizer.count
            )
        np.savez(
            ckpt_path / 'training_state.npz',
            timesteps=self.timesteps,
            episodes=self.episodes,
            best_eval_reward=self.best_eval_reward
        )
    
    def load_normalizer(self, path: Path):
        if self.obs_normalizer is not None:
            data = np.load(path)
            self.obs_normalizer.mean = data['mean']
            self.obs_normalizer.var = data['var']
            self.obs_normalizer.count = data['count']


def main():
    import yaml
    
    config_path = os.path.join(ROOT_DIR, 'config', 'train_params.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    train_config = TrainConfig(
        agent_type=cfg['agent']['type'],
        total_timesteps=cfg['training']['total_timesteps'],
        batch_size=cfg['training']['batch_size'],
        buffer_size=cfg['replay_buffer']['size'],
        learning_starts=cfg['training']['learning_starts'],
        train_freq=cfg['training']['train_freq'],
        gradient_steps=cfg['training']['gradient_steps'],
        
        lr_actor=float(cfg['agent']['lr_actor']),
        lr_critic=float(cfg['agent']['lr_critic']),
        hidden_dim=cfg['agent']['hidden_dim'],
        gamma=cfg['agent']['gamma'],
        tau=cfg['agent']['tau'],
        
        policy_noise=cfg['td3_specific']['policy_noise'],
        noise_clip=cfg['td3_specific']['noise_clip'],
        policy_delay=cfg['td3_specific']['policy_delay'],
        
        exploration_noise_start=cfg['exploration']['noise'],
        exploration_noise_end=cfg['exploration']['noise_min'],
        exploration_warmup_steps=int(cfg['training']['total_timesteps'] * 0.1), # Auto-calc warmup
        exploration_noise_decay=cfg['exploration']['noise_decay'],
        exploration_noise_min=cfg['exploration']['noise_min'],
        
        use_per=cfg['replay_buffer']['use_per'],
        per_alpha=cfg['replay_buffer']['per_alpha'],
        per_beta_start=cfg['replay_buffer']['per_beta_start'],
        
        eval_freq=cfg['evaluation']['freq'],
        eval_episodes=cfg['evaluation']['episodes'],
        save_freq=cfg['checkpointing']['save_freq'],
        log_freq=cfg['checkpointing']['log_freq'],
        checkpoint_dir=cfg['checkpointing']['dir'],
        resume_from=cfg['checkpointing'].get('resume_from', None),
        
        seed=cfg['seed'],
        device=cfg['device'],
        num_envs=cfg['environment']['num_envs'],
        
        lstm_hidden=cfg['lstm'].get('hidden_dim', 128),
        lstm_layers=cfg['lstm'].get('num_layers', 2),
        sequence_length=cfg['lstm'].get('sequence_length', 16),
        burn_in_length=cfg['lstm'].get('burn_in', 0)
    )
    
    env_cfg = cfg['environment']
    rew_cfg = cfg['rewards']
    term_cfg = cfg['termination']
    curr_cfg = cfg.get('curriculum', {})
    
    env_config = EnvConfig(
        dt=env_cfg['dt'],
        max_steps=env_cfg['max_steps'],
        domain_randomization=env_cfg['domain_randomization'],
        wind_enabled=env_cfg['wind_enabled'],
        motor_dynamics=env_cfg['motor_dynamics'],
        physics_sub_steps=env_cfg.get('physics_sub_steps', 8),
        use_sub_stepping=env_cfg.get('use_sub_stepping', True),
        
        mass=float(env_cfg.get('mass', 0.6)),
        max_rpm=float(env_cfg.get('max_rpm', 35000.0)),
        min_rpm=float(env_cfg.get('min_rpm', 3000.0)),
        hover_rpm=float(env_cfg.get('hover_rpm', 5500.0)),
        rpm_range=float(env_cfg.get('rpm_range', 15000.0)),
        use_sota_actuator=env_cfg.get('use_sota_actuator', False),
        
        reward_proximity_scale=float(rew_cfg['proximity_scale']),
        reward_proximity_sigma=float(rew_cfg['proximity_sigma']),
        reward_velocity_approach=float(rew_cfg['velocity_approach']),
        reward_velocity_brake=float(rew_cfg['velocity_brake']),
        reward_orientation_scale=float(rew_cfg['orientation_scale']),
        reward_stability_scale=float(rew_cfg['stability_scale']),
        reward_smoothness_scale=float(rew_cfg['smoothness_scale']),
        reward_altitude_scale=float(rew_cfg['altitude_scale']),
        reward_hover_scale=float(rew_cfg['hover_scale']),
        reward_hover_max_bonus=float(rew_cfg['hover_max_bonus']),
        reward_low_height_penalty=float(rew_cfg['low_height_penalty']),
        reward_crash=float(rew_cfg['crash']),
        reward_success=float(rew_cfg['success']),
        
        crash_height=float(term_cfg['crash_height']),
        crash_distance=float(term_cfg['crash_distance']),
        crash_angle=float(term_cfg['crash_angle']),
        success_distance=float(term_cfg['success_distance']),
        success_velocity=float(term_cfg['success_velocity']),
        success_hold_steps=int(term_cfg['success_hold_steps']),
        
        curriculum_enabled=curr_cfg.get('enabled', True),
        curriculum_init_range=float(curr_cfg.get('init_range', 0.1)),
        curriculum_max_range=float(curr_cfg.get('max_range', 0.5)),
        curriculum_progress_rate=float(curr_cfg.get('progress_rate', 0.00005))
    )

    
    trainer = Trainer(train_config, env_config)
    
    print("=" * 60)
    print("HelixDrone TD3 Training")
    print("=" * 60)
    print(f"Device: {trainer.device}")
    print(f"State dim: {trainer.state_dim}, Action dim: {trainer.action_dim}")
    print(f"Agent: {train_config.agent_type.upper()}")
    buffer_type = 'PER (C++)' if train_config.use_per and hasattr(trainer.buffer, 'using_cpp') and trainer.buffer.using_cpp else ('PER (Python)' if train_config.use_per else 'Uniform')
    print(f"Buffer: {buffer_type}")
    print(f"Observation Normalization: {'Enabled' if train_config.use_obs_normalization else 'Disabled'}")
    print(f"Exploration: warmup={train_config.exploration_warmup_steps}, start={train_config.exploration_noise_start}")
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