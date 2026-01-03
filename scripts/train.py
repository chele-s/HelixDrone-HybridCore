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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARN] wandb not installed. Run: pip install wandb")

from python_src.envs.drone_env import QuadrotorEnv, ExtendedEnvConfig as EnvConfig, TaskType, VectorizedQuadrotorEnv
from python_src.envs.reward_functions import RewardConfig
from python_src.envs.frame_stack import FrameStack
from python_src.agents import TD3Agent, DDPGAgent, CppPrioritizedReplayBuffer as PrioritizedReplayBuffer, ReplayBuffer, create_agent
from python_src.utils.helix_math import RunningMeanStd
from python_src.training.domain_randomizer import AutomaticDomainRandomizer, ProgressiveDomainRandomizer, ADRConfig


@dataclass
class TrainConfig:
    agent_type: str = 'td3'
    total_timesteps: int = 5_000_000
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 25_000
    train_freq: int = 2
    gradient_steps: int = 2
    
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    hidden_dim: int = 512
    
    policy_noise: float = 0.15
    noise_clip: float = 0.4
    policy_delay: int = 2
    
    exploration_noise_start: float = 0.25
    exploration_noise_end: float = 0.05
    exploration_warmup_steps: int = 50000
    exploration_noise_decay: float = 0.99998
    exploration_noise_min: float = 0.05
    
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    
    use_obs_normalization: bool = True
    obs_norm_clip: float = 10.0
    obs_norm_update_freq: int = 100
    frame_stack_size: int = 1
    
    eval_freq: int = 5000
    eval_episodes: int = 10
    save_freq: int = 25000
    log_freq: int = 1000
    
    seed: int = 42
    device: str = 'auto'
    checkpoint_dir: str = 'checkpoints'
    resume_from: Optional[str] = None
    
    num_envs: int = 1
    
                                          
    use_adr: bool = True
    adr_progressive: bool = True
    adr_update_freq: int = 50
    adr_success_threshold: float = 8.0
    
                   
    use_wandb: bool = True
    wandb_project: str = 'helixdrone'
    wandb_run_name: Optional[str] = None
    
                    
    use_frame_stack: bool = False
    n_frames: int = 4
    frame_include_delta: bool = True
    use_temporal_network: bool = False
    
                             
    use_asymmetric: bool = True
    asymmetric_include_wind: bool = True
    asymmetric_include_true_state: bool = True


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
    
    def step(self) -> float:
        self.steps += 1
        
        if self.steps < self.warmup_steps:
            progress = self.steps / self.warmup_steps
            self.current_noise = self.start_noise + (self.end_noise - self.start_noise) * progress
        else:
            self.warmup_complete = True
            self.current_noise = max(
                self.min_noise,
                self.current_noise * self.decay_rate
            )
        return self.current_noise

    def update_on_episode(self, crashed: bool = False):
        self.recent_crashes.append(float(crashed))
        
        crash_rate = np.mean(self.recent_crashes) if self.recent_crashes else 0.0
        if crash_rate > 0.5 and self.steps > 1000:
            reduction = 1.0 - (crash_rate - 0.5) * 0.5
            self.current_noise *= max(0.5, reduction)
    
    @property
    def noise(self) -> float:
        return self.current_noise


class Trainer:
    def __init__(self, config: TrainConfig, env_config: Optional[EnvConfig] = None, reward_config: Optional[RewardConfig] = None):
        self.config = config
        self.env_config = env_config or EnvConfig()
        self.reward_config = reward_config
        
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
            base_env = VectorizedQuadrotorEnv(
                num_envs=self.config.num_envs,
                config=self.env_config,
                reward_config=self.reward_config,
                task=TaskType.HOVER
            )
            self.is_vectorized = True
        else:
            base_env = QuadrotorEnv(config=self.env_config, reward_config=self.reward_config, task=TaskType.HOVER)
            self.is_vectorized = False
        
        if self.config.frame_stack_size > 1:
            base_env = FrameStack(base_env, num_stack=self.config.frame_stack_size)
        
        if self.config.use_adr:
            adr_config = ADRConfig(
                enabled=True,
                update_frequency=self.config.adr_update_freq,
                success_reward_threshold=self.config.adr_success_threshold,
            )
            if self.config.adr_progressive:
                self.env = ProgressiveDomainRandomizer(base_env, adr_config)
            else:
                self.env = AutomaticDomainRandomizer(base_env, adr_config)
            self.adr = self.env
        else:
            self.env = base_env
            self.adr = None
        
        self.eval_env = QuadrotorEnv(config=self.env_config, reward_config=self.reward_config, task=TaskType.HOVER)
        if self.config.frame_stack_size > 1:
            self.eval_env = FrameStack(self.eval_env, num_stack=self.config.frame_stack_size)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
    
    def _setup_agent(self):
        kwargs = {
            'hidden_dim': self.config.hidden_dim,
            'lr_actor': self.config.lr_actor,
            'lr_critic': self.config.lr_critic,
            'gamma': self.config.gamma,
            'tau': self.config.tau,
            'policy_noise': self.config.policy_noise,
            'noise_clip': self.config.noise_clip,
            'policy_delay': self.config.policy_delay
        }
            
        self.agent = create_agent(
            agent_type=self.config.agent_type,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
            **kwargs
        )
    
    def _setup_buffer(self):
        kwargs = {}
            
        if self.config.use_per:
            self.buffer = PrioritizedReplayBuffer(
                capacity=self.config.buffer_size,
                device=self.device,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                alpha=self.config.per_alpha,
                beta_start=self.config.per_beta_start,
                beta_frames=self.config.total_timesteps,
                **kwargs
            )
        else:
            self.buffer = ReplayBuffer(
                capacity=self.config.buffer_size,
                device=self.device,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                **kwargs
            )
    
    def _setup_normalization(self):
        if self.config.use_obs_normalization:
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
        
                              
        if self.config.use_wandb and WANDB_AVAILABLE:
            run_name = self.config.wandb_run_name or f"td3_{time.strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config={
                    'agent': self.config.agent_type,
                    'total_timesteps': self.config.total_timesteps,
                    'lr_actor': self.config.lr_actor,
                    'lr_critic': self.config.lr_critic,
                    'batch_size': self.config.batch_size,
                    'use_adr': self.config.use_adr,
                    'use_per': self.config.use_per,
                    'obs_mode': self.env_config.observation_mode,
                }
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
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
        
        obs, info = self.env.reset(seed=self.config.seed)
        obs = self._normalize_obs(obs, update=True)
        
        episode_reward = 0.0 if not self.is_vectorized else np.zeros(self.config.num_envs)
        episode_length = 0 if not self.is_vectorized else np.zeros(self.config.num_envs, dtype=int)
        episode_crashes = np.zeros(self.config.num_envs, dtype=bool) if self.is_vectorized else False
        
        crash_stats = {'ground': 0, 'distance': 0, 'angle': 0, 'velocity': 0, 'truncated': 0, 'unknown': 0}
        physics_stats = {'rpm_sum': 0.0, 'speed_sum': 0.0, 'angle_sum': 0.0, 'action_sum': 0.0, 'samples': 0}
        last_crash_info = {'reason': None, 'rpm': 0, 'speed': 0, 'roll': 0, 'pitch': 0, 'distance': 0}
        diag_interval = 50000
        last_diag_step = 0
        
        while self.timesteps < self.target_timesteps:
            if self.timesteps < self.config.learning_starts:
                if self.is_vectorized:
                    action = np.zeros((self.config.num_envs, 4))
                    for i in range(self.config.num_envs):
                        thrust_phase = (self.timesteps + i * 1000) % 5000
                        if thrust_phase < 1000:
                            base_thrust = 0.3
                        elif thrust_phase < 2000:
                            base_thrust = 0.0
                        elif thrust_phase < 3000:
                            base_thrust = -0.2
                        elif thrust_phase < 4000:
                            base_thrust = 0.5
                        else:
                            base_thrust = np.random.uniform(-0.3, 0.5)
                        action[i, 0] = base_thrust + np.random.uniform(-0.1, 0.1)
                        action[i, 1:] = np.random.uniform(-0.3, 0.3, 3)
                else:
                    thrust_phase = self.timesteps % 5000
                    if thrust_phase < 1000:
                        base_thrust = 0.3
                    elif thrust_phase < 2000:
                        base_thrust = 0.0
                    elif thrust_phase < 3000:
                        base_thrust = -0.2
                    elif thrust_phase < 4000:
                        base_thrust = 0.5
                    else:
                        base_thrust = np.random.uniform(-0.3, 0.5)
                    action = np.array([
                        base_thrust + np.random.uniform(-0.1, 0.1),
                        np.random.uniform(-0.3, 0.3),
                        np.random.uniform(-0.3, 0.3),
                        np.random.uniform(-0.3, 0.3)
                    ])
            else:
                noise_scale = self.exploration.noise
                if self.is_vectorized:
                    action = self.agent.get_actions_batch(obs, add_noise=True, noise_scale=noise_scale)
                else:
                    action = self._get_action_with_noise(obs, noise_scale)
            
            next_obs_raw, reward, terminated, truncated, info = self.env.step(action)
            next_obs = self._normalize_obs(next_obs_raw, update=True)
            
            if self.is_vectorized:
                self.buffer.push_batch(
                    states=obs if obs.ndim == 2 else obs.reshape(1, -1),
                    actions=action if action.ndim == 2 else action.reshape(1, -1),
                    rewards=reward if isinstance(reward, np.ndarray) else np.array([reward]),
                    next_states=next_obs if next_obs.ndim == 2 else next_obs.reshape(1, -1),
                    dones=terminated if isinstance(terminated, np.ndarray) else np.array([terminated])
                )
                
                for i in range(self.config.num_envs):
                    episode_reward[i] += reward[i]
                    episode_length[i] += 1
                    
                    env_info = info['envs'][i] if 'envs' in info else info
                    if env_info:
                        physics_stats['rpm_sum'] += env_info.get('rpm_mean', 0)
                        physics_stats['speed_sum'] += env_info.get('speed', 0)
                        physics_stats['angle_sum'] += abs(env_info.get('roll', 0)) + abs(env_info.get('pitch', 0))
                        physics_stats['action_sum'] += np.mean(action[i]) if action.ndim > 1 else np.mean(action)
                        physics_stats['samples'] += 1
                        
                        motor_rpms = env_info.get('rpm', [0, 0, 0, 0])
                        if len(motor_rpms) == 4:
                            physics_stats['m0_sum'] = physics_stats.get('m0_sum', 0) + motor_rpms[0]
                            physics_stats['m1_sum'] = physics_stats.get('m1_sum', 0) + motor_rpms[1]
                            physics_stats['m2_sum'] = physics_stats.get('m2_sum', 0) + motor_rpms[2]
                            physics_stats['m3_sum'] = physics_stats.get('m3_sum', 0) + motor_rpms[3]
                            max_diff = max(motor_rpms) - min(motor_rpms)
                            physics_stats['motor_diff_sum'] = physics_stats.get('motor_diff_sum', 0) + max_diff
                        
                        act = action[i] if action.ndim > 1 else action
                        physics_stats['a0_sum'] = physics_stats.get('a0_sum', 0) + act[0]
                        physics_stats['a1_sum'] = physics_stats.get('a1_sum', 0) + act[1]
                        physics_stats['a2_sum'] = physics_stats.get('a2_sum', 0) + act[2]
                        physics_stats['a3_sum'] = physics_stats.get('a3_sum', 0) + act[3]
                    
                    if terminated[i] or truncated[i]:
                        crashed = terminated[i] and episode_length[i] < 100
                        self.exploration.update_on_episode(crashed=crashed)
                        self.stats.add(episode_reward[i], episode_length[i])
                        history['rewards'].append(episode_reward[i])
                        self.episodes += 1
                        
                        if truncated[i] and not terminated[i]:
                            crash_stats['truncated'] += 1
                        elif env_info and env_info.get('crash_reason'):
                            reason = env_info['crash_reason']
                            crash_stats[reason] = crash_stats.get(reason, 0) + 1
                            last_crash_info = {
                                'reason': reason,
                                'rpm': env_info.get('rpm_mean', 0),
                                'speed': env_info.get('speed', 0),
                                'roll': np.degrees(env_info.get('roll', 0)),
                                'pitch': np.degrees(env_info.get('pitch', 0)),
                                'distance': env_info.get('target_error', 0),
                                'voltage': env_info.get('battery_voltage', 0),
                            }
                        else:
                            crash_stats['unknown'] += 1
                        
                        episode_reward[i] = 0.0
                        episode_length[i] = 0
                
                self.timesteps += self.config.num_envs
                for _ in range(self.config.num_envs):
                    self.exploration.step()
            else:
                done = terminated or truncated
                
                self.buffer.push(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    done=terminated
                )
                episode_reward += reward
                episode_length += 1
                self.timesteps += 1
                self.exploration.step()
                
                if done:
                    if self.adr:
                        self.adr.end_episode(episode_reward)
                    
                    crashed = terminated and episode_length < 100
                    self.exploration.update_on_episode(crashed=crashed)
                    self.stats.add(episode_reward, episode_length)
                    history['rewards'].append(episode_reward)
                    self.episodes += 1
                    
                    episode_reward = 0.0
                    episode_length = 0
                    obs_raw, info = self.env.reset()
                    obs = self._normalize_obs(obs_raw, update=True)
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
            
            if self.timesteps % self.config.log_freq == 0:
                self._log_progress()
            
            if self.timesteps - last_diag_step >= diag_interval:
                last_diag_step = self.timesteps
                total_crashes = sum(crash_stats.values())
                print("\n" + "=" * 80)
                print(f"[DIAGNOSTIC @ Step {self.timesteps}] WHY IS THE DRONE FAILING?")
                print("=" * 80)
                
                if total_crashes > 0:
                    print("\n[CRASH REASONS]")
                    for reason, count in sorted(crash_stats.items(), key=lambda x: -x[1]):
                        pct = count / total_crashes * 100
                        bar = "#" * int(pct / 2)
                        print(f"  {reason:12s}: {count:4d} ({pct:5.1f}%) {bar}")
                    
                    most_common = max(crash_stats.items(), key=lambda x: x[1])
                    print(f"\n  >>> MAIN PROBLEM: {most_common[0].upper()} ({most_common[1]} crashes, {most_common[1]/total_crashes*100:.1f}%)")
                
                if physics_stats['samples'] > 0:
                    avg_rpm = physics_stats['rpm_sum'] / physics_stats['samples']
                    avg_speed = physics_stats['speed_sum'] / physics_stats['samples']
                    avg_angle = np.degrees(physics_stats['angle_sum'] / physics_stats['samples'])
                    avg_action = physics_stats['action_sum'] / physics_stats['samples']
                    print(f"\n[AVERAGE PHYSICS STATE]")
                    print(f"  Avg RPM: {avg_rpm:.0f} (hover_rpm: {self.env_config.hover_rpm})")
                    print(f"  Avg Speed: {avg_speed:.2f} m/s")
                    print(f"  Avg Total Angle: {avg_angle:.1f} degrees")
                    print(f"  Avg Action: {avg_action:+.3f} (0=hover, -1=min, +1=max)")
                    
                    n = physics_stats['samples']
                    if physics_stats.get('m0_sum') is not None:
                        m0 = physics_stats['m0_sum'] / n
                        m1 = physics_stats['m1_sum'] / n
                        m2 = physics_stats['m2_sum'] / n
                        m3 = physics_stats['m3_sum'] / n
                        avg_diff = physics_stats.get('motor_diff_sum', 0) / n
                        print(f"\n[MOTOR RPM ANALYSIS]")
                        print(f"  M0: {m0:.0f}  |  M1: {m1:.0f}  |  M2: {m2:.0f}  |  M3: {m3:.0f}")
                        print(f"  Avg Max Diff: {avg_diff:.1f} RPM")
                        if avg_diff < 50:
                            print(f"  >>> WARNING: Motors are SYMMETRIC! (diff={avg_diff:.1f} < 50)")
                            print(f"      Drone cannot correct roll/pitch without motor differential!")
                        elif avg_diff < 150:
                            print(f"  >>> LOW: Motor differential is weak (diff={avg_diff:.1f})")
                        else:
                            print(f"  >>> GOOD: Motor differential is healthy (diff={avg_diff:.1f})")
                    
                    if physics_stats.get('a0_sum') is not None:
                        a0 = physics_stats['a0_sum'] / n
                        a1 = physics_stats['a1_sum'] / n
                        a2 = physics_stats['a2_sum'] / n
                        a3 = physics_stats['a3_sum'] / n
                        action_std = np.std([a0, a1, a2, a3])
                        print(f"\n[ACTOR OUTPUT ANALYSIS]")
                        print(f"  Thrust: {a0:+.3f}  |  Roll: {a1:+.3f}  |  Pitch: {a2:+.3f}  |  Yaw: {a3:+.3f}")
                        print(f"  Action Std: {action_std:.4f}")
                        if action_std < 0.02:
                            print(f"  >>> WARNING: Actor outputs are IDENTICAL! (std={action_std:.4f})")
                            print(f"      Actor is not learning to differentiate control channels!")
                        elif a0 < -0.1:
                            print(f"  >>> WARNING: Thrust is NEGATIVE ({a0:+.3f}) - drone wants to fall!")
                        elif abs(a1) > 0.3 or abs(a2) > 0.3:
                            print(f"  >>> WARNING: Roll/Pitch commands are extreme!")

                
                if last_crash_info['reason']:
                    print(f"\n[LAST CRASH DETAILS]")
                    print(f"  Reason: {last_crash_info['reason']}")
                    print(f"  RPM: {last_crash_info['rpm']:.0f}")
                    print(f"  Speed: {last_crash_info['speed']:.2f} m/s")
                    print(f"  Roll: {last_crash_info['roll']:.1f}°, Pitch: {last_crash_info['pitch']:.1f}°")
                    print(f"  Distance to target: {last_crash_info['distance']:.2f} m")
                    print(f"  Battery voltage: {last_crash_info['voltage']:.2f} V")
                    
                    if last_crash_info['reason'] == 'angle':
                        print("  >>> DIAGNOSIS: Drone is flipping! Check hover_rpm or control authority")
                    elif last_crash_info['reason'] == 'distance':
                        print("  >>> DIAGNOSIS: Drone drifts away! Check target placement or thrust")
                    elif last_crash_info['reason'] == 'ground':
                        print("  >>> DIAGNOSIS: Drone falls! Check if thrust < weight")
                    elif last_crash_info['reason'] == 'velocity':
                        print("  >>> DIAGNOSIS: Drone too fast! Reduce reward aggressiveness")
                
                if self.timesteps >= self.config.learning_starts:
                    import torch
                    test_obs = obs[0] if obs.ndim > 1 else obs
                    test_obs_t = torch.FloatTensor(test_obs).unsqueeze(0).to(self.device)
                    
                    neg_action = torch.FloatTensor([[-0.3, -0.3, -0.3, -0.3]]).to(self.device)
                    zero_action = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]]).to(self.device)
                    pos_action = torch.FloatTensor([[0.3, 0.3, 0.3, 0.3]]).to(self.device)
                    
                    with torch.no_grad():
                        q_neg = self.agent.critic.q1_forward(test_obs_t, neg_action).item()
                        q_zero = self.agent.critic.q1_forward(test_obs_t, zero_action).item()
                        q_pos = self.agent.critic.q1_forward(test_obs_t, pos_action).item()
                        actor_output = self.agent.actor(test_obs_t).mean().item()
                    
                    print(f"\n[Q-VALUE ANALYSIS]")
                    print(f"  Q(action=-0.3) = {q_neg:+.2f}")
                    print(f"  Q(action= 0.0) = {q_zero:+.2f}")
                    print(f"  Q(action=+0.3) = {q_pos:+.2f}")
                    print(f"  Actor output: {actor_output:+.3f}")
                    
                    if q_neg > q_pos + 0.5:
                        print(f"  >>> BUG: Critic values NEGATIVE actions {q_neg - q_pos:.1f} higher than positive!")
                    elif q_pos > q_neg + 0.5:
                        print(f"  >>> GOOD: Critic values POSITIVE actions {q_pos - q_neg:.1f} higher!")
                    else:
                        print(f"  >>> Critic values slightly differ (diff: {q_neg - q_pos:+.2f})")
                    
                    if len(history['critic_loss']) > 100:
                        recent_loss = np.mean(history['critic_loss'][-100:])
                        old_loss = np.mean(history['critic_loss'][-500:-400]) if len(history['critic_loss']) > 500 else recent_loss
                        loss_trend = recent_loss - old_loss
                        print(f"\n[LOSS TREND]")
                        print(f"  Critic loss (recent): {recent_loss:.2f}")
                        print(f"  Critic loss change: {loss_trend:+.2f}")
                        if loss_trend > 10:
                            print(f"  >>> WARNING: Critic loss INCREASING - possible divergence!")
                
                print("=" * 80 + "\n")
                
                crash_stats = {k: 0 for k in crash_stats}
                physics_stats = {'rpm_sum': 0.0, 'speed_sum': 0.0, 'angle_sum': 0.0, 'action_sum': 0.0, 'samples': 0}
            
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
        
        for _ in range(self.config.eval_episodes):
            obs_raw, _ = self.eval_env.reset()
            obs = self._normalize_obs(obs_raw, update=False)
            done = False
            
            while not done:
                action = self.agent.get_action(obs, add_noise=False)
                obs_raw, reward, terminated, truncated, _ = self.eval_env.step(action)
                obs = self._normalize_obs(obs_raw, update=False)
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
        
                   
        if self.adr is not None:
            adr_stats = self.adr.get_stats()
            difficulty = self.adr.get_difficulty_score()
            print(f"  ADR:        Difficulty: {difficulty:.2f}  |  Stage: {adr_stats.get('adr/current_stage', 0)}")
        
        print("=" * 70 + "\n")
        
                       
        if self.use_wandb:
            metrics = {
                'train/mean_reward': self.stats.mean_reward,
                'train/max_reward': max_reward,
                'train/min_reward': min_reward,
                'train/avg_length': avg_length,
                'train/crash_rate': crash_rate,
                'train/stable_rate': stable_rate,
                'train/success_rate': self.stats.success_rate * 100,
                'train/noise': self.exploration.noise,
                'train/timesteps': self.timesteps,
                'train/episodes': self.episodes,
            }
            
                         
            if self.adr is not None:
                metrics.update(self.adr.get_stats())
            
            wandb.log(metrics, step=self.timesteps)
    
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
        exploration_warmup_steps=int(cfg['training']['total_timesteps'] * 0.1),                   
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
        num_envs=cfg['environment']['num_envs']
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
        
        observation_mode='true_state',
        use_eskf=False,
        
        mass=float(env_cfg.get('mass', 0.6)),
        max_rpm=float(env_cfg.get('max_rpm', 35000.0)),
        min_rpm=float(env_cfg.get('min_rpm', 2000.0)),
        hover_rpm=float(env_cfg.get('hover_rpm', 2750.0)),
        rpm_range=float(env_cfg.get('rpm_range', 2000.0)),
        
        curriculum_enabled=curr_cfg.get('enabled', True),
        curriculum_init_range=float(curr_cfg.get('init_range', 0.1)),
        curriculum_max_range=float(curr_cfg.get('max_range', 0.5)),
        curriculum_progress_rate=float(curr_cfg.get('progress_rate', 0.00005)),
    )

    reward_config = RewardConfig(
        position_exp_weight=float(rew_cfg.get('position', 2.0)),
        position_exp_decay=4.0,
        alive_bonus=float(rew_cfg.get('alive', 0.5)),
        progress_weight=5.0,
        action_rate_weight=float(rew_cfg.get('action_rate', -0.1)),
        action_magnitude_weight=float(rew_cfg.get('action', -0.01)),
        angular_velocity_weight=float(rew_cfg.get('angular', -0.5)),
        orientation_weight=float(rew_cfg.get('orientation', -2.0)),
        stability_weight=float(rew_cfg.get('stability', -0.5)),
        crash_penalty=float(rew_cfg.get('crash', -100.0)),
        success_bonus=float(rew_cfg.get('success', 100.0)),
        crash_height=float(term_cfg.get('crash_height', 0.05)),
        crash_distance=float(term_cfg.get('crash_distance', 3.0)),
        crash_angle=float(term_cfg.get('crash_angle', 0.5)),
        success_distance=float(term_cfg.get('success_distance', 0.1)),
        success_velocity=float(term_cfg.get('success_velocity', 0.2)),
    )
    
    trainer = Trainer(train_config, env_config, reward_config)
    
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