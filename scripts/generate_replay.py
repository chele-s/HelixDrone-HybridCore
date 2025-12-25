import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'Release'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'build'))

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import argparse

from python_src.envs.drone_env import QuadrotorEnv, EnvConfig, TaskType
from python_src.agents import TD3Agent, DDPGAgent, create_agent
from python_src.utils.csv_logger import CSVLogger, CSVLoggerConfig
from python_src.utils.visualization import DroneVisualizer, TrajectoryData, VisualizationConfig
from python_src.utils.helix_math import RunningMeanStd


@dataclass
class ReplayConfig:
    checkpoint_path: str = 'checkpoints/best'
    agent_type: str = 'td3'
    output_dir: str = 'replays'
    max_steps: int = 500
    seed: int = 42
    unity_format: bool = True
    unity_scale: float = 1.0
    generate_gif: bool = True
    generate_plots: bool = True
    device: str = 'auto'
    obs_norm_clip: float = 10.0


class ReplayGenerator:
    def __init__(self, config: ReplayConfig, env_config: Optional[EnvConfig] = None):
        self.config = config
        self.env_config = env_config or EnvConfig(domain_randomization=False)
        
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_device()
        self._setup_env()
        self._load_agent()
        self._load_normalizer()
        self._setup_logger()
    
    def _setup_device(self):
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
    
    def _setup_env(self):
        self.env = QuadrotorEnv(config=self.env_config, task=TaskType.HOVER)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
    
    def _load_agent(self):
        self.agent = create_agent(
            agent_type=self.config.agent_type,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
            hidden_dim=512
        )
        
        checkpoint_path = Path(self.config.checkpoint_path)
        if checkpoint_path.exists():
            self.agent.load(checkpoint_path)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    def _load_normalizer(self):
        self.obs_normalizer = None
        checkpoint_path = Path(self.config.checkpoint_path)
        
        normalizer_paths = [
            checkpoint_path.parent / 'obs_normalizer.npz',
            checkpoint_path / 'obs_normalizer.npz',
            Path('checkpoints') / 'obs_normalizer.npz',
        ]
        
        for norm_path in normalizer_paths:
            if norm_path.exists():
                self.obs_normalizer = RunningMeanStd(shape=(self.state_dim,))
                data = np.load(norm_path)
                self.obs_normalizer.mean = data['mean']
                self.obs_normalizer.var = data['var']
                self.obs_normalizer.count = float(data['count'])
                print(f"  Loaded observation normalizer from: {norm_path}")
                break
        
        if self.obs_normalizer is None:
            print("  Warning: No observation normalizer found, using raw observations")
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_normalizer is None:
            return obs
        return self.obs_normalizer.normalize(obs, clip=self.config.obs_norm_clip)
    
    def _setup_logger(self):
        logger_config = CSVLoggerConfig(
            output_dir=str(self.output_dir),
            prefix='unity_replay',
            unity_format=self.config.unity_format,
            unity_scale=self.config.unity_scale,
            include_actions=True,
            include_rewards=True
        )
        self.logger = CSVLogger(logger_config)
    
    def generate(self, num_episodes: int = 1) -> None:
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        for ep in range(num_episodes):
            print(f"Generating replay {ep + 1}/{num_episodes}...")
            self._generate_episode(ep)
    
    def _generate_episode(self, episode_id: int) -> str:
        obs_raw, info = self.env.reset(seed=self.config.seed + episode_id)
        obs = self._normalize_obs(obs_raw)
        target = self.env.target.copy()
        
        self.logger.start_episode(episode_id)
        
        positions = []
        orientations = []
        velocities = []
        motor_rpms = []
        rewards = []
        
        total_reward = 0.0
        
        for step in range(self.config.max_steps):
            action = self.agent.get_action(obs, add_noise=False)
            
            state = self.env.get_drone_state()
            self.logger.log_step(state, action, 0.0, step * 0.02)
            
            next_obs_raw, reward, terminated, truncated, info = self.env.step(action)
            obs = self._normalize_obs(next_obs_raw)
            total_reward += reward
            
            positions.append(info['position'].copy())
            
            euler = state.orientation.to_euler_zyx()
            orientations.append([euler.x, euler.y, euler.z])
            
            velocities.append([
                state.velocity.x, state.velocity.y, state.velocity.z
            ])
            
            motor_rpms.append(list(state.motor_rpm))
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        csv_path = self.logger.save(f'unity_replay_ep{episode_id}.csv')
        print(f"  CSV saved: {csv_path}")
        
        stats = self.logger.get_statistics()
        print(f"  Frames: {stats['num_frames']}, Duration: {stats['duration']:.2f}s, "
              f"Total Reward: {total_reward:.2f}")
        
        if self.config.generate_plots or self.config.generate_gif:
            trajectory_data = TrajectoryData(
                positions=np.array(positions),
                orientations=np.array(orientations),
                velocities=np.array(velocities),
                motor_rpms=np.array(motor_rpms),
                timestamps=np.arange(len(positions)) * 0.02,
                rewards=np.array(rewards),
                target=target
            )
            
            self._generate_visualizations(trajectory_data, episode_id)
        
        return csv_path
    
    def _generate_visualizations(self, data: TrajectoryData, episode_id: int):
        try:
            viz_config = VisualizationConfig(
                figsize=(12, 10),
                dpi=100,
                fps=30,
                trail_length=100
            )
            
            visualizer = DroneVisualizer(viz_config)
            
            if self.config.generate_plots:
                plot_path = self.output_dir / f'trajectory_3d_ep{episode_id}.png'
                visualizer.plot_trajectory_3d(data, save_path=str(plot_path))
                print(f"  3D plot saved: {plot_path}")
                
                plot_path_2d = self.output_dir / f'trajectory_2d_ep{episode_id}.png'
                visualizer.plot_trajectory_2d(data, save_path=str(plot_path_2d))
                print(f"  2D plots saved: {plot_path_2d}")
                
                state_path = self.output_dir / f'state_history_ep{episode_id}.png'
                visualizer.plot_state_history(data, save_path=str(state_path))
                print(f"  State history saved: {state_path}")
            
            if self.config.generate_gif:
                gif_path = self.output_dir / f'animation_ep{episode_id}.gif'
                visualizer.create_animation(data, save_path=str(gif_path), format='gif')
                print(f"  Animation saved: {gif_path}")
        
        except Exception as e:
            print(f"  Warning: Visualization failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best')
    parser.add_argument('--agent', type=str, default='td3')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--output', type=str, default='replays')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-gif', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--unity-scale', type=float, default=1.0)
    parser.add_argument('--obs-norm-clip', type=float, default=10.0)
    args = parser.parse_args()
    
    config = ReplayConfig(
        checkpoint_path=args.checkpoint,
        agent_type=args.agent,
        output_dir=args.output,
        seed=args.seed,
        generate_gif=not args.no_gif,
        generate_plots=not args.no_plots,
        unity_scale=args.unity_scale,
        obs_norm_clip=args.obs_norm_clip
    )
    
    env_config = EnvConfig(
        domain_randomization=False,
        wind_enabled=False,
        motor_dynamics=True
    )
    
    print("=" * 60)
    print("HelixDrone Replay Generator")
    print("=" * 60)
    
    generator = ReplayGenerator(config, env_config)
    generator.generate(num_episodes=args.episodes)
    
    print("=" * 60)
    print("Replay generation complete!")
    print(f"Output directory: {config.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
