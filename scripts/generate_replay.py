import sys
import os
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import argparse
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'Release'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'build'))

from python_src.envs.drone_env import QuadrotorEnv, EnvConfig, TaskType
from python_src.agents import TD3Agent, create_agent
from python_src.utils.csv_logger import CSVLogger, CSVLoggerConfig
from python_src.utils.visualization import DroneVisualizer, TrajectoryData, VisualizationConfig
from python_src.utils.helix_math import RunningMeanStd

@dataclass
class ReplayConfig:
    checkpoint_path: str = 'checkpoints/final'
    agent_type: str = 'td3'
    output_dir: str = 'replays'
    max_steps: int = 1500
    seed: int = 42
    unity_format: bool = True
    unity_scale: float = 1.0
    generate_gif: bool = True
    generate_plots: bool = True
    device: str = 'auto'
    obs_norm_clip: float = 10.0
    trajectory_type: str = 'figure8'

class DynamicTarget:
    def __init__(self, mode: str = 'figure8', center: np.ndarray = np.array([0, 0, 2.0])):
        self.mode = mode
        self.center = center
        self.t = 0.0
        
    def update(self, dt: float) -> np.ndarray:
        self.t += dt
        if self.mode == 'hover':
            return self.center.copy()
        elif self.mode == 'figure8':
            x = 2.0 * np.sin(0.5 * self.t)
            y = 2.0 * np.sin(1.0 * self.t)
            z = self.center[2] + 0.5 * np.sin(0.2 * self.t)
            return np.array([x, y, z])
        elif self.mode == 'circle':
            radius = 2.5
            x = radius * np.cos(0.5 * self.t)
            y = radius * np.sin(0.5 * self.t)
            z = self.center[2]
            return np.array([x, y, z])
        elif self.mode == 'vertical':
            x = 0.0
            y = 0.0
            z = self.center[2] + 1.5 * np.sin(0.8 * self.t)
            return np.array([x, y, z])
        return self.center.copy()

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
            if (checkpoint_path / 'td3_checkpoint.pt').exists():
                self.agent.load(checkpoint_path)
            elif checkpoint_path.is_file():
                self.agent.load(checkpoint_path.parent)
            else:
                try:
                    self.agent.load(checkpoint_path)
                except:
                    print(f"Warning: Could not load checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint path {checkpoint_path} does not exist")

    def _load_normalizer(self):
        self.obs_normalizer = None
        checkpoint_path = Path(self.config.checkpoint_path)
        possible_paths = [
            checkpoint_path / 'obs_normalizer.npz',
            checkpoint_path.parent / 'obs_normalizer.npz',
            Path('checkpoints') / 'obs_normalizer.npz'
        ]
        for path in possible_paths:
            if path.exists():
                self.obs_normalizer = RunningMeanStd(shape=(self.state_dim,))
                data = np.load(path)
                self.obs_normalizer.mean = data['mean']
                self.obs_normalizer.var = data['var']
                self.obs_normalizer.count = float(data['count'])
                break
                
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
            print(f"Generating SOTA replay {ep + 1}/{num_episodes}...")
            self._generate_episode(ep)
            
    def _generate_episode(self, episode_id: int) -> str:
        obs_raw, info = self.env.reset(seed=self.config.seed + episode_id)
        
        target_generator = DynamicTarget(mode=self.config.trajectory_type)
        self.env.target = target_generator.update(0.0)
        
        obs = self._normalize_obs(obs_raw)
        
        self.logger.start_episode(episode_id)
        
        data = {
            'pos': [], 'ori': [], 'vel': [], 'rpm': [], 
            'rew': [], 'target': []
        }
        
        total_reward = 0.0
        
        for step in range(self.config.max_steps):
            current_target = target_generator.update(self.env_config.dt)
            self.env.target = current_target
            
            action = self.agent.get_action(obs, add_noise=False)
            
            state = self.env.get_drone_state()
            self.logger.log_step(state, action, 0.0, step * self.env_config.dt)
            
            next_obs_raw, reward, terminated, truncated, info = self.env.step(action)
            obs = self._normalize_obs(next_obs_raw)
            total_reward += reward
            
            data['pos'].append([state.position.x, state.position.y, state.position.z])
            euler = state.orientation.to_euler_zyx()
            data['ori'].append([euler.x, euler.y, euler.z])
            data['vel'].append([state.velocity.x, state.velocity.y, state.velocity.z])
            data['rpm'].append(list(state.motor_rpm))
            data['rew'].append(reward)
            data['target'].append(current_target.copy())
            
            if terminated or truncated:
                break
                
        csv_path = self.logger.save(f'unity_replay_ep{episode_id}.csv')
        
        if self.config.generate_plots or self.config.generate_gif:
            traj_data = TrajectoryData(
                positions=np.array(data['pos']),
                orientations=np.array(data['ori']),
                velocities=np.array(data['vel']),
                motor_rpms=np.array(data['rpm']),
                timestamps=np.arange(len(data['pos'])) * self.env_config.dt,
                rewards=np.array(data['rew']),
                target=np.array(data['target'])
            )
            self._generate_visualizations(traj_data, episode_id)
            
        print(f"Replay generated: {len(data['pos'])} frames, Reward: {total_reward:.2f}")
        return csv_path
        
    def _generate_visualizations(self, data: TrajectoryData, episode_id: int):
        try:
            viz_config = VisualizationConfig(
                figsize=(12, 10),
                dpi=100,
                fps=30,
                trail_length=150
            )
            visualizer = DroneVisualizer(viz_config)
            
            if self.config.generate_plots:
                visualizer.plot_trajectory_3d(data, self.output_dir / f'trajectory_3d_ep{episode_id}.png')
                visualizer.plot_state_history(data, self.output_dir / f'state_history_ep{episode_id}.png')
                
            if self.config.generate_gif:
                visualizer.create_animation(data, self.output_dir / f'animation_ep{episode_id}.gif')
                
        except Exception as e:
            print(f"Viz Error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final')
    parser.add_argument('--output', type=str, default='replays')
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--mode', type=str, default='figure8')
    args = parser.parse_args()
    
    config_path = os.path.join(ROOT_DIR, 'config', 'train_params.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    env_cfg = cfg['environment']
    
    env_config = EnvConfig(
        dt=env_cfg['dt'],
        max_steps=args.steps,
        domain_randomization=False,
        wind_enabled=env_cfg['wind_enabled'],
        motor_dynamics=env_cfg['motor_dynamics'],
        physics_sub_steps=env_cfg.get('physics_sub_steps', 8),
        use_sub_stepping=env_cfg.get('use_sub_stepping', True),
        mass=float(env_cfg.get('mass', 0.6)),
        max_rpm=float(env_cfg.get('max_rpm', 35000.0)),
        min_rpm=float(env_cfg.get('min_rpm', 3000.0)),
        hover_rpm=float(env_cfg.get('hover_rpm', 5500.0)),
        rpm_range=float(env_cfg.get('rpm_range', 15000.0)),
        use_sota_actuator=env_cfg.get('use_sota_actuator', False)
    )
    
    config = ReplayConfig(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        max_steps=args.steps,
        trajectory_type=args.mode
    )
    
    generator = ReplayGenerator(config, env_config)
    generator.generate()

if __name__ == "__main__":
    main()
