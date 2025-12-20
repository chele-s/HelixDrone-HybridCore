import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'Release'))

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
import json
import time

from python_src.envs.drone_env import QuadrotorEnv, EnvConfig, TaskType
from python_src.agents import TD3Agent, DDPGAgent, create_agent


@dataclass
class EvalConfig:
    checkpoint_path: str = 'checkpoints/best'
    agent_type: str = 'td3'
    num_episodes: int = 100
    max_steps: int = 500
    seed: int = 0
    deterministic: bool = True
    render: bool = False
    save_trajectories: bool = False
    output_dir: str = 'eval_results'
    device: str = 'auto'


@dataclass
class EpisodeResult:
    reward: float
    length: int
    success: bool
    final_distance: float
    mean_distance: float
    max_distance: float
    rmse_position: float
    crash: bool
    trajectory: Optional[np.ndarray] = None


@dataclass
class EvalResults:
    num_episodes: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    median_reward: float
    
    mean_length: float
    std_length: float
    
    success_rate: float
    crash_rate: float
    
    mean_final_distance: float
    mean_rmse: float
    
    episode_results: List[EpisodeResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_episodes': self.num_episodes,
            'mean_reward': self.mean_reward,
            'std_reward': self.std_reward,
            'min_reward': self.min_reward,
            'max_reward': self.max_reward,
            'median_reward': self.median_reward,
            'mean_length': self.mean_length,
            'std_length': self.std_length,
            'success_rate': self.success_rate,
            'crash_rate': self.crash_rate,
            'mean_final_distance': self.mean_final_distance,
            'mean_rmse': self.mean_rmse
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            f"Episodes:           {self.num_episodes}",
            f"Mean Reward:        {self.mean_reward:.2f} ± {self.std_reward:.2f}",
            f"Min/Max Reward:     {self.min_reward:.2f} / {self.max_reward:.2f}",
            f"Median Reward:      {self.median_reward:.2f}",
            "-" * 60,
            f"Mean Episode Length: {self.mean_length:.1f} ± {self.std_length:.1f}",
            f"Success Rate:        {self.success_rate * 100:.1f}%",
            f"Crash Rate:          {self.crash_rate * 100:.1f}%",
            "-" * 60,
            f"Mean Final Distance: {self.mean_final_distance:.4f} m",
            f"Mean RMSE Position:  {self.mean_rmse:.4f} m",
            "=" * 60
        ]
        return "\n".join(lines)


class Evaluator:
    def __init__(self, config: EvalConfig, env_config: Optional[EnvConfig] = None):
        self.config = config
        self.env_config = env_config or EnvConfig(domain_randomization=False)
        
        self._setup_device()
        self._setup_env()
        self._load_agent()
    
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
            device=self.device
        )
        
        checkpoint_path = Path(self.config.checkpoint_path)
        if checkpoint_path.exists():
            self.agent.load(checkpoint_path)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    def evaluate(self) -> EvalResults:
        episode_results: List[EpisodeResult] = []
        
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        for ep in range(self.config.num_episodes):
            result = self._run_episode(ep)
            episode_results.append(result)
            
            status = "SUCCESS" if result.success else ("CRASH" if result.crash else "TIMEOUT")
            print(f"Episode {ep+1}/{self.config.num_episodes} | "
                  f"Reward: {result.reward:>8.2f} | "
                  f"Length: {result.length:>4} | "
                  f"Status: {status}")
        
        return self._compute_statistics(episode_results)
    
    def _run_episode(self, episode_id: int) -> EpisodeResult:
        obs, info = self.env.reset(seed=self.config.seed + episode_id)
        target = self.env.target.copy()
        
        total_reward = 0.0
        positions = []
        distances = []
        
        for step in range(self.config.max_steps):
            action = self.agent.get_action(obs, add_noise=not self.config.deterministic)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            total_reward += reward
            positions.append(info['position'].copy())
            distances.append(info['distance'])
            
            if self.config.render:
                self.env.render()
            
            if terminated or truncated:
                break
        
        positions = np.array(positions)
        distances = np.array(distances)
        
        final_distance = distances[-1] if len(distances) > 0 else float('inf')
        success = info.get('success_counter', 0) >= 50
        crash = terminated and not success
        
        target_pos = np.tile(target, (len(positions), 1))
        rmse = np.sqrt(np.mean(np.sum((positions - target_pos) ** 2, axis=1)))
        
        trajectory = positions if self.config.save_trajectories else None
        
        return EpisodeResult(
            reward=total_reward,
            length=step + 1,
            success=success,
            final_distance=final_distance,
            mean_distance=np.mean(distances),
            max_distance=np.max(distances),
            rmse_position=rmse,
            crash=crash,
            trajectory=trajectory
        )
    
    def _compute_statistics(self, results: List[EpisodeResult]) -> EvalResults:
        rewards = [r.reward for r in results]
        lengths = [r.length for r in results]
        successes = [r.success for r in results]
        crashes = [r.crash for r in results]
        final_distances = [r.final_distance for r in results]
        rmses = [r.rmse_position for r in results]
        
        return EvalResults(
            num_episodes=len(results),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            min_reward=float(np.min(rewards)),
            max_reward=float(np.max(rewards)),
            median_reward=float(np.median(rewards)),
            mean_length=float(np.mean(lengths)),
            std_length=float(np.std(lengths)),
            success_rate=float(np.mean(successes)),
            crash_rate=float(np.mean(crashes)),
            mean_final_distance=float(np.mean(final_distances)),
            mean_rmse=float(np.mean(rmses)),
            episode_results=results
        )
    
    def save_results(self, results: EvalResults, filename: Optional[str] = None):
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'eval_{timestamp}.json'
        
        filepath = output_dir / filename
        
        data = results.to_dict()
        data['config'] = {
            'checkpoint_path': self.config.checkpoint_path,
            'agent_type': self.config.agent_type,
            'num_episodes': self.config.num_episodes,
            'seed': self.config.seed,
            'deterministic': self.config.deterministic
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best')
    parser.add_argument('--agent', type=str, default='td3')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-trajectories', action='store_true')
    args = parser.parse_args()
    
    config = EvalConfig(
        checkpoint_path=args.checkpoint,
        agent_type=args.agent,
        num_episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        save_trajectories=args.save_trajectories
    )
    
    env_config = EnvConfig(
        domain_randomization=False,
        wind_enabled=False,
        motor_dynamics=True
    )
    
    evaluator = Evaluator(config, env_config)
    results = evaluator.evaluate()
    
    print(results)
    
    filepath = evaluator.save_results(results)
    print(f"\nResults saved to: {filepath}")


if __name__ == '__main__':
    main()
