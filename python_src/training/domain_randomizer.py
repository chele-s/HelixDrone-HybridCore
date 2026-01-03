import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from collections import deque


@dataclass
class ADRParameter:
    name: str
    base_value: float
    min_range: float = 0.0
    max_range: float = 1.0
    current_range: float = 0.0
    
    expansion_rate: float = 0.01
    contraction_rate: float = 0.005
    success_threshold: float = 0.7
    failure_threshold: float = 0.3
    
    min_bound: Optional[float] = None
    max_bound: Optional[float] = None
    
    def sample(self) -> float:
        if self.current_range <= 0:
            return self.base_value
        
        delta = np.random.uniform(-self.current_range, self.current_range)
        value = self.base_value + delta * (self.max_range - self.min_range)
        
        if self.min_bound is not None:
            value = max(self.min_bound, value)
        if self.max_bound is not None:
            value = min(self.max_bound, value)
            
        return value
    
    def expand(self):
        self.current_range = min(1.0, self.current_range + self.expansion_rate)
    
    def contract(self):
        self.current_range = max(0.0, self.current_range - self.contraction_rate)
    
    def get_range_fraction(self) -> float:
        return self.current_range


@dataclass
class ADRConfig:
    enabled: bool = True
    history_window: int = 100
    update_frequency: int = 50
    
    success_reward_threshold: float = 8.0
    failure_reward_threshold: float = 2.0
    
    mass_base: float = 0.6
    mass_range: float = 0.3
    
    motor_lag_base: float = 0.02
    motor_lag_range: float = 0.03
    
    wind_base: float = 0.5
    wind_range: float = 2.0
    
    friction_base: float = 0.1
    friction_range: float = 0.2
    
    inertia_scale_base: float = 1.0
    inertia_scale_range: float = 0.3


class AutomaticDomainRandomizer:
    def __init__(self, env, config: Optional[ADRConfig] = None):
        self.env = env
        self.config = config or ADRConfig()
        
        self.parameters: Dict[str, ADRParameter] = {}
        self._setup_parameters()
        
        self.episode_rewards: deque = deque(maxlen=self.config.history_window)
        self.episode_count = 0
        self.total_steps = 0
        
        self.current_params: Dict[str, float] = {}
        
    def _setup_parameters(self):
        cfg = self.config
        
        self.parameters['mass'] = ADRParameter(
            name='mass',
            base_value=cfg.mass_base,
            max_range=cfg.mass_range,
            min_bound=0.1,
            max_bound=2.0,
            expansion_rate=0.02,
            contraction_rate=0.01
        )
        
        self.parameters['motor_lag'] = ADRParameter(
            name='motor_lag',
            base_value=cfg.motor_lag_base,
            max_range=cfg.motor_lag_range,
            min_bound=0.001,
            max_bound=0.1,
            expansion_rate=0.015,
            contraction_rate=0.008
        )
        
        self.parameters['wind_intensity'] = ADRParameter(
            name='wind_intensity',
            base_value=cfg.wind_base,
            max_range=cfg.wind_range,
            min_bound=0.0,
            max_bound=5.0,
            expansion_rate=0.025,
            contraction_rate=0.01
        )
        
        self.parameters['friction'] = ADRParameter(
            name='friction',
            base_value=cfg.friction_base,
            max_range=cfg.friction_range,
            min_bound=0.0,
            max_bound=0.5,
            expansion_rate=0.01,
            contraction_rate=0.005
        )
        
        self.parameters['inertia_scale'] = ADRParameter(
            name='inertia_scale',
            base_value=cfg.inertia_scale_base,
            max_range=cfg.inertia_scale_range,
            min_bound=0.5,
            max_bound=2.0,
            expansion_rate=0.015,
            contraction_rate=0.008
        )
    
    def sample_parameters(self) -> Dict[str, float]:
        params = {}
        for name, param in self.parameters.items():
            params[name] = param.sample()
        return params
    
    def apply_parameters(self, params: Dict[str, float]):
        self.current_params = params
        
        if hasattr(self.env, 'set_physics_params'):
            self.env.set_physics_params(
                mass=params.get('mass'),
                motor_lag=params.get('motor_lag'),
                friction=params.get('friction'),
                inertia_scale=params.get('inertia_scale')
            )
        
        if hasattr(self.env, 'set_wind_intensity'):
            self.env.set_wind_intensity(params.get('wind_intensity', 0.5))
    
    def reset(self, **kwargs):
        if self.config.enabled:
            params = self.sample_parameters()
            self.apply_parameters(params)
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        self.total_steps += 1
        return self.env.step(action)
    
    def end_episode(self, episode_reward: float):
        self.episode_rewards.append(episode_reward)
        self.episode_count += 1
        
        if self.episode_count % self.config.update_frequency == 0:
            self._update_ranges()
    
    def _update_ranges(self):
        if len(self.episode_rewards) < 10:
            return
        
        recent_rewards = list(self.episode_rewards)[-50:]
        mean_reward = np.mean(recent_rewards)
        success_rate = np.mean([
            r > self.config.success_reward_threshold 
            for r in recent_rewards
        ])
        failure_rate = np.mean([
            r < self.config.failure_reward_threshold 
            for r in recent_rewards
        ])
        
        for param in self.parameters.values():
            if success_rate > param.success_threshold:
                param.expand()
            elif failure_rate > (1 - param.failure_threshold):
                param.contract()
    
    def get_stats(self) -> Dict[str, float]:
        stats = {
            'adr/episode_count': self.episode_count,
            'adr/total_steps': self.total_steps,
        }
        
        for name, param in self.parameters.items():
            stats[f'adr/{name}_range'] = param.get_range_fraction()
            stats[f'adr/{name}_current'] = self.current_params.get(name, param.base_value)
        
        if self.episode_rewards:
            stats['adr/mean_reward'] = np.mean(self.episode_rewards)
            stats['adr/success_rate'] = np.mean([
                r > self.config.success_reward_threshold 
                for r in self.episode_rewards
            ])
        
        return stats
    
    def get_difficulty_score(self) -> float:
        if not self.parameters:
            return 0.0
        return np.mean([p.get_range_fraction() for p in self.parameters.values()])
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class ProgressiveDomainRandomizer(AutomaticDomainRandomizer):
    def __init__(self, env, config: Optional[ADRConfig] = None):
        super().__init__(env, config)
        
        self.stages = [
            ['mass'],
            ['mass', 'motor_lag'],
            ['mass', 'motor_lag', 'wind_intensity'],
            ['mass', 'motor_lag', 'wind_intensity', 'friction'],
            ['mass', 'motor_lag', 'wind_intensity', 'friction', 'inertia_scale'],
        ]
        self.current_stage = 0
        self.stage_mastery_threshold = 0.8
        self.stage_episodes_required = 100
        self.stage_episode_count = 0
    
    def sample_parameters(self) -> Dict[str, float]:
        params = {}
        active_params = self.stages[self.current_stage] if self.current_stage < len(self.stages) else list(self.parameters.keys())
        
        for name, param in self.parameters.items():
            if name in active_params:
                params[name] = param.sample()
            else:
                params[name] = param.base_value
        
        return params
    
    def end_episode(self, episode_reward: float):
        super().end_episode(episode_reward)
        self.stage_episode_count += 1
        
        if self.stage_episode_count >= self.stage_episodes_required:
            self._check_stage_progression()
    
    def _check_stage_progression(self):
        if self.current_stage >= len(self.stages) - 1:
            return
        
        recent_rewards = list(self.episode_rewards)[-self.stage_episodes_required:]
        success_rate = np.mean([
            r > self.config.success_reward_threshold 
            for r in recent_rewards
        ])
        
        if success_rate >= self.stage_mastery_threshold:
            self.current_stage += 1
            self.stage_episode_count = 0
            print(f"[ADR] Advanced to stage {self.current_stage + 1}/{len(self.stages)}: {self.stages[self.current_stage]}")
    
    def get_stats(self) -> Dict[str, float]:
        stats = super().get_stats()
        stats['adr/current_stage'] = self.current_stage
        stats['adr/stage_progress'] = self.stage_episode_count / self.stage_episodes_required
        return stats
