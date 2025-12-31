"""Modular observation builder for drone environments."""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple
from enum import IntEnum
import gymnasium as gym

import drone_core


class ObservationMode(IntEnum):
    TRUE_STATE = 0
    ESTIMATED = 1
    SENSOR_ONLY = 2


@dataclass
class ObsConfig:
    mode: ObservationMode = ObservationMode.ESTIMATED
    
    position_scale: float = 5.0
    velocity_scale: float = 5.0
    angular_velocity_scale: float = 10.0
    
    include_target_error: bool = True
    include_target_velocity: bool = True
    include_prev_action: bool = True
    include_cable_angles: bool = False
    include_payload_state: bool = False
    
    observation_noise: float = 0.01
    
    cable_angle_scale: float = 1.0
    cable_rate_scale: float = 5.0


class ObservationComponent(Protocol):
    @property
    def dim(self) -> int: ...
    def observe(self, env_state: 'EnvState') -> np.ndarray: ...
    def reset(self) -> None: ...


@dataclass
class EnvState:
    drone_state: 'drone_core.State'
    eskf_state: Optional['drone_core.EKFState'] = None
    sensor_reading: Optional['drone_core.SensorReading'] = None
    payload_state: Optional['drone_core.PayloadState'] = None
    cable_reading: Optional['drone_core.CableSensorReading'] = None
    
    target_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    cable_angle_rate: np.ndarray = field(default_factory=lambda: np.zeros(2))


class PositionComponent:
    def __init__(self, config: ObsConfig, use_estimated: bool = False):
        self.config = config
        self.use_estimated = use_estimated
    
    @property
    def dim(self) -> int:
        return 3
    
    def observe(self, env_state: EnvState) -> np.ndarray:
        if self.use_estimated and env_state.eskf_state is not None:
            pos = env_state.eskf_state.position
        else:
            pos = env_state.drone_state.position
        return np.array([pos.x, pos.y, pos.z]) / self.config.position_scale
    
    def reset(self) -> None:
        pass


class VelocityComponent:
    def __init__(self, config: ObsConfig, use_estimated: bool = False):
        self.config = config
        self.use_estimated = use_estimated
    
    @property
    def dim(self) -> int:
        return 3
    
    def observe(self, env_state: EnvState) -> np.ndarray:
        if self.use_estimated and env_state.eskf_state is not None:
            vel = env_state.eskf_state.velocity
        else:
            vel = env_state.drone_state.velocity
        return np.array([vel.x, vel.y, vel.z]) / self.config.velocity_scale
    
    def reset(self) -> None:
        pass


class OrientationComponent:
    def __init__(self, config: ObsConfig, use_estimated: bool = False):
        self.config = config
        self.use_estimated = use_estimated
    
    @property
    def dim(self) -> int:
        return 4
    
    def observe(self, env_state: EnvState) -> np.ndarray:
        if self.use_estimated and env_state.eskf_state is not None:
            ori = env_state.eskf_state.orientation
        else:
            ori = env_state.drone_state.orientation
        return np.array([ori.w, ori.x, ori.y, ori.z])
    
    def reset(self) -> None:
        pass


class AngularVelocityComponent:
    def __init__(self, config: ObsConfig, use_estimated: bool = False):
        self.config = config
        self.use_estimated = use_estimated
    
    @property
    def dim(self) -> int:
        return 3
    
    def observe(self, env_state: EnvState) -> np.ndarray:
        ang_vel = env_state.drone_state.angular_velocity
        return np.array([ang_vel.x, ang_vel.y, ang_vel.z]) / self.config.angular_velocity_scale
    
    def reset(self) -> None:
        pass


class TargetErrorComponent:
    def __init__(self, config: ObsConfig, use_estimated: bool = False):
        self.config = config
        self.use_estimated = use_estimated
    
    @property
    def dim(self) -> int:
        return 3
    
    def observe(self, env_state: EnvState) -> np.ndarray:
        if self.use_estimated and env_state.eskf_state is not None:
            pos = env_state.eskf_state.position
            current = np.array([pos.x, pos.y, pos.z])
        else:
            pos = env_state.drone_state.position
            current = np.array([pos.x, pos.y, pos.z])
        
        error = env_state.target_position - current
        return error / self.config.position_scale
    
    def reset(self) -> None:
        pass


class TargetVelocityComponent:
    def __init__(self, config: ObsConfig, use_estimated: bool = False):
        self.config = config
        self.use_estimated = use_estimated
    
    @property
    def dim(self) -> int:
        return 3
    
    def observe(self, env_state: EnvState) -> np.ndarray:
        if self.use_estimated and env_state.eskf_state is not None:
            vel = env_state.eskf_state.velocity
            current = np.array([vel.x, vel.y, vel.z])
        else:
            vel = env_state.drone_state.velocity
            current = np.array([vel.x, vel.y, vel.z])
        
        relative = env_state.target_velocity - current
        return relative / self.config.velocity_scale
    
    def reset(self) -> None:
        pass


class PrevActionComponent:
    def __init__(self, config: ObsConfig):
        self.config = config
    
    @property
    def dim(self) -> int:
        return 4
    
    def observe(self, env_state: EnvState) -> np.ndarray:
        return env_state.prev_action.copy()
    
    def reset(self) -> None:
        pass


class CableAngleComponent:
    def __init__(self, config: ObsConfig):
        self.config = config
        self._prev_theta = np.zeros(2)
        self._prev_time = 0.0
    
    @property
    def dim(self) -> int:
        return 4
    
    def observe(self, env_state: EnvState) -> np.ndarray:
        if env_state.cable_reading is not None and env_state.cable_reading.valid:
            theta = np.array([
                env_state.cable_reading.theta_x,
                env_state.cable_reading.theta_y
            ])
            rate = env_state.cable_angle_rate
        else:
            theta = np.zeros(2)
            rate = np.zeros(2)
        
        return np.concatenate([
            theta / self.config.cable_angle_scale,
            rate / self.config.cable_rate_scale
        ])
    
    def reset(self) -> None:
        self._prev_theta = np.zeros(2)
        self._prev_time = 0.0


class PayloadStateComponent:
    def __init__(self, config: ObsConfig):
        self.config = config
    
    @property
    def dim(self) -> int:
        return 6
    
    def observe(self, env_state: EnvState) -> np.ndarray:
        if env_state.payload_state is not None:
            pos = env_state.payload_state.position
            vel = env_state.payload_state.velocity
            
            if env_state.eskf_state is not None:
                drone_pos = env_state.eskf_state.position
                drone_vel = env_state.eskf_state.velocity
            else:
                drone_pos = env_state.drone_state.position
                drone_vel = env_state.drone_state.velocity
            
            rel_pos = np.array([
                pos.x - drone_pos.x,
                pos.y - drone_pos.y,
                pos.z - drone_pos.z
            ]) / self.config.position_scale
            
            rel_vel = np.array([
                vel.x - drone_vel.x,
                vel.y - drone_vel.y,
                vel.z - drone_vel.z
            ]) / self.config.velocity_scale
            
            return np.concatenate([rel_pos, rel_vel])
        
        return np.zeros(6)
    
    def reset(self) -> None:
        pass


class IMUSensorComponent:
    def __init__(self, config: ObsConfig):
        self.config = config
    
    @property
    def dim(self) -> int:
        return 6
    
    def observe(self, env_state: EnvState) -> np.ndarray:
        if env_state.sensor_reading is not None:
            accel = env_state.sensor_reading.accelerometer
            gyro = env_state.sensor_reading.gyroscope
            return np.array([
                accel.x / 10.0, accel.y / 10.0, accel.z / 10.0,
                gyro.x / 5.0, gyro.y / 5.0, gyro.z / 5.0
            ])
        return np.zeros(6)
    
    def reset(self) -> None:
        pass


class ObservationBuilder:
    def __init__(self, config: ObsConfig):
        self.config = config
        self.components: List[ObservationComponent] = []
        self._rng = np.random.default_rng()
        self._setup_components()
    
    def _setup_components(self):
        use_estimated = self.config.mode == ObservationMode.ESTIMATED
        
        if self.config.mode == ObservationMode.SENSOR_ONLY:
            self.components.append(IMUSensorComponent(self.config))
        else:
            self.components.append(PositionComponent(self.config, use_estimated))
            self.components.append(VelocityComponent(self.config, use_estimated))
            self.components.append(OrientationComponent(self.config, use_estimated))
            self.components.append(AngularVelocityComponent(self.config, use_estimated))
        
        if self.config.include_target_error:
            self.components.append(TargetErrorComponent(self.config, use_estimated))
        
        if self.config.include_target_velocity:
            self.components.append(TargetVelocityComponent(self.config, use_estimated))
        
        if self.config.include_prev_action:
            self.components.append(PrevActionComponent(self.config))
        
        if self.config.include_cable_angles:
            self.components.append(CableAngleComponent(self.config))
        
        if self.config.include_payload_state:
            self.components.append(PayloadStateComponent(self.config))
    
    @property
    def observation_dim(self) -> int:
        return sum(c.dim for c in self.components)
    
    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )
    
    def build(self, env_state: EnvState) -> np.ndarray:
        obs_parts = [comp.observe(env_state) for comp in self.components]
        obs = np.concatenate(obs_parts).astype(np.float32)
        
        if self.config.observation_noise > 0:
            noise = self._rng.normal(0, self.config.observation_noise, size=obs.shape)
            obs = obs + noise.astype(np.float32)
        
        return obs
    
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        for comp in self.components:
            comp.reset()


def create_observation_builder(
    mode: str = 'estimated',
    include_payload: bool = False,
    include_cable: bool = False,
    **kwargs
) -> ObservationBuilder:
    mode_map = {
        'true_state': ObservationMode.TRUE_STATE,
        'estimated': ObservationMode.ESTIMATED,
        'sensor_only': ObservationMode.SENSOR_ONLY,
    }
    
    config = ObsConfig(
        mode=mode_map.get(mode, ObservationMode.ESTIMATED),
        include_cable_angles=include_cable,
        include_payload_state=include_payload,
        **kwargs
    )
    
    return ObservationBuilder(config)
