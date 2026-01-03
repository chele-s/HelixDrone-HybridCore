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


@dataclass
class FrameStackConfig:
    n_frames: int = 4
    flatten: bool = True
    include_delta: bool = True
    delta_scale: float = 10.0


class FrameStackWrapper:
    def __init__(
        self,
        env,
        config: Optional[FrameStackConfig] = None
    ):
        self.env = env
        self.config = config or FrameStackConfig()
        
        self._base_obs_dim = env.observation_space.shape[0]
        self._frames: List[np.ndarray] = []
        self._prev_frame: Optional[np.ndarray] = None
        
        self._setup_observation_space()
    
    def _setup_observation_space(self):
        n = self.config.n_frames
        base_dim = self._base_obs_dim
        
        if self.config.include_delta:
            frame_dim = base_dim * 2
        else:
            frame_dim = base_dim
        
        if self.config.flatten:
            total_dim = frame_dim * n
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(total_dim,),
                dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(n, frame_dim),
                dtype=np.float32
            )
    
    def _process_frame(self, obs: np.ndarray) -> np.ndarray:
        if not self.config.include_delta:
            return obs.astype(np.float32)
        
        if self._prev_frame is None:
            delta = np.zeros_like(obs)
        else:
            delta = (obs - self._prev_frame) * self.config.delta_scale
        
        self._prev_frame = obs.copy()
        return np.concatenate([obs, delta]).astype(np.float32)
    
    def _stack_frames(self) -> np.ndarray:
        while len(self._frames) < self.config.n_frames:
            if self._frames:
                self._frames.insert(0, self._frames[0].copy())
            else:
                frame_dim = self._base_obs_dim * (2 if self.config.include_delta else 1)
                self._frames.append(np.zeros(frame_dim, dtype=np.float32))
        
        stacked = np.array(self._frames[-self.config.n_frames:])
        
        if self.config.flatten:
            return stacked.flatten()
        return stacked
    
    def reset(self, **kwargs):
        self._frames.clear()
        self._prev_frame = None
        
        obs, info = self.env.reset(**kwargs)
        processed = self._process_frame(obs)
        self._frames.append(processed)
        
        return self._stack_frames(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed = self._process_frame(obs)
        self._frames.append(processed)
        
        if len(self._frames) > self.config.n_frames:
            self._frames.pop(0)
        
        return self._stack_frames(), reward, terminated, truncated, info
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def n_frames(self) -> int:
        return self.config.n_frames
    
    @property
    def base_obs_dim(self) -> int:
        return self._base_obs_dim
    
    @property
    def frame_dim(self) -> int:
        return self._base_obs_dim * (2 if self.config.include_delta else 1)
    
    def __getattr__(self, name):
        return getattr(self.env, name)


@dataclass
class AsymmetricObsConfig:
    include_wind: bool = True
    include_motor_forces: bool = True
    include_true_state: bool = True
    include_payload_forces: bool = True


class AsymmetricObsWrapper:
    def __init__(
        self,
        env,
        config: Optional[AsymmetricObsConfig] = None
    ):
        self.env = env
        self.config = config or AsymmetricObsConfig()
        
        self._actor_obs_dim = env.observation_space.shape[0]
        self._critic_extra_dim = self._compute_extra_dim()
        
        self._setup_observation_spaces()
        
        self._last_actor_obs = None
        self._last_critic_obs = None
    
    def _compute_extra_dim(self) -> int:
        dim = 0
        if self.config.include_wind:
            dim += 3
        if self.config.include_motor_forces:
            dim += 4
        if self.config.include_true_state:
            dim += 13
        if self.config.include_payload_forces:
            dim += 3
        return dim
    
    def _setup_observation_spaces(self):
        self.actor_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._actor_obs_dim,),
            dtype=np.float32
        )
        
        self.critic_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._actor_obs_dim + self._critic_extra_dim,),
            dtype=np.float32
        )
        
        self.observation_space = self.actor_observation_space
    
    def _get_privileged_obs(self) -> np.ndarray:
        privileged = []
        
        if self.config.include_wind and hasattr(self.env, '_drone'):
            try:
                wind = self.env._drone.get_wind_velocity()
                privileged.extend([wind.x, wind.y, wind.z])
            except:
                privileged.extend([0.0, 0.0, 0.0])
        
        if self.config.include_motor_forces and hasattr(self.env, '_current_rpm'):
            rpm = self.env._current_rpm if hasattr(self.env, '_current_rpm') else np.zeros(4)
            normalized_rpm = rpm / 6000.0
            privileged.extend(normalized_rpm.tolist())
        
        if self.config.include_true_state and hasattr(self.env, '_drone'):
            state = self.env._drone.get_state()
            privileged.extend([
                state.position.x / 5.0,
                state.position.y / 5.0,
                state.position.z / 5.0,
                state.velocity.x / 5.0,
                state.velocity.y / 5.0,
                state.velocity.z / 5.0,
                state.orientation.w,
                state.orientation.x,
                state.orientation.y,
                state.orientation.z,
                state.angular_velocity.x / 10.0,
                state.angular_velocity.y / 10.0,
                state.angular_velocity.z / 10.0,
            ])
        
        if self.config.include_payload_forces and hasattr(self.env, '_payload') and self.env._payload is not None:
            try:
                forces = self.env._payload.get_coupling_forces()
                privileged.extend([
                    forces.cable_force.x / 10.0,
                    forces.cable_force.y / 10.0,
                    forces.cable_force.z / 10.0,
                ])
            except:
                privileged.extend([0.0, 0.0, 0.0])
        
        return np.array(privileged, dtype=np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        self._last_actor_obs = obs.astype(np.float32)
        privileged = self._get_privileged_obs()
        self._last_critic_obs = np.concatenate([obs, privileged]).astype(np.float32)
        
        info['critic_obs'] = self._last_critic_obs
        
        return self._last_actor_obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._last_actor_obs = obs.astype(np.float32)
        privileged = self._get_privileged_obs()
        self._last_critic_obs = np.concatenate([obs, privileged]).astype(np.float32)
        
        info['critic_obs'] = self._last_critic_obs
        
        return self._last_actor_obs, reward, terminated, truncated, info
    
    def get_actor_obs(self) -> np.ndarray:
        return self._last_actor_obs
    
    def get_critic_obs(self) -> np.ndarray:
        return self._last_critic_obs
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def actor_obs_dim(self) -> int:
        return self._actor_obs_dim
    
    @property
    def critic_obs_dim(self) -> int:
        return self._actor_obs_dim + self._critic_extra_dim
    
    def __getattr__(self, name):
        return getattr(self.env, name)
