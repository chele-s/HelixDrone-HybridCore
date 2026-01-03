                                                                               
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'Release'))

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import IntEnum

import drone_core

from .observation_builder import (
    ObservationBuilder, ObsConfig, ObservationMode, EnvState, 
    create_observation_builder
)
from .reward_functions import RewardBuilder, RewardConfig, RewardState
from .motor_mixer import MotorMixer, MotorMixerConfig


class TaskType(IntEnum):
    HOVER = 0
    WAYPOINT = 1
    TRAJECTORY = 2
    VELOCITY = 3
    PAYLOAD_HOVER = 4
    PAYLOAD_TRANSPORT = 5


@dataclass
class ExtendedEnvConfig:
    dt: float = 0.01
    physics_sub_steps: int = 8
    max_steps: int = 1000
    max_rpm: float = 35000.0
    min_rpm: float = 1000.0
    hover_rpm: float = 2700.0
    rpm_range: float = 2500.0
    mass: float = 0.6
    
    position_scale: float = 5.0
    velocity_scale: float = 5.0
    angular_velocity_scale: float = 10.0
    
    observation_mode: str = 'estimated'
    use_eskf: bool = True
    
    payload_enabled: bool = False
    payload_mass: float = 0.5
    cable_length: float = 1.0
    cable_sensor_enabled: bool = True
    cable_angle_noise: float = 0.05
    
    collision_enabled: bool = False
    num_obstacles: int = 0
    
    domain_randomization: bool = True
    mass_randomization: Tuple[float, float] = (0.85, 1.15)
    wind_enabled: bool = True
    wind_speed_range: Tuple[float, float] = (0.0, 4.0)
    motor_dynamics: bool = True
    use_sub_stepping: bool = True
    use_motor_mixer: bool = False
    
    observation_noise: float = 0.01
    action_delay_steps: int = 0
    
    curriculum_enabled: bool = True
    curriculum_init_range: float = 0.02
    curriculum_max_range: float = 0.6
    curriculum_progress_rate: float = 0.00005
    
    target_curriculum_enabled: bool = True
    target_speed_init: float = 0.0
    target_speed_max: float = 2.5
    target_speed_curriculum_rate: float = 0.00003


class ActionHistory:
    def __init__(self, size: int = 4):
        self._size = size
        self._history = np.zeros((size, 4), dtype=np.float64)
        self._idx = 0
    
    def add(self, action: np.ndarray):
        self._history[self._idx % self._size] = action
        self._idx += 1
    
    def get_rate(self) -> np.ndarray:
        if self._idx < 2:
            return np.zeros(4, dtype=np.float64)
        curr = self._history[(self._idx - 1) % self._size]
        prev = self._history[(self._idx - 2) % self._size]
        return curr - prev
    
    def get_accel(self) -> np.ndarray:
        if self._idx < 3:
            return np.zeros(4, dtype=np.float64)
        curr = self._history[(self._idx - 1) % self._size]
        prev = self._history[(self._idx - 2) % self._size]
        prev2 = self._history[(self._idx - 3) % self._size]
        return curr - 2*prev + prev2
    
    def reset(self):
        self._history.fill(0)
        self._idx = 0


class TargetGenerator:
    def __init__(self, mode: str = 'static', center: np.ndarray = None):
        self._mode = mode
        self._center = center if center is not None else np.array([0.0, 0.0, 2.0])
        self._t = 0.0
        self._speed = 0.0
        self._position = self._center.copy()
        self._velocity = np.zeros(3)
        self._phase = np.random.uniform(0, 2*np.pi)
    
    def set_speed(self, speed: float):
        self._speed = speed
    
    def update(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        self._t += dt
        
        if self._mode == 'static':
            self._velocity = np.zeros(3)
            return self._center.copy(), self._velocity
        
        if self._mode == 'figure8':
            omega = self._speed * 0.3
            x = 2.0 * np.sin(omega * self._t + self._phase)
            y = 2.0 * np.sin(2 * omega * self._t + self._phase)
            z = self._center[2] + 0.3 * np.sin(0.5 * omega * self._t)
            
            vx = 2.0 * omega * np.cos(omega * self._t + self._phase)
            vy = 4.0 * omega * np.cos(2 * omega * self._t + self._phase)
            vz = 0.15 * omega * np.cos(0.5 * omega * self._t)
            
            self._position = np.array([x, y, z])
            self._velocity = np.array([vx, vy, vz])
        
        elif self._mode == 'circle':
            omega = self._speed * 0.4
            radius = 2.5
            x = radius * np.cos(omega * self._t + self._phase)
            y = radius * np.sin(omega * self._t + self._phase)
            z = self._center[2]
            
            vx = -radius * omega * np.sin(omega * self._t + self._phase)
            vy = radius * omega * np.cos(omega * self._t + self._phase)
            vz = 0.0
            
            self._position = np.array([x, y, z])
            self._velocity = np.array([vx, vy, vz])
        
        return self._position.copy(), self._velocity.copy()
    
    def reset(self, center: np.ndarray = None):
        self._t = 0.0
        self._phase = np.random.uniform(0, 2*np.pi)
        if center is not None:
            self._center = center
        self._position = self._center.copy()
        self._velocity = np.zeros(3)


class QuadrotorEnvV2(gym.Env):
                                                                                
    
    metadata = {'render_modes': ['human'], 'render_fps': 50}
    
    def __init__(
        self,
        config: Optional[ExtendedEnvConfig] = None,
        reward_config: Optional['RewardConfig'] = None,
        task: TaskType = TaskType.HOVER,
        target: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.config = config or ExtendedEnvConfig()
        self._reward_config = reward_config
        self.task = task
        self.render_mode = render_mode
        
        if task in (TaskType.PAYLOAD_HOVER, TaskType.PAYLOAD_TRANSPORT):
            self.config.payload_enabled = True
        
        self._setup_drone()
        self._setup_sota_actuator()
        self._setup_estimation()
        self._setup_payload()
        self._setup_observation_builder()
        self._setup_reward_builder()
        
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        self.observation_space = self._obs_builder.observation_space
        
        self._target_generator = TargetGenerator(
            mode='static' if task in (TaskType.HOVER, TaskType.PAYLOAD_HOVER) else 'figure8',
            center=target if target is not None else np.array([0.0, 0.0, 2.0])
        )
        self.target = self._target_generator._center.copy()
        self._target_velocity = np.zeros(3)
        
        self._action_history = ActionHistory(size=4)
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._success_counter = 0
        self._steps = 0
        self._episode_reward = 0.0
        self._rng = np.random.default_rng()
        self._total_episodes = 0
        self._curriculum_progress = 0.0
        self._target_speed_progress = 0.0
        
        self._prev_cable_theta = np.zeros(2)
        self._cable_angle_rate = np.zeros(2)
    
    def _setup_drone(self):
        self._cfg = drone_core.QuadrotorConfig()
        self._cfg.integration_method = drone_core.IntegrationMethod.RK4
        self._cfg.motor_config = drone_core.MotorConfiguration.X
        self._cfg.enable_ground_effect = True
        self._cfg.enable_wind_disturbance = self.config.wind_enabled
        self._cfg.enable_motor_dynamics = self.config.motor_dynamics
        self._cfg.enable_battery_dynamics = False
        self._cfg.enable_blade_flapping = True
        self._cfg.enable_advanced_aero = True
        
        self._cfg.sub_step.physics_sub_steps = self.config.physics_sub_steps
        self._cfg.sub_step.enable_sub_stepping = self.config.use_sub_stepping
        self._cfg.sub_step.min_sub_step_dt = 0.0001
        self._cfg.sub_step.max_sub_step_dt = 0.005
        
        self._cfg.mass = self.config.mass
        self._cfg.arm_length = 0.25
        
        self._cfg.rotor.radius = 0.127
        self._cfg.rotor.chord = 0.02
        self._cfg.rotor.pitch_angle = 0.18
        self._cfg.rotor.flapping.enabled = True
        self._cfg.rotor.flapping.lock_number = 8.0
        
        self._cfg.motor.kv = 2300
        self._cfg.motor.max_current = 30
        self._cfg.motor.max_rpm = self.config.max_rpm
        self._cfg.motor.esc.nonlinear_gamma = 1.2
        
        self._cfg.aero.air_density = 1.225
        self._cfg.aero.ground_effect_coeff = 0.5
        
        self._drone = drone_core.Quadrotor(self._cfg)
    
    def _setup_sota_actuator(self):
        actuator_cfg = drone_core.SOTAActuatorConfig()
        actuator_cfg.delay_ms = 0.0
        actuator_cfg.tau_spin_up = 0.01
        actuator_cfg.tau_spin_down = 0.008
        actuator_cfg.rotor_inertia = 1.0e-6
        actuator_cfg.voltage_sag_factor = 0.0
        actuator_cfg.max_rpm = self.config.max_rpm
        actuator_cfg.min_rpm = self.config.min_rpm
        actuator_cfg.hover_rpm = self.config.hover_rpm
        actuator_cfg.max_slew_rate = 200000.0
        actuator_cfg.process_noise_std = 0.0
        actuator_cfg.active_braking_gain = 3.0
        actuator_cfg.thermal_time_constant = 1000.0
        actuator_cfg.nominal_voltage = 16.8
        self._sota_actuator = drone_core.SOTAActuatorModel(actuator_cfg)
        
        if self.config.use_motor_mixer:
            self._motor_mixer = MotorMixer(MotorMixerConfig())
        else:
            self._motor_mixer = None
    
    def _setup_estimation(self):
        if self.config.use_eskf:
            sensor_noise = drone_core.SensorNoise()
            sensor_noise.cable_sensor_enabled = self.config.cable_sensor_enabled
            sensor_noise.cable_angle_std = self.config.cable_angle_noise
            
            self._sensor_sim = drone_core.SensorSimulator(sensor_noise)
            self._estimator = drone_core.StateEstimator(sensor_noise)
        else:
            self._sensor_sim = None
            self._estimator = None
    
    def _setup_payload(self):
        if self.config.payload_enabled:
            payload_cfg = drone_core.PayloadConfig()
            payload_cfg.mass = self.config.payload_mass
            
            cable_cfg = drone_core.CableConfig()
            cable_cfg.rest_length = self.config.cable_length
            
            self._payload = drone_core.PayloadDynamics(payload_cfg, cable_cfg)
            self._swing_controller = drone_core.SwingingPayloadController()
        else:
            self._payload = None
            self._swing_controller = None
    
    def _setup_observation_builder(self):
        mode_map = {
            'true': ObservationMode.TRUE_STATE,
            'estimated': ObservationMode.ESTIMATED,
            'sensor': ObservationMode.SENSOR_ONLY,
        }
        
        obs_config = ObsConfig(
            mode=mode_map.get(self.config.observation_mode, ObservationMode.ESTIMATED),
            position_scale=self.config.position_scale,
            velocity_scale=self.config.velocity_scale,
            angular_velocity_scale=self.config.angular_velocity_scale,
            include_cable_angles=self.config.payload_enabled and self.config.cable_sensor_enabled,
            include_payload_state=self.config.payload_enabled,
            observation_noise=self.config.observation_noise,
        )
        
        self._obs_builder = ObservationBuilder(obs_config)
    
    def _setup_reward_builder(self):
        reward_cfg = self._reward_config if self._reward_config is not None else RewardConfig()
        self._reward_builder = RewardBuilder(
            config=reward_cfg,
            include_payload=self.config.payload_enabled
        )
    
    def _apply_domain_randomization(self):
        if not self.config.domain_randomization:
            return
        
        mass_low, mass_high = self.config.mass_randomization
        mass_factor = self._rng.uniform(mass_low, mass_high)
        self._cfg.mass = self.config.mass * mass_factor
        
        if self.config.wind_enabled:
            wind_low, wind_high = self.config.wind_speed_range
            wind_speed = self._rng.uniform(wind_low, wind_high)
            wind_dir = self._rng.uniform(0, 2 * np.pi)
            wind = drone_core.Vec3(
                wind_speed * np.cos(wind_dir),
                wind_speed * np.sin(wind_dir),
                self._rng.uniform(-0.3, 0.3)
            )
            self._drone.set_wind(wind)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        self._drone.reset()
        self._apply_domain_randomization()
        
        if self.config.curriculum_enabled:
            self._curriculum_progress = min(1.0, self._curriculum_progress + self.config.curriculum_progress_rate)
            curr_range = self.config.curriculum_init_range + (
                self.config.curriculum_max_range - self.config.curriculum_init_range
            ) * self._curriculum_progress
        else:
            curr_range = self.config.curriculum_max_range
        
        target_center = np.array([
            self._rng.uniform(-2 * curr_range, 2 * curr_range),
            self._rng.uniform(-2 * curr_range, 2 * curr_range),
            2.0 + self._rng.uniform(-0.5 * curr_range, 0.5 * curr_range)
        ])
        self._target_generator.reset(target_center)
        self.target, self._target_velocity = self._target_generator.update(0.0)
        
        init_x = self._rng.uniform(-0.15 * curr_range, 0.15 * curr_range)
        init_y = self._rng.uniform(-0.15 * curr_range, 0.15 * curr_range)
        init_z = self.target[2] + self._rng.uniform(-0.3 * curr_range, 0.3 * curr_range)
        init_z = np.clip(init_z, 0.5, 5.0)
        self._drone.set_position(drone_core.Vec3(init_x, init_y, init_z))
        
        init_vx = self._rng.uniform(-0.2 * curr_range, 0.2 * curr_range)
        init_vy = self._rng.uniform(-0.2 * curr_range, 0.2 * curr_range)
        init_vz = self._rng.uniform(-0.1 * curr_range, 0.1 * curr_range)
        self._drone.set_velocity(drone_core.Vec3(init_vx, init_vy, init_vz))
        
        roll = self._rng.uniform(-0.03 * curr_range, 0.03 * curr_range)
        pitch = self._rng.uniform(-0.03 * curr_range, 0.03 * curr_range)
        yaw = self._rng.uniform(-0.15 * curr_range, 0.15 * curr_range)
        self._drone.set_orientation(
            drone_core.Quaternion.from_euler_zyx(roll, pitch, yaw)
        )
        
        if self._estimator is not None:
            self._estimator.reset(self._drone.get_state())
            self._sensor_sim.reset()
        
        if self._payload is not None:
            self._payload.reset()
            s = self._drone.get_state()
            self._payload.step(s.position, s.orientation, s.velocity, s.angular_velocity, 0.001)
            self._payload.attach(drone_core.Vec3(0, 0, -self.config.cable_length))
        
        self._action_history.reset()
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._success_counter = 0
        self._steps = 0
        self._episode_reward = 0.0
        self._total_episodes += 1
        self._prev_rpm = np.full(4, self.config.hover_rpm, dtype=np.float64)
        self._sota_actuator.reset()
        self._reward_builder.reset()
        self._obs_builder.reset(seed)
        self._prev_cable_theta = np.zeros(2)
        self._cable_angle_rate = np.zeros(2)
        
        self._prev_drone_state = self._drone.get_state()
        
        return self._get_obs(), self._get_info()
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        
        if self._motor_mixer is not None:
            motor_action = self._motor_mixer.mix(action)
        else:
            motor_action = action
        
        self._action_history.add(motor_action)
        
        voltage = self._drone.get_state().battery_voltage
        rpm = self._sota_actuator.step_normalized(motor_action, self.config.dt, voltage)
        rpm = np.array(rpm, dtype=np.float64)
        
        self._prev_rpm = rpm.copy()
        self._current_rpm = rpm.copy()
        
        cmd = drone_core.MotorCommand(float(rpm[0]), float(rpm[1]), float(rpm[2]), float(rpm[3]))
        
        if self.config.use_sub_stepping:
            self._drone.step_with_sub_stepping(cmd, self.config.dt)
        else:
            self._drone.step(cmd, self.config.dt)
        
        drone_state = self._drone.get_state()
        
        if self._payload is not None:
            self._payload.step(
                drone_state.position,
                drone_state.orientation,
                drone_state.velocity,
                drone_state.angular_velocity,
                self.config.dt
            )
        
        if self._estimator is not None:
            accel = drone_core.Vec3(0, 0, 0)
            self._last_sensor_reading = self._sensor_sim.simulate(drone_state, accel, self.config.dt)
            self._last_eskf_state = self._estimator.update(drone_state, accel, self.config.dt)
            
            if self._payload is not None and self.config.cable_sensor_enabled:
                swing = self._payload.get_swing_angle()
                tension = self._payload.get_coupling_forces().total_tension
                self._last_cable_reading = self._sensor_sim.simulate_cable_angle(
                    swing.x, swing.y, tension, self.config.dt
                )
                
                if self._last_cable_reading.valid:
                    new_theta = np.array([self._last_cable_reading.theta_x, self._last_cable_reading.theta_y])
                    self._cable_angle_rate = (new_theta - self._prev_cable_theta) / self.config.dt
                    self._prev_cable_theta = new_theta
            else:
                self._last_cable_reading = None
        else:
            self._last_sensor_reading = None
            self._last_eskf_state = None
            self._last_cable_reading = None
        
        self.target, self._target_velocity = self._target_generator.update(self.config.dt)
        
        self._steps += 1
        
        obs = self._get_obs()
        reward, terminated = self._compute_reward(action)
        truncated = self._steps >= self.config.max_steps
        
        self._prev_action = action.astype(np.float32).copy()
        self._episode_reward += reward
        self._prev_drone_state = drone_state
        
        return obs, reward, terminated, truncated, self._get_info()
    
    def _get_obs(self) -> np.ndarray:
        drone_state = self._drone.get_state()
        
        payload_state = None
        if self._payload is not None:
            payload_state = self._payload.get_payload_state()
        
        env_state = EnvState(
            drone_state=drone_state,
            eskf_state=getattr(self, '_last_eskf_state', None),
            sensor_reading=getattr(self, '_last_sensor_reading', None),
            payload_state=payload_state,
            cable_reading=getattr(self, '_last_cable_reading', None),
            target_position=self.target,
            target_velocity=self._target_velocity,
            prev_action=self._prev_action,
            cable_angle_rate=self._cable_angle_rate,
        )
        
        return self._obs_builder.build(env_state)
    
    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool]:
        s = self._drone.get_state()
        
        pos = np.array([s.position.x, s.position.y, s.position.z])
        vel = np.array([s.velocity.x, s.velocity.y, s.velocity.z])
        ang_vel = np.array([s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z])
        
        euler = s.orientation.to_euler_zyx()
        roll, pitch, yaw = abs(euler.x), abs(euler.y), euler.z
        
        error_vec = self.target - pos
        dist = np.linalg.norm(error_vec)
        speed = np.linalg.norm(vel)
        omega_xy = np.linalg.norm(ang_vel[:2])
        omega_z = abs(ang_vel[2])
        
        prev_dist = getattr(self, '_prev_dist', dist)
        self._prev_dist = dist
        
        r_progress = (prev_dist - dist) * 3.0
        
        r_proximity = np.exp(-dist * 2.0) * 1.5
        
        target_dir = error_vec / (dist + 1e-6)
        forward_body = np.array([np.cos(yaw), np.sin(yaw), 0])
        heading_alignment = np.dot(forward_body[:2], target_dir[:2])
        r_heading = heading_alignment * 2.0
        
        action_diff = np.linalg.norm(self._prev_action - action) if hasattr(self, '_prev_action') else 0.0
        r_smooth = -action_diff ** 2 * 0.5
        
        r_stability = -omega_xy * 0.1
        
        r_orientation = np.exp(-(roll + pitch) * 5.0) * 2.0
        
        if speed < 0.5:
            r_anti_spin = -omega_z * 0.3
        else:
            r_anti_spin = -omega_z * 0.05
        
        r_anti_kamikaze = 0.0
        if dist < 1.0 and speed > 2.0:
            r_anti_kamikaze = -1.0
        
        if pos[2] < 0.3:
            r_ground = -(0.3 - pos[2]) * 5.0
        else:
            r_ground = 0.0
        
        reward = (
            r_progress +
            r_proximity +
            r_heading +
            r_smooth +
            r_stability +
            r_orientation +
            r_anti_spin +
            r_anti_kamikaze +
            r_ground
        )
        
        is_hovering = dist < 0.3 and speed < 0.3 and roll < 0.15 and pitch < 0.15
        if is_hovering:
            self._hover_duration = getattr(self, '_hover_duration', 0) + 1
            reward += min(self._hover_duration / 20.0, 4.0)
        else:
            self._hover_duration = max(0, getattr(self, '_hover_duration', 0) - 1)
        
        terminated = False
        crashed = False
        crash_reason = None
        
        if pos[2] < 0.05:
            crashed = True
            crash_reason = "ground"
        elif dist > 3.0:
            crashed = True
            crash_reason = "distance"
        elif roll > 1.0 or pitch > 1.0:
            crashed = True
            crash_reason = "angle"
        elif speed > 10.0:
            crashed = True
            crash_reason = "velocity"
        
        if crashed:
            reward = -10.0
            terminated = True
            self._last_crash_reason = crash_reason
        else:
            self._last_crash_reason = None
        
        return float(reward), terminated
    
    def _get_info(self) -> Dict[str, Any]:
        s = self._drone.get_state()
        pos = np.array([s.position.x, s.position.y, s.position.z])
        vel = np.array([s.velocity.x, s.velocity.y, s.velocity.z])
        euler = s.orientation.to_euler_zyx()
        
        info = {
            'position': pos,
            'velocity': vel,
            'speed': np.linalg.norm(vel),
            'roll': euler.x,
            'pitch': euler.y,
            'yaw': euler.z,
            'target': self.target,
            'target_velocity': self._target_velocity,
            'target_error': np.linalg.norm(self.target - pos),
            'steps': self._steps,
            'reward': self._episode_reward,
            'success_counter': self._success_counter,
            'curriculum_progress': self._curriculum_progress,
            'crash_reason': getattr(self, '_last_crash_reason', None),
            'rpm': self._current_rpm.copy() if hasattr(self, '_current_rpm') else np.zeros(4),
            'rpm_mean': np.mean(self._current_rpm) if hasattr(self, '_current_rpm') else 0,
            'battery_voltage': s.battery_voltage,
        }
        
        if self._payload is not None:
            info['payload_swing'] = self._payload.get_cable_angle_from_vertical()
            info['payload_tension'] = self._payload.get_coupling_forces().total_tension
        
        return info
    
    def set_target(self, target: np.ndarray):
        self.target = np.asarray(target, dtype=np.float32)
        self._target_generator.reset(self.target)
    
    def get_drone_state(self):
        return self._drone.get_state()
    
    def get_estimated_state(self):
        if self._estimator is not None:
            return self._estimator.get_estimated_state()
        return None
    
    def get_payload_state(self):
        if self._payload is not None:
            return self._payload.get_payload_state()
        return None

    def set_physics_params(
        self,
        mass: Optional[float] = None,
        motor_lag: Optional[float] = None,
        friction: Optional[float] = None,
        inertia_scale: Optional[float] = None
    ):
        if mass is not None:
            self._cfg.mass = mass
            if hasattr(self._drone, 'set_mass'):
                self._drone.set_mass(mass)
            self._drone.update_config(self._cfg)
        
        if motor_lag is not None:
            self._sota_actuator.config.tau_spin_up = motor_lag
            self._sota_actuator.config.tau_spin_down = motor_lag * 0.5
            if hasattr(self._sota_actuator, 'set_motor_lag'):
                self._sota_actuator.set_motor_lag(motor_lag)

        if inertia_scale is not None and hasattr(self._drone, 'set_inertia_scale'):
            self._drone.set_inertia_scale(inertia_scale)

    def set_wind_intensity(self, intensity: float):
        if not self.config.wind_enabled:
            return
            
        if hasattr(self._drone, 'set_wind_config'):
            wind_cfg = drone_core.WindConfig()
            wind_cfg.enabled = intensity > 0.01
            wind_cfg.intensity = intensity
            self._drone.set_wind_config(wind_cfg)
        else:
            wind_dir = self._rng.uniform(0, 2 * np.pi)
            wind = drone_core.Vec3(
                intensity * np.cos(wind_dir),
                intensity * np.sin(wind_dir),
                self._rng.uniform(-0.1 * intensity, 0.1 * intensity)
            )
            self._drone.set_wind(wind)
    
    def render(self):
        if self.render_mode == 'human':
            s = self._drone.get_state()
            msg = f"Step {self._steps}: pos=({s.position.x:.2f}, {s.position.y:.2f}, {s.position.z:.2f})"
            if self._payload is not None:
                msg += f" swing={np.degrees(self._payload.get_cable_angle_from_vertical()):.1f}°"
            print(msg)


QuadrotorEnv = QuadrotorEnvV2


class VectorizedQuadrotorEnv:
    def __init__(
        self,
        num_envs: int,
        config: Optional[ExtendedEnvConfig] = None,
        reward_config: Optional['RewardConfig'] = None,
        task: TaskType = TaskType.HOVER
    ):
        self.num_envs = num_envs
        self.envs = [QuadrotorEnvV2(config, reward_config, task) for _ in range(num_envs)]
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(
        self, 
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs_list = []
        info_list = []
        
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            info_list.append(info)
        
        return np.stack(obs_list), {'envs': info_list}
    
    def step(
        self, 
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                final_info = info.copy()
                obs, _ = env.reset()
                info['final_info'] = final_info
            
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)
        
        return (
            np.stack(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(terminated_list),
            np.array(truncated_list),
            {'envs': info_list}
        )


EnvConfig = ExtendedEnvConfig