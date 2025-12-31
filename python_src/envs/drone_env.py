"""Extended drone environment with ESKF, Payload, and Collision integration."""
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
    physics_sub_steps: int = 4
    max_steps: int = 2000
    max_rpm: float = 35000.0
    min_rpm: float = 2000.0
    hover_rpm: float = 2750.0
    rpm_range: float = 2000.0
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
    """SOTA Quadrotor environment with ESKF, Payload, and modular components."""
    
    metadata = {'render_modes': ['human'], 'render_fps': 50}
    
    def __init__(
        self,
        config: Optional[ExtendedEnvConfig] = None,
        task: TaskType = TaskType.HOVER,
        target: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.config = config or ExtendedEnvConfig()
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
        actuator_cfg.delay_ms = 15.0
        actuator_cfg.tau_spin_up = 0.10
        actuator_cfg.tau_spin_down = 0.05
        actuator_cfg.rotor_inertia = 2.0e-5
        actuator_cfg.voltage_sag_factor = 0.06
        actuator_cfg.max_rpm = self.config.max_rpm
        actuator_cfg.min_rpm = self.config.min_rpm
        actuator_cfg.hover_rpm = self.config.hover_rpm
        actuator_cfg.max_slew_rate = 50000.0
        actuator_cfg.process_noise_std = 20.0
        actuator_cfg.active_braking_gain = 1.8
        actuator_cfg.thermal_time_constant = 25.0
        actuator_cfg.nominal_voltage = 16.8
        self._sota_actuator = drone_core.SOTAActuatorModel(actuator_cfg)
    
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
            'true_state': ObservationMode.TRUE_STATE,
            'estimated': ObservationMode.ESTIMATED,
            'sensor_only': ObservationMode.SENSOR_ONLY,
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
        self._reward_builder = RewardBuilder(
            config=RewardConfig(),
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
        self._action_history.add(action)
        
        voltage = self._drone.get_state().battery_voltage
        rpm = self._sota_actuator.step_normalized(action, self.config.dt, voltage)
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
        euler_arr = np.array([euler.x, euler.y, euler.z])
        
        payload_pos = None
        swing_angle = 0.0
        payload_energy = 0.0
        
        if self._payload is not None:
            ps = self._payload.get_payload_state()
            payload_pos = np.array([ps.position.x, ps.position.y, ps.position.z])
            swing_angle = self._payload.get_cable_angle_from_vertical()
            payload_energy = self._payload.get_total_energy()
        
        reward_state = RewardState(
            position=pos,
            velocity=vel,
            angular_velocity=ang_vel,
            euler_angles=euler_arr,
            target_position=self.target,
            target_velocity=self._target_velocity,
            action=action,
            action_rate=self._action_history.get_rate(),
            action_accel=self._action_history.get_accel(),
            motor_rpm=self._current_rpm if hasattr(self, '_current_rpm') else np.zeros(4),
            hover_rpm=self.config.hover_rpm,
            payload_swing_angle=swing_angle,
            payload_position=payload_pos,
            payload_energy=payload_energy,
        )
        
        reward = self._reward_builder.compute(reward_state)
        
        crashed, _ = self._reward_builder.check_termination(reward_state)
        if crashed:
            reward = self._reward_builder.config.crash_penalty
        
        if self._reward_builder.check_success(reward_state):
            self._success_counter += 1
            if self._success_counter >= 50:
                reward += self._reward_builder.config.success_bonus
        else:
            self._success_counter = max(0, self._success_counter - 1)
        
        return float(reward), crashed
    
    def _get_info(self) -> Dict[str, Any]:
        s = self._drone.get_state()
        pos = np.array([s.position.x, s.position.y, s.position.z])
        
        info = {
            'position': pos,
            'target': self.target,
            'target_velocity': self._target_velocity,
            'distance': np.linalg.norm(self.target - pos),
            'steps': self._steps,
            'episode_reward': self._episode_reward,
            'success_counter': self._success_counter,
            'curriculum_progress': self._curriculum_progress,
            'sub_step_count': self._drone.get_sub_step_count() if self.config.use_sub_stepping else 1,
            'observation_mode': self.config.observation_mode,
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
        task: TaskType = TaskType.HOVER
    ):
        self.num_envs = num_envs
        self.envs = [QuadrotorEnvV2(config, task) for _ in range(num_envs)]
        
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