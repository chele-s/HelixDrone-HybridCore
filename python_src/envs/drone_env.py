import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'Release'))

import gymnasium as gym
import numpy as np
from scipy import signal
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import IntEnum

import drone_core


class TaskType(IntEnum):
    HOVER = 0
    WAYPOINT = 1
    TRAJECTORY = 2
    VELOCITY = 3


@dataclass
class EnvConfig:
    dt: float = 0.01
    physics_sub_steps: int = 4
    max_steps: int = 1000
    max_rpm: float = 35000.0
    min_rpm: float = 2000.0
    hover_rpm: float = 2750.0
    rpm_range: float = 2000.0
    mass: float = 0.6
    
    position_scale: float = 5.0
    velocity_scale: float = 5.0
    angular_velocity_scale: float = 10.0
    
    reward_position: float = -0.5
    reward_velocity: float = -0.025
    reward_angular: float = -0.01
    reward_action: float = -0.0005
    reward_action_rate: float = -0.1
    reward_action_accel: float = -0.05
    reward_action_jerk: float = -0.05
    reward_alive: float = 1.0
    reward_crash: float = -10.0
    reward_success: float = 50.0
    
    reward_height_bonus: float = 0.5
    reward_stability_bonus: float = 0.25
    reward_hover_bonus: float = 2.5
    
    crash_height: float = 0.03
    crash_distance: float = 10.0
    crash_angle: float = 1.4
    success_distance: float = 0.3
    success_velocity: float = 0.5
    success_hold_steps: int = 1000
    
    curriculum_enabled: bool = True
    curriculum_init_range: float = 0.05
    curriculum_max_range: float = 0.5
    curriculum_progress_rate: float = 0.00002
    
    domain_randomization: bool = True
    wind_enabled: bool = True
    motor_dynamics: bool = True
    use_sub_stepping: bool = True
    use_sota_actuator: bool = False


class QuadrotorEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 50}
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        task: TaskType = TaskType.HOVER,
        target: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.config = config or EnvConfig()
        self.task = task
        self.render_mode = render_mode
        
        self._setup_drone()
        
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        self._history_length = 4
        self._base_obs_dim = 20
        self._history_dim = self._history_length * 4 + self._history_length * 4
        self._total_obs_dim = self._base_obs_dim + self._history_dim
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._total_obs_dim,), dtype=np.float32
        )
        
        self._action_history = np.zeros((self._history_length, 4), dtype=np.float32)
        self._rpm_history = np.zeros((self._history_length, 4), dtype=np.float32)
        
        self.target = target if target is not None else np.array([0.0, 0.0, 2.0])
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._success_counter = 0
        self._steps = 0
        self._episode_reward = 0.0
        self._rng = np.random.default_rng()
        self._total_episodes = 0
        self._curriculum_progress = 0.0
        
        if not self.config.use_sota_actuator:
            fs = 1.0 / self.config.dt
            fc = 40.0
            w = min(fc / (fs / 2), 0.99)
            self._b, self._a = signal.butter(2, w, 'low')
            self._action_buffer = np.zeros((4, 3), dtype=np.float32)
            self._filter_state = np.zeros((4, 2), dtype=np.float32)
    
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
        
        if self.config.use_sota_actuator:
            actuator_cfg = drone_core.SOTAActuatorConfig()
            actuator_cfg.hover_rpm = self.config.hover_rpm
            actuator_cfg.max_rpm = self.config.max_rpm
            actuator_cfg.min_rpm = self.config.min_rpm
            actuator_cfg.rpm_range = self.config.rpm_range
            actuator_cfg.simulation_dt = self.config.dt / self.config.physics_sub_steps
            actuator_cfg.damping_ratio = 0.9
            actuator_cfg.tau_spin_up = 0.012
            actuator_cfg.tau_spin_down = 0.010
            actuator_cfg.rotor_inertia = 2.5e-5
            actuator_cfg.process_noise_std = 5.0 if self.config.domain_randomization else 0.0
            actuator_cfg.thermal_derating = 0.0
            actuator_cfg.max_temperature = 80.0
            actuator_cfg.delay_ms = 2.0
            actuator_cfg.max_slew_rate = 50000.0
            actuator_cfg.active_braking_gain = 1.2
            self._sota_actuator = drone_core.SOTAActuatorModel(actuator_cfg)
        else:
            self._sota_actuator = None
    
    def _apply_domain_randomization(self):
        if not self.config.domain_randomization:
            return
        
        mass_factor = self._rng.uniform(0.9, 1.1)
        self._cfg.mass = self.config.mass * mass_factor
        
        if self.config.wind_enabled:
            wind_speed = self._rng.uniform(0, 3)
            wind_dir = self._rng.uniform(0, 2 * np.pi)
            wind = drone_core.Vec3(
                wind_speed * np.cos(wind_dir),
                wind_speed * np.sin(wind_dir),
                self._rng.uniform(-0.5, 0.5)
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
        
        if self.task == TaskType.HOVER:
            init_x = self._rng.uniform(-0.1 * curr_range, 0.1 * curr_range)
            init_y = self._rng.uniform(-0.1 * curr_range, 0.1 * curr_range)
            init_z = self.target[2] + self._rng.uniform(-0.2 * curr_range, 0.2 * curr_range)
            init_z = max(init_z, 0.5)
            self._drone.set_position(drone_core.Vec3(init_x, init_y, init_z))
            
            init_vx = self._rng.uniform(-0.1 * curr_range, 0.1 * curr_range)
            init_vy = self._rng.uniform(-0.1 * curr_range, 0.1 * curr_range)
            init_vz = self._rng.uniform(-0.05 * curr_range, 0.05 * curr_range)
            self._drone.set_velocity(drone_core.Vec3(init_vx, init_vy, init_vz))
            
            roll = self._rng.uniform(-0.02 * curr_range, 0.02 * curr_range)
            pitch = self._rng.uniform(-0.02 * curr_range, 0.02 * curr_range)
            yaw = self._rng.uniform(-0.1 * curr_range, 0.1 * curr_range)
            self._drone.set_orientation(
                drone_core.Quaternion.from_euler_zyx(roll, pitch, yaw)
            )
        
        elif self.task == TaskType.WAYPOINT:
            self._drone.set_position(drone_core.Vec3(0, 0, 0.5))
            self.target = np.array([
                self._rng.uniform(-3, 3),
                self._rng.uniform(-3, 3),
                self._rng.uniform(1, 4)
            ])
        
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._success_counter = 0
        self._steps = 0
        self._episode_reward = 0.0
        self._total_episodes += 1
        self._hover_duration = 0
        self._prev_rpm = np.full(4, self.config.hover_rpm, dtype=np.float32)
        
        if not self.config.use_sota_actuator:
            self._action_buffer = np.zeros((4, 3), dtype=np.float32)
            self._filter_state = np.zeros((4, 2), dtype=np.float32)
        
        if self._sota_actuator is not None:
            self._sota_actuator.reset()
        
        self._action_history = np.zeros((self._history_length, 4), dtype=np.float32)
        self._rpm_history = np.full((self._history_length, 4), self.config.hover_rpm, dtype=np.float32)
        
        if self.config.motor_dynamics:
            warmup_rpm = self.config.hover_rpm
            warmup_cmd = drone_core.MotorCommand(warmup_rpm, warmup_rpm, warmup_rpm, warmup_rpm)
            for _ in range(10):
                if self.config.use_sub_stepping:
                    self._drone.step_with_sub_stepping(warmup_cmd, self.config.dt)
                else:
                    self._drone.step(warmup_cmd, self.config.dt)
            self._drone.set_velocity(drone_core.Vec3(0, 0, 0))
            self._drone.set_angular_velocity(drone_core.Vec3(0, 0, 0))
        
        return self._get_obs(), self._get_info()
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        if self._sota_actuator is not None:
            voltage = 16.8
            rpm = np.array(self._sota_actuator.step_normalized(
                action.tolist(), self.config.dt, voltage
            ), dtype=np.float64)
        else:
            self._action_buffer[:, 2] = self._action_buffer[:, 1]
            self._action_buffer[:, 1] = self._action_buffer[:, 0]
            self._action_buffer[:, 0] = action
            
            x = self._action_buffer
            y = self._filter_state
            smoothed = (
                self._b[0] * x[:, 0] + self._b[1] * x[:, 1] + self._b[2] * x[:, 2]
                - self._a[1] * y[:, 0] - self._a[2] * y[:, 1]
            ) / self._a[0]
            
            self._filter_state[:, 1] = self._filter_state[:, 0]
            self._filter_state[:, 0] = smoothed
            
            rpm = self.config.hover_rpm + smoothed * self.config.rpm_range
            rpm = np.clip(rpm, self.config.min_rpm, self.config.max_rpm)
            
            max_rpm_change = 45000.0 * self.config.dt
            rpm_delta = np.clip(rpm - self._prev_rpm, -max_rpm_change, max_rpm_change)
            rpm = self._prev_rpm + rpm_delta
        
        self._prev_rpm = rpm.copy()
        
        self._action_history[1:] = self._action_history[:-1]
        self._action_history[0] = action
        self._rpm_history[1:] = self._rpm_history[:-1]
        self._rpm_history[0] = rpm.astype(np.float32)
        
        cmd = drone_core.MotorCommand(rpm[0], rpm[1], rpm[2], rpm[3])
        
        if self.config.use_sub_stepping:
            self._drone.step_with_sub_stepping(cmd, self.config.dt)
        else:
            self._drone.step(cmd, self.config.dt)
        
        self._steps += 1
        
        obs = self._get_obs()
        reward, terminated = self._compute_reward(action)
        truncated = self._steps >= self.config.max_steps
        
        self._prev_action = action.copy()
        self._episode_reward += reward
        
        return obs, reward, terminated, truncated, self._get_info()
    
    def _get_obs(self) -> np.ndarray:
        s = self._drone.get_state()
        
        pos = np.array([s.position.x, s.position.y, s.position.z]) / self.config.position_scale
        vel = np.array([s.velocity.x, s.velocity.y, s.velocity.z]) / self.config.velocity_scale
        
        quat = np.array([s.orientation.w, s.orientation.x, s.orientation.y, s.orientation.z])
        
        ang_vel = np.array([
            s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z
        ]) / self.config.angular_velocity_scale
        
        error = (self.target - np.array([s.position.x, s.position.y, s.position.z])) / self.config.position_scale
        
        action_history_flat = self._action_history.flatten()
        rpm_history_normalized = (self._rpm_history - self.config.hover_rpm) / self.config.rpm_range
        rpm_history_flat = rpm_history_normalized.flatten()
        
        return np.concatenate([
            pos, vel, quat, ang_vel, error, 
            self._prev_action, action_history_flat, rpm_history_flat
        ]).astype(np.float32)
    
    def _get_base_obs(self) -> np.ndarray:
        s = self._drone.get_state()
        
        pos = np.array([s.position.x, s.position.y, s.position.z]) / self.config.position_scale
        vel = np.array([s.velocity.x, s.velocity.y, s.velocity.z]) / self.config.velocity_scale
        quat = np.array([s.orientation.w, s.orientation.x, s.orientation.y, s.orientation.z])
        ang_vel = np.array([
            s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z
        ]) / self.config.angular_velocity_scale
        error = (self.target - np.array([s.position.x, s.position.y, s.position.z])) / self.config.position_scale
        
        return np.concatenate([pos, vel, quat, ang_vel, error, self._prev_action]).astype(np.float32)
    
    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool]:
        s = self._drone.get_state()
        
        pos = np.array([s.position.x, s.position.y, s.position.z])
        vel = np.array([s.velocity.x, s.velocity.y, s.velocity.z])
        ang_vel = np.array([s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z])
        
        euler = s.orientation.to_euler_zyx()
        roll, pitch = abs(euler.x), abs(euler.y)
        
        error_vec = self.target - pos
        dist = np.linalg.norm(error_vec)
        speed = np.linalg.norm(vel)
        
        r_distance = -dist * 0.8
        
        sigma = 0.5
        r_proximity = 2.5 * np.exp(-(dist ** 2) / sigma)
        
        if dist > 0.25:
            target_dir = error_vec / dist
            approach_speed = np.dot(vel, target_dir)
            r_velocity = np.clip(approach_speed, -3.0, 3.0) * 1.5
        else:
            r_velocity = -speed * 4.0
        
        tilt_magnitude = np.sqrt(roll ** 2 + pitch ** 2)
        r_orientation = 2.0 * np.exp(-tilt_magnitude * 6.0)
        
        omega_magnitude = np.linalg.norm(ang_vel)
        r_stability = -omega_magnitude * 0.15
        
        action_delta = action - self._prev_action
        r_smoothness = -np.dot(action_delta, action_delta) * 0.4
        
        height_error = abs(pos[2] - self.target[2])
        r_altitude = 0.5 * np.exp(-height_error * 3.0)
        
        reward = (
            r_distance +
            r_proximity +
            r_velocity +
            r_orientation +
            r_stability +
            r_smoothness +
            r_altitude
        )
        
        is_hovering = dist < 0.25 and speed < 0.25 and tilt_magnitude < 0.12
        if is_hovering:
            self._hover_duration += 1
            hover_bonus = min(self._hover_duration / 15.0, 5.0)
            reward += hover_bonus
        else:
            self._hover_duration = max(0, self._hover_duration - 2)
        
        if pos[2] < 0.2:
            reward -= (0.2 - pos[2]) * 8.0
        
        terminated = False
        
        crashed = (
            pos[2] < self.config.crash_height or
            dist > self.config.crash_distance or
            roll > self.config.crash_angle or
            pitch > self.config.crash_angle
        )

        if crashed:
            reward = -100.0
            terminated = True
        
        if dist < self.config.success_distance and speed < self.config.success_velocity:
            self._success_counter += 1
            reward += 3.0
        else:
            self._success_counter = max(0, self._success_counter - 1)
        
        return float(reward), terminated
    
    def _get_info(self) -> Dict[str, Any]:
        s = self._drone.get_state()
        pos = np.array([s.position.x, s.position.y, s.position.z])
        return {
            'position': pos,
            'target': self.target,
            'distance': np.linalg.norm(self.target - pos),
            'steps': self._steps,
            'episode_reward': self._episode_reward,
            'success_counter': self._success_counter,
            'sub_step_count': self._drone.get_sub_step_count() if self.config.use_sub_stepping else 1
        }
    
    def set_target(self, target: np.ndarray):
        self.target = np.asarray(target, dtype=np.float32)
    
    def get_drone_state(self):
        return self._drone.get_state()
    
    def render(self):
        if self.render_mode == 'human':
            s = self._drone.get_state()
            print(f"Step {self._steps}: pos=({s.position.x:.2f}, {s.position.y:.2f}, {s.position.z:.2f})")


class VectorizedQuadrotorEnv:
    def __init__(
        self,
        num_envs: int,
        config: Optional[EnvConfig] = None,
        task: TaskType = TaskType.HOVER
    ):
        self.num_envs = num_envs
        self.envs = [QuadrotorEnv(config, task) for _ in range(num_envs)]
        
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