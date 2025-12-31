import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'Release'))

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
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
    max_steps: int = 2000
    max_rpm: float = 35000.0
    min_rpm: float = 2000.0
    hover_rpm: float = 2750.0
    rpm_range: float = 2000.0
    mass: float = 0.6
    
    position_scale: float = 5.0
    velocity_scale: float = 5.0
    angular_velocity_scale: float = 10.0
    
    reward_position: float = -0.5
    reward_velocity: float = -0.02
    reward_angular: float = -0.01
    reward_alive: float = 0.3
    reward_crash: float = -50.0
    reward_success: float = 100.0
    reward_velocity_toward: float = 3.0
    reward_orientation: float = 2.0
    reward_smoothness: float = -0.3
    reward_action_rate: float = -0.15
    reward_action_accel: float = -0.08
    
    reward_height_bonus: float = 0.3
    reward_stability_bonus: float = 0.2
    reward_hover_bonus: float = 3.0
    reward_progress: float = 4.0
    
    crash_height: float = 0.05
    crash_distance: float = 15.0
    crash_angle: float = 1.2
    crash_velocity: float = 8.0
    success_distance: float = 0.25
    success_velocity: float = 0.4
    success_hold_steps: int = 50
    
    curriculum_enabled: bool = True
    curriculum_init_range: float = 0.02
    curriculum_max_range: float = 0.6
    curriculum_progress_rate: float = 0.00005
    
    target_curriculum_enabled: bool = True
    target_speed_init: float = 0.0
    target_speed_max: float = 2.5
    target_speed_curriculum_rate: float = 0.00003
    
    domain_randomization: bool = True
    mass_randomization: Tuple[float, float] = (0.85, 1.15)
    wind_enabled: bool = True
    wind_speed_range: Tuple[float, float] = (0.0, 4.0)
    motor_dynamics: bool = True
    use_sub_stepping: bool = True
    
    observation_noise: float = 0.01
    action_delay_steps: int = 0


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
        
        elif self._mode == 'random_walk':
            if self._t < dt * 1.5:
                self._velocity = np.random.uniform(-1, 1, 3) * self._speed
                self._velocity[2] *= 0.3
            
            if np.random.random() < 0.01:
                self._velocity = np.random.uniform(-1, 1, 3) * self._speed
                self._velocity[2] *= 0.3
            
            self._position += self._velocity * dt
            self._position[2] = np.clip(self._position[2], 0.5, 5.0)
            self._position[:2] = np.clip(self._position[:2], -5.0, 5.0)
        
        return self._position.copy(), self._velocity.copy()
    
    def reset(self, center: np.ndarray = None):
        self._t = 0.0
        self._phase = np.random.uniform(0, 2*np.pi)
        if center is not None:
            self._center = center
        self._position = self._center.copy()
        self._velocity = np.zeros(3)


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
        self._setup_sota_actuator()
        
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )
        
        self._target_generator = TargetGenerator(
            mode='static' if task == TaskType.HOVER else 'figure8',
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
        self._hover_duration = 0
        self._prev_dist = 0.0
        self._prev_vel_toward = 0.0
    
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
        
        if self.config.target_curriculum_enabled:
            self._target_speed_progress = min(1.0, self._target_speed_progress + self.config.target_speed_curriculum_rate)
            target_speed = self.config.target_speed_init + (
                self.config.target_speed_max - self.config.target_speed_init
            ) * self._target_speed_progress
            self._target_generator.set_speed(target_speed)
        
        target_center = np.array([
            self._rng.uniform(-2 * curr_range, 2 * curr_range),
            self._rng.uniform(-2 * curr_range, 2 * curr_range),
            2.0 + self._rng.uniform(-0.5 * curr_range, 0.5 * curr_range)
        ])
        self._target_generator.reset(target_center)
        self.target, self._target_velocity = self._target_generator.update(0.0)
        
        if self.task == TaskType.HOVER:
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
        
        elif self.task == TaskType.WAYPOINT:
            self._drone.set_position(drone_core.Vec3(0, 0, 1.0))
            self.target = np.array([
                self._rng.uniform(-4, 4),
                self._rng.uniform(-4, 4),
                self._rng.uniform(1, 4)
            ])
        
        self._action_history.reset()
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._success_counter = 0
        self._steps = 0
        self._episode_reward = 0.0
        self._total_episodes += 1
        self._hover_duration = 0
        self._prev_rpm = np.full(4, self.config.hover_rpm, dtype=np.float64)
        self._sota_actuator.reset()
        
        s = self._drone.get_state()
        pos = np.array([s.position.x, s.position.y, s.position.z])
        self._prev_dist = np.linalg.norm(self.target - pos)
        self._prev_vel_toward = 0.0
        
        if self.config.motor_dynamics:
            warmup_rpm = self.config.hover_rpm
            warmup_cmd = drone_core.MotorCommand(warmup_rpm, warmup_rpm, warmup_rpm, warmup_rpm)
            for _ in range(5):
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
        
        self.target, self._target_velocity = self._target_generator.update(self.config.dt)
        
        self._steps += 1
        
        obs = self._get_obs()
        reward, terminated = self._compute_reward(action)
        truncated = self._steps >= self.config.max_steps
        
        self._prev_action = action.copy()
        self._episode_reward += reward
        
        return obs, reward, terminated, truncated, self._get_info()
    
    def _get_obs(self) -> np.ndarray:
        s = self._drone.get_state()
        
        pos = np.array([s.position.x, s.position.y, s.position.z])
        vel = np.array([s.velocity.x, s.velocity.y, s.velocity.z])
        
        pos_scaled = pos / self.config.position_scale
        vel_scaled = vel / self.config.velocity_scale
        
        quat = np.array([s.orientation.w, s.orientation.x, s.orientation.y, s.orientation.z])
        
        ang_vel = np.array([
            s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z
        ]) / self.config.angular_velocity_scale
        
        error = (self.target - pos) / self.config.position_scale
        
        relative_vel = (self._target_velocity - vel) / self.config.velocity_scale
        
        target_vel_scaled = self._target_velocity / self.config.velocity_scale
        
        if self.config.observation_noise > 0:
            noise = self._rng.normal(0, self.config.observation_noise, size=26)
        else:
            noise = np.zeros(26)
        
        obs = np.concatenate([
            pos_scaled,
            vel_scaled,
            quat,
            ang_vel,
            error,
            relative_vel,
            target_vel_scaled,
            self._prev_action
        ]).astype(np.float32)
        
        return obs + noise.astype(np.float32)
    
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
        
        target_dir = error_vec / (dist + 1e-8)
        vel_toward = np.dot(vel, target_dir)
        
        r_progress = (self._prev_dist - dist) * self.config.reward_progress
        self._prev_dist = dist
        
        r_vel_toward = vel_toward * self.config.reward_velocity_toward * 0.3
        if vel_toward > 0:
            r_vel_toward *= 1.5
        
        r_proximity = np.exp(-dist * 1.5) * 2.0
        
        r_orientation = np.exp(-(roll + pitch) * 4.0) * self.config.reward_orientation
        
        omega_mag = np.linalg.norm(ang_vel)
        r_stability = -omega_mag * 0.05
        
        action_rate = self._action_history.get_rate()
        action_accel = self._action_history.get_accel()
        
        r_smooth = -np.sum(action_rate ** 2) * self.config.reward_action_rate
        r_smooth += -np.sum(action_accel ** 2) * self.config.reward_action_accel
        
        if hasattr(self, '_current_rpm'):
            hover_rpm = self.config.hover_rpm
            rpm_deviation = np.mean((self._current_rpm - hover_rpm) ** 2)
            r_efficiency = -rpm_deviation * 5e-9
        else:
            r_efficiency = 0.0
        
        if pos[2] < 0.4:
            r_ground = -(0.4 - pos[2]) * 3.0
        else:
            r_ground = 0.0
        
        r_alive = self.config.reward_alive
        
        reward = (
            r_progress +
            r_vel_toward +
            r_proximity +
            r_orientation +
            r_stability +
            r_smooth +
            r_efficiency +
            r_ground +
            r_alive
        )
        
        is_hovering = dist < 0.3 and speed < 0.4 and roll < 0.12 and pitch < 0.12
        if is_hovering:
            self._hover_duration += 1
            hover_bonus = min(self._hover_duration / 30.0, self.config.reward_hover_bonus)
            reward += hover_bonus
        else:
            self._hover_duration = max(0, self._hover_duration - 2)
        
        terminated = False
        crashed = False
        
        if pos[2] < self.config.crash_height:
            crashed = True
        elif dist > self.config.crash_distance:
            crashed = True
        elif roll > self.config.crash_angle or pitch > self.config.crash_angle:
            crashed = True
        elif speed > self.config.crash_velocity:
            crashed = True
        
        if crashed:
            reward = self.config.reward_crash
            terminated = True
        
        if dist < self.config.success_distance and speed < self.config.success_velocity:
            self._success_counter += 1
            if self._success_counter >= self.config.success_hold_steps:
                reward += self.config.reward_success
        else:
            self._success_counter = max(0, self._success_counter - 1)
        
        return float(reward), terminated
    
    def _get_info(self) -> Dict[str, Any]:
        s = self._drone.get_state()
        pos = np.array([s.position.x, s.position.y, s.position.z])
        return {
            'position': pos,
            'target': self.target,
            'target_velocity': self._target_velocity,
            'distance': np.linalg.norm(self.target - pos),
            'steps': self._steps,
            'episode_reward': self._episode_reward,
            'success_counter': self._success_counter,
            'curriculum_progress': self._curriculum_progress,
            'target_speed_progress': self._target_speed_progress,
            'sub_step_count': self._drone.get_sub_step_count() if self.config.use_sub_stepping else 1
        }
    
    def set_target(self, target: np.ndarray):
        self.target = np.asarray(target, dtype=np.float32)
        self._target_generator.reset(self.target)
    
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