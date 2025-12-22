import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'Release'))

import gymnasium as gym
import numpy as np
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
    max_rpm: float = 8000.0
    min_rpm: float = 1000.0
    hover_rpm: float = 2600.0
    rpm_range: float = 1500.0
    
    position_scale: float = 5.0
    velocity_scale: float = 5.0
    angular_velocity_scale: float = 10.0
    
    reward_position: float = -0.5
    reward_velocity: float = -0.025
    reward_angular: float = -0.01
    reward_action: float = -0.0005
    reward_action_rate: float = -0.0025
    reward_alive: float = 1.0
    reward_crash: float = -10.0
    reward_success: float = 50.0
    
    reward_height_bonus: float = 0.5
    reward_stability_bonus: float = 0.25
    reward_hover_bonus: float = 2.5
    
    reward_saturation_penalty: float = -0.01
    saturation_threshold: float = 0.95
    
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
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        
        self.target = target if target is not None else np.array([0.0, 0.0, 2.0])
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._success_counter = 0
        self._steps = 0
        self._episode_reward = 0.0
        self._rng = np.random.default_rng()
        self._total_episodes = 0
        self._curriculum_progress = 0.0
        self._saturation_count = 0
    
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
        
        self._cfg.mass = 1.0
        self._cfg.arm_length = 0.25
        
        self._cfg.rotor.radius = 0.127
        self._cfg.rotor.chord = 0.02
        self._cfg.rotor.pitch_angle = 0.26
        self._cfg.rotor.flapping.enabled = True
        self._cfg.rotor.flapping.lock_number = 8.0
        
        self._cfg.motor.kv = 2300
        self._cfg.motor.max_current = 30
        self._cfg.motor.esc.nonlinear_gamma = 1.2
        
        self._cfg.aero.air_density = 1.225
        self._cfg.aero.ground_effect_coeff = 0.5
        
        self._drone = drone_core.Quadrotor(self._cfg)
    
    def _apply_domain_randomization(self):
        if not self.config.domain_randomization:
            return
        
        mass_factor = self._rng.uniform(0.9, 1.1)
        self._cfg.mass = 1.0 * mass_factor
        
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
        self._saturation_count = 0
        
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
        
        rpm = self.config.hover_rpm + action * self.config.rpm_range
        rpm = np.clip(rpm, self.config.min_rpm, self.config.max_rpm)
        
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
        
        prev_action = self._prev_action
        
        return np.concatenate([pos, vel, quat, ang_vel, error, prev_action]).astype(np.float32)
    
    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool]:
        s = self._drone.get_state()
        
        pos = np.array([s.position.x, s.position.y, s.position.z])
        vel = np.array([s.velocity.x, s.velocity.y, s.velocity.z])
        ang_vel = np.array([s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z])
        
        euler = s.orientation.to_euler_zyx()
        roll, pitch = abs(euler.x), abs(euler.y)
        
        dist = np.linalg.norm(self.target - pos)
        speed = np.linalg.norm(vel)
        ang_speed = np.linalg.norm(ang_vel)
        
        action_norm = np.linalg.norm(action)
        action_rate_norm = np.linalg.norm(action - self._prev_action)
        
        reward = self.config.reward_alive
        
        dist_reward = np.exp(-dist * 0.5)
        reward += dist_reward * 1.0
        
        vel_reward = np.exp(-speed * 0.5)
        reward += vel_reward * 0.25
        
        stability_factor = np.exp(-(roll + pitch) * 3.0)
        reward += self.config.reward_stability_bonus * stability_factor
        
        reward += self.config.reward_angular * min(ang_speed, 10.0)
        reward += self.config.reward_action * action_norm
        reward += self.config.reward_action_rate * action_rate_norm
        
        saturated_actions = np.sum(np.abs(action) > self.config.saturation_threshold)
        if saturated_actions > 0:
            self._saturation_count += 1
            reward += self.config.reward_saturation_penalty * saturated_actions
            
            saturation_severity = np.sum(np.maximum(0, np.abs(action) - self.config.saturation_threshold))
            reward += self.config.reward_saturation_penalty * saturation_severity * 2.0
        
        is_hovering = dist < 1.0 and speed < 1.0 and roll < 0.3 and pitch < 0.3
        if is_hovering:
            self._hover_duration += 1
            hover_bonus = min(self._hover_duration / 50.0, 2.0) * self.config.reward_hover_bonus
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

        if crashed:
            survival_bonus = min(1.0, self._steps / 200.0)
            reward += self.config.reward_crash * (1.0 - 0.3 * survival_bonus)
            terminated = True
        
        if dist < self.config.success_distance and speed < self.config.success_velocity:
            self._success_counter += 1
            reward += 0.5
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
            'saturation_count': self._saturation_count,
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