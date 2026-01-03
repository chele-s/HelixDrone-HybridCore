"""Composable reward functions for drone environments."""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Protocol, Optional
from enum import IntEnum


@dataclass
class RewardConfig:
    # Exponential position: R = weight * exp(-decay * dist)
    position_exp_weight: float = 2.0
    position_exp_decay: float = 4.0
    
    alive_bonus: float = 0.5
    progress_weight: float = 5.0
    action_rate_weight: float = -0.1
    action_magnitude_weight: float = -0.01
    angular_velocity_weight: float = -0.05
    
    # Legacy (compatibility)
    velocity_weight: float = 0.0
    orientation_weight: float = 0.0
    velocity_toward_weight: float = 0.0
    proximity_weight: float = 0.0
    action_accel_weight: float = 0.0
    hover_bonus: float = 0.0
    stability_weight: float = 0.0
    efficiency_weight: float = 0.0
    position_weight: float = 0.0
    
    # Termination
    crash_penalty: float = -10.0
    success_bonus: float = 100.0
    
    ground_penalty_weight: float = -5.0
    ground_threshold: float = 0.4
    
    payload_swing_weight: float = -0.5
    payload_position_weight: float = -0.3
    payload_energy_weight: float = -0.1
    
    collision_penalty: float = -10.0
    thrust_activity_weight: float = 0.3
    
    success_distance: float = 0.1     
    success_velocity: float = 0.2     
    
    crash_height: float = 0.05        
    crash_distance: float = 3.0      
    crash_angle: float = 1.0         
    crash_velocity: float = 10.0


@dataclass
class RewardState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    euler_angles: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    target_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    action: np.ndarray = field(default_factory=lambda: np.zeros(4))
    action_rate: np.ndarray = field(default_factory=lambda: np.zeros(4))
    action_accel: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    motor_rpm: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hover_rpm: float = 5500.0
    
    payload_swing_angle: float = 0.0
    payload_position: Optional[np.ndarray] = None
    payload_energy: float = 0.0
    
    prev_distance: float = 0.0
    hover_duration: int = 0
    
    collision_occurred: bool = False


class RewardTerm(Protocol):
    @property
    def name(self) -> str: ...
    def compute(self, state: RewardState, config: RewardConfig) -> float: ...


class ProgressReward:
    @property
    def name(self) -> str:
        return "progress"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        dist = np.linalg.norm(state.target_position - state.position)
        progress = state.prev_distance - dist
        return progress * config.progress_weight


class VelocityTowardReward:
    @property
    def name(self) -> str:
        return "velocity_toward"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        error_vec = state.target_position - state.position
        dist = np.linalg.norm(error_vec)
        if dist < 1e-6:
            return 0.0
        
        target_dir = error_vec / dist
        vel_toward = np.dot(state.velocity, target_dir)
        
        reward = vel_toward * config.velocity_toward_weight * 0.3
        if vel_toward > 0:
            reward *= 1.5
        return reward


class ProximityReward:
    @property
    def name(self) -> str:
        return "proximity"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        dist = np.linalg.norm(state.target_position - state.position)
        return config.position_exp_weight * np.exp(-config.position_exp_decay * dist)


class OrientationReward:
    @property
    def name(self) -> str:
        return "orientation"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        roll = abs(state.euler_angles[0])
        pitch = abs(state.euler_angles[1])
        tilt_penalty = roll + pitch
        return tilt_penalty * config.orientation_weight


class StabilityReward:
    @property
    def name(self) -> str:
        return "stability"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        omega_mag = np.linalg.norm(state.angular_velocity)
        return omega_mag * config.stability_weight


class ActionSmoothnessReward:
    @property
    def name(self) -> str:
        return "action_smoothness"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        rate_penalty = np.sum(state.action_rate ** 2) * config.action_rate_weight
        accel_penalty = np.sum(state.action_accel ** 2) * config.action_accel_weight
        return rate_penalty + accel_penalty


class EfficiencyReward:
    @property
    def name(self) -> str:
        return "efficiency"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        rpm_deviation = np.mean((state.motor_rpm - state.hover_rpm) ** 2)
        return rpm_deviation * config.efficiency_weight


class GroundProximityPenalty:
    @property
    def name(self) -> str:
        return "ground_proximity"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        if state.position[2] < config.ground_threshold:
            return (config.ground_threshold - state.position[2]) * config.ground_penalty_weight
        return 0.0


class AliveBonus:
    @property
    def name(self) -> str:
        return "alive"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        return config.alive_bonus


class HoverBonus:
    @property
    def name(self) -> str:
        return "hover"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        dist = np.linalg.norm(state.target_position - state.position)
        speed = np.linalg.norm(state.velocity)
        roll = abs(state.euler_angles[0])
        pitch = abs(state.euler_angles[1])
        
        is_hovering = dist < 0.2 and speed < 0.3 and roll < 0.1 and pitch < 0.1
        
        if is_hovering:
            return config.hover_bonus
        return 0.0


class PayloadSwingPenalty:
    @property
    def name(self) -> str:
        return "payload_swing"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        return abs(state.payload_swing_angle) * config.payload_swing_weight


class PayloadPositionReward:
    @property
    def name(self) -> str:
        return "payload_position"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        if state.payload_position is None:
            return 0.0
        
        payload_target = state.target_position.copy()
        payload_target[2] -= 1.0
        
        payload_error = np.linalg.norm(state.payload_position - payload_target)
        return payload_error * config.payload_position_weight


class PayloadEnergyPenalty:
    @property
    def name(self) -> str:
        return "payload_energy"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        return abs(state.payload_energy) * config.payload_energy_weight


class ThrustActivityBonus:
    @property
    def name(self) -> str:
        return "thrust_activity"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        mean_rpm = np.mean(state.motor_rpm)
        ratio = mean_rpm / state.hover_rpm if state.hover_rpm > 0 else 0
        if ratio < 0.85:
            return -config.thrust_activity_weight * 2.0
        elif ratio < 0.95:
            return -config.thrust_activity_weight * 0.5
        elif ratio > 1.15:
            return config.thrust_activity_weight * 0.5
        return config.thrust_activity_weight


class CollisionPenalty:
    @property
    def name(self) -> str:
        return "collision"
    
    def compute(self, state: RewardState, config: RewardConfig) -> float:
        return config.collision_penalty if state.collision_occurred else 0.0


class RewardBuilder:
    def __init__(self, config: Optional[RewardConfig] = None, include_payload: bool = False):
        self.config = config or RewardConfig()
        self.terms: List[RewardTerm] = []
        self._setup_terms(include_payload)
        
        self._prev_distance = 0.0
        self._hover_duration = 0
    
    def _setup_terms(self, include_payload: bool):
        self.terms.append(ProgressReward())
        self.terms.append(VelocityTowardReward())
        self.terms.append(ProximityReward())
        self.terms.append(OrientationReward())
        self.terms.append(StabilityReward())
        self.terms.append(ActionSmoothnessReward())
        self.terms.append(EfficiencyReward())
        self.terms.append(GroundProximityPenalty())
        self.terms.append(AliveBonus())
        self.terms.append(HoverBonus())
        self.terms.append(ThrustActivityBonus())
        
        if include_payload:
            self.terms.append(PayloadSwingPenalty())
            self.terms.append(PayloadPositionReward())
            self.terms.append(PayloadEnergyPenalty())
        
        self.terms.append(CollisionPenalty())
    
    def compute(self, state: RewardState) -> float:
        state.prev_distance = self._prev_distance
        state.hover_duration = self._hover_duration
        
        dist = np.linalg.norm(state.target_position - state.position)
        speed = np.linalg.norm(state.velocity)
        roll = abs(state.euler_angles[0])
        pitch = abs(state.euler_angles[1])
        
        is_hovering = dist < 0.3 and speed < 0.4 and roll < 0.12 and pitch < 0.12
        if is_hovering:
            self._hover_duration += 1
        else:
            self._hover_duration = max(0, self._hover_duration - 2)
        
        total = sum(term.compute(state, self.config) for term in self.terms)
        
        self._prev_distance = dist
        
        return float(total)
    
    def compute_with_breakdown(self, state: RewardState) -> tuple:
        state.prev_distance = self._prev_distance
        state.hover_duration = self._hover_duration
        
        breakdown = {}
        total = 0.0
        
        for term in self.terms:
            value = term.compute(state, self.config)
            breakdown[term.name] = value
            total += value
        
        dist = np.linalg.norm(state.target_position - state.position)
        self._prev_distance = dist
        
        speed = np.linalg.norm(state.velocity)
        roll = abs(state.euler_angles[0])
        pitch = abs(state.euler_angles[1])
        is_hovering = dist < 0.3 and speed < 0.4 and roll < 0.12 and pitch < 0.12
        if is_hovering:
            self._hover_duration += 1
        else:
            self._hover_duration = max(0, self._hover_duration - 2)
        
        return float(total), breakdown
    
    def check_termination(self, state: RewardState) -> tuple:
        crashed = False
        crash_reason = None
        
        if state.position[2] < self.config.crash_height:
            crashed = True
            crash_reason = "ground"
        
        dist = np.linalg.norm(state.target_position - state.position)
        if dist > self.config.crash_distance:
            crashed = True
            crash_reason = "distance"
        
        roll = abs(state.euler_angles[0])
        pitch = abs(state.euler_angles[1])
        if roll > self.config.crash_angle or pitch > self.config.crash_angle:
            crashed = True
            crash_reason = "angle"
        
        speed = np.linalg.norm(state.velocity)
        if speed > self.config.crash_velocity:
            crashed = True
            crash_reason = "velocity"
        
        return crashed, crash_reason
    
    def check_success(self, state: RewardState) -> bool:
        dist = np.linalg.norm(state.target_position - state.position)
        speed = np.linalg.norm(state.velocity)
        return dist < self.config.success_distance and speed < self.config.success_velocity
    
    def reset(self):
        self._prev_distance = 0.0
        self._hover_duration = 0
