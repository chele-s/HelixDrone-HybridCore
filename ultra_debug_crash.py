import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build', 'Release'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict
import json

from python_src.envs.drone_env import QuadrotorEnv, EnvConfig, TaskType

@dataclass
class StepDiagnostic:
    step: int
    action: np.ndarray
    rpm: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    euler: np.ndarray
    angular_velocity: np.ndarray
    reward: float
    reward_components: Dict[str, float]
    crash_reasons: List[str]
    terminated: bool
    truncated: bool

class UltraCrashDiagnostic:
    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.diagnostics: List[StepDiagnostic] = []
        
    def _compute_reward_components(self, env, action) -> Dict[str, float]:
        s = env._drone.get_state()
        pos = np.array([s.position.x, s.position.y, s.position.z])
        vel = np.array([s.velocity.x, s.velocity.y, s.velocity.z])
        ang_vel = np.array([s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z])
        
        euler = s.orientation.to_euler_zyx()
        roll, pitch = abs(euler.x), abs(euler.y)
        
        error_vec = env.target - pos
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
        
        prev_action = env._prev_action if hasattr(env, '_prev_action') else np.zeros(4)
        action_delta = action - prev_action
        r_smoothness = -np.dot(action_delta, action_delta) * 0.4
        
        height_error = abs(pos[2] - env.target[2])
        r_altitude = 0.5 * np.exp(-height_error * 3.0)
        
        hover_duration = env._hover_duration if hasattr(env, '_hover_duration') else 0
        is_hovering = dist < 0.25 and speed < 0.25 and tilt_magnitude < 0.12
        if is_hovering:
            hover_bonus = min((hover_duration + 1) / 15.0, 5.0)
        else:
            hover_bonus = 0.0
            
        height_penalty = 0.0
        if pos[2] < 0.2:
            height_penalty = -(0.2 - pos[2]) * 8.0
        
        return {
            'r_distance': r_distance,
            'r_proximity': r_proximity,
            'r_velocity': r_velocity,
            'r_orientation': r_orientation,
            'r_stability': r_stability,
            'r_smoothness': r_smoothness,
            'r_altitude': r_altitude,
            'r_hover_bonus': hover_bonus,
            'r_height_penalty': height_penalty,
            'dist': dist,
            'speed': speed,
            'tilt_mag_rad': tilt_magnitude,
            'tilt_mag_deg': np.degrees(tilt_magnitude),
            'omega_mag': omega_magnitude
        }
    
    def _check_crash_reasons(self, env) -> List[str]:
        s = env._drone.get_state()
        pos = np.array([s.position.x, s.position.y, s.position.z])
        euler = s.orientation.to_euler_zyx()
        roll, pitch = abs(euler.x), abs(euler.y)
        dist = np.linalg.norm(env.target - pos)
        
        reasons = []
        if pos[2] < self.config.crash_height:
            reasons.append(f"HEIGHT: z={pos[2]:.4f} < {self.config.crash_height}")
        if dist > self.config.crash_distance:
            reasons.append(f"DISTANCE: {dist:.2f} > {self.config.crash_distance}")
        if roll > self.config.crash_angle:
            reasons.append(f"ROLL: {np.degrees(roll):.1f}° > {np.degrees(self.config.crash_angle):.1f}°")
        if pitch > self.config.crash_angle:
            reasons.append(f"PITCH: {np.degrees(pitch):.1f}° > {np.degrees(self.config.crash_angle):.1f}°")
        return reasons
    
    def run_episode(self, action_mode: str = 'zero', max_steps: int = 100, verbose: bool = True):
        env = QuadrotorEnv(config=self.config, task=TaskType.HOVER)
        obs, _ = env.reset()
        self.diagnostics = []
        
        if verbose:
            print("\n" + "=" * 100)
            print(f"ULTRA CRASH DIAGNOSTIC - Mode: {action_mode.upper()}")
            print("=" * 100)
            print(f"\n[CONFIG]")
            print(f"  hover_rpm={self.config.hover_rpm}, rpm_range={self.config.rpm_range}")
            print(f"  crash_height={self.config.crash_height}, crash_angle={np.degrees(self.config.crash_angle):.1f}°")
            print(f"  target={env.target}")
            print(f"  use_sota_actuator={self.config.use_sota_actuator}")
            print(f"  motor_dynamics={self.config.motor_dynamics}")
            print(f"  mass={self.config.mass}")
            
            initial_state = env._drone.get_state()
            print(f"\n[INITIAL STATE]")
            print(f"  Position: ({initial_state.position.x:.4f}, {initial_state.position.y:.4f}, {initial_state.position.z:.4f})")
            print(f"  Velocity: ({initial_state.velocity.x:.4f}, {initial_state.velocity.y:.4f}, {initial_state.velocity.z:.4f})")
            euler = initial_state.orientation.to_euler_zyx()
            print(f"  Euler (deg): roll={np.degrees(euler.x):.2f}, pitch={np.degrees(euler.y):.2f}, yaw={np.degrees(euler.z):.2f}")
            
            print("\n" + "-" * 100)
            print(f"{'Step':>4} | {'Z':>7} | {'Vz':>7} | {'Roll°':>7} | {'Pitch°':>7} | {'|Tilt|°':>7} | {'Dist':>6} | {'RPM_avg':>8} | {'Reward':>8} | {'Reason'}")
            print("-" * 100)
        
        for step_i in range(max_steps):
            if action_mode == 'zero':
                action = np.zeros(4, dtype=np.float32)
            elif action_mode == 'random':
                action = np.random.uniform(-1.0, 1.0, 4).astype(np.float32)
            elif action_mode == 'random_small':
                action = np.random.uniform(-0.1, 0.1, 4).astype(np.float32)
            elif action_mode == 'hover_test':
                action = np.zeros(4, dtype=np.float32)
            else:
                action = np.zeros(4, dtype=np.float32)
            
            reward_components = self._compute_reward_components(env, action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            state = env._drone.get_state()
            pos = np.array([state.position.x, state.position.y, state.position.z])
            vel = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
            euler = state.orientation.to_euler_zyx()
            euler_arr = np.array([euler.x, euler.y, euler.z])
            ang_vel = np.array([state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z])
            rpm = np.array(state.motor_rpm)
            
            crash_reasons = self._check_crash_reasons(env)
            
            diag = StepDiagnostic(
                step=step_i,
                action=action.copy(),
                rpm=rpm.copy(),
                position=pos.copy(),
                velocity=vel.copy(),
                euler=euler_arr.copy(),
                angular_velocity=ang_vel.copy(),
                reward=reward,
                reward_components=reward_components,
                crash_reasons=crash_reasons,
                terminated=terminated,
                truncated=truncated
            )
            self.diagnostics.append(diag)
            
            if verbose:
                roll_deg = np.degrees(abs(euler_arr[0]))
                pitch_deg = np.degrees(abs(euler_arr[1]))
                tilt_deg = np.degrees(np.sqrt(euler_arr[0]**2 + euler_arr[1]**2))
                dist = np.linalg.norm(env.target - pos)
                reason_str = crash_reasons[0] if crash_reasons else ""
                
                print(f"{step_i:>4} | {pos[2]:>7.4f} | {vel[2]:>7.4f} | {roll_deg:>7.2f} | {pitch_deg:>7.2f} | {tilt_deg:>7.2f} | {dist:>6.3f} | {np.mean(rpm):>8.1f} | {reward:>8.2f} | {reason_str}")
            
            if terminated or truncated:
                break
        
        if verbose:
            print("-" * 100)
            self._print_analysis()
        
        return self.diagnostics
    
    def _print_analysis(self):
        if not self.diagnostics:
            return
            
        final = self.diagnostics[-1]
        print(f"\n[EPISODE SUMMARY]")
        print(f"  Total Steps: {len(self.diagnostics)}")
        print(f"  Terminated: {final.terminated}, Truncated: {final.truncated}")
        if final.crash_reasons:
            print(f"  Crash Reason(s): {', '.join(final.crash_reasons)}")
        
        print(f"\n[FINAL STATE]")
        print(f"  Position: ({final.position[0]:.4f}, {final.position[1]:.4f}, {final.position[2]:.4f})")
        print(f"  Velocity: ({final.velocity[0]:.4f}, {final.velocity[1]:.4f}, {final.velocity[2]:.4f})")
        print(f"  Euler (deg): roll={np.degrees(final.euler[0]):.2f}, pitch={np.degrees(final.euler[1]):.2f}")
        print(f"  RPM: [{', '.join([f'{r:.0f}' for r in final.rpm])}]")
        
        print(f"\n[PHYSICS ANALYSIS]")
        
        z_values = [d.position[2] for d in self.diagnostics]
        vz_values = [d.velocity[2] for d in self.diagnostics]
        
        if len(z_values) > 1:
            dz = np.diff(z_values)
            if np.all(dz < 0):
                print("  ⚠️  FALLING: Z is monotonically decreasing - drone is NOT gaining altitude")
                avg_fall_rate = np.mean(dz) / 0.01
                print(f"       Average fall rate: {avg_fall_rate:.4f} m/s (per step)")
            elif np.all(dz > 0):
                print("  ⚠️  ROCKETING: Z is monotonically increasing - too much thrust")
            else:
                print("  ✓  Z shows mixed behavior - some control happening")
        
        tilt_values = [np.sqrt(d.euler[0]**2 + d.euler[1]**2) for d in self.diagnostics]
        max_tilt = max(tilt_values)
        max_tilt_step = tilt_values.index(max_tilt)
        print(f"\n  Max Tilt: {np.degrees(max_tilt):.2f}° at step {max_tilt_step}")
        
        tilt_crash_angle = self.config.crash_angle
        if max_tilt > tilt_crash_angle * 0.8:
            print(f"  ⚠️  TILT WARNING: Approaching crash angle ({np.degrees(tilt_crash_angle):.1f}°)")
        
        print(f"\n[REWARD BREAKDOWN - Last 5 Steps]")
        for d in self.diagnostics[-5:]:
            rc = d.reward_components
            print(f"  Step {d.step}: total={d.reward:.2f}")
            print(f"    dist={rc['r_distance']:.2f}, prox={rc['r_proximity']:.2f}, vel={rc['r_velocity']:.2f}")
            print(f"    orient={rc['r_orientation']:.2f}, stab={rc['r_stability']:.2f}, smooth={rc['r_smoothness']:.2f}")
            print(f"    ||| dist_to_target={rc['dist']:.3f}, speed={rc['speed']:.3f}, tilt_deg={rc['tilt_mag_deg']:.1f}")
        
        print(f"\n[ROOT CAUSE HYPOTHESIS]")
        
        if len(self.diagnostics) < 50:
            if final.crash_reasons:
                for reason in final.crash_reasons:
                    if "HEIGHT" in reason:
                        print("  🔴 GRAVITY PROBLEM: Drone fell to ground")
                        print("     → Check if hover_rpm produces enough thrust for mass")
                        print("     → Current: hover_rpm={}, mass={}".format(self.config.hover_rpm, self.config.mass))
                        
                        thrust_coefficient = 1e-9
                        gravity = 9.81
                        required_thrust = self.config.mass * gravity
                        hover_thrust_per_motor = thrust_coefficient * (self.config.hover_rpm ** 2)
                        total_hover_thrust = 4 * hover_thrust_per_motor
                        print(f"     → Estimated hover thrust: {total_hover_thrust:.4f} N vs Required: {required_thrust:.4f} N")
                        
                    if "ROLL" in reason or "PITCH" in reason:
                        print("  🔴 STABILITY PROBLEM: Drone flipped over")
                        print("     → Check if rpm_range is too aggressive")
                        print("     → Current: rpm_range={} (actions can change RPM by ±{})".format(
                            self.config.rpm_range, self.config.rpm_range))
                        max_rpm_diff = self.config.rpm_range
                        arm_length = 0.25
                        print(f"     → Max torque differential creates extreme rotation")
        else:
            print("  ✓ Episode survived >50 steps - physics may be stable")
        
        print("\n" + "=" * 100)


def main():
    print("\n" + "🔬" * 30)
    print("ULTRA CRASH DIAGNOSTIC SUITE")
    print("🔬" * 30)
    
    config = EnvConfig()
    config.use_sota_actuator = True
    config.hover_rpm = 2600.0
    config.rpm_range = 3600.0
    config.mass = 0.6
    config.domain_randomization = False
    config.wind_enabled = False
    config.curriculum_enabled = False
    
    diagnostician = UltraCrashDiagnostic(config)
    
    print("\n\n[TEST 1] ZERO ACTION (Pure Hover Test)")
    print("If this fails, physics/thrust is fundamentally broken")
    diagnostician.run_episode(action_mode='zero', max_steps=100)
    
    print("\n\n[TEST 2] SMALL RANDOM ACTIONS (±0.1)")
    print("If this fails but TEST 1 passes, control authority is too sensitive")
    diagnostician.run_episode(action_mode='random_small', max_steps=100)
    
    print("\n\n[TEST 3] FULL RANDOM ACTIONS (±1.0)")
    print("Simulating untrained agent exploration")
    diagnostician.run_episode(action_mode='random', max_steps=100)
    
    print("\n\n[MULTI-EPISODE STATISTICS]")
    survival_counts = []
    crash_types = defaultdict(int)
    
    for i in range(20):
        diags = diagnostician.run_episode(action_mode='random', max_steps=200, verbose=False)
        survival_counts.append(len(diags))
        if diags[-1].crash_reasons:
            for reason in diags[-1].crash_reasons:
                key = reason.split(":")[0]
                crash_types[key] += 1
    
    print(f"  Average Survival: {np.mean(survival_counts):.1f} steps")
    print(f"  Min Survival: {min(survival_counts)} steps")
    print(f"  Max Survival: {max(survival_counts)} steps")
    print(f"  Std Dev: {np.std(survival_counts):.1f}")
    
    print(f"\n  Crash Type Distribution (over 20 episodes):")
    for crash_type, count in sorted(crash_types.items(), key=lambda x: -x[1]):
        print(f"    {crash_type}: {count} ({count/20*100:.0f}%)")
    
    print("\n" + "=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
