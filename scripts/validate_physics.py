import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'Release'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'build'))

import numpy as np
from python_src.envs.drone_env import QuadrotorEnv, ExtendedEnvConfig, TaskType

ROTOR_RADIUS = 0.127
ROTOR_CHORD = 0.02
ROTOR_PITCH = 0.18
LIFT_SLOPE = 5.7
INFLOW_RATIO = 0.05
AIR_DENSITY = 1.225


def compute_thrust_bet(rpm: float) -> float:
    if rpm <= 0:
        return 0.0
    omega = rpm * np.pi / 30.0
    R = ROTOR_RADIUS
    c = ROTOR_CHORD
    theta = ROTOR_PITCH
    a = LIFT_SLOPE
    lam = INFLOW_RATIO
    solidity = 4.0 * c / (np.pi * R)
    Ct = solidity * a * (theta / 3.0 - lam / 2.0) / 2.0
    thrust = Ct * AIR_DENSITY * np.pi * R * R * (omega * R) * (omega * R)
    return thrust


def find_hover_rpm(mass_kg: float) -> float:
    weight = mass_kg * 9.81
    thrust_per_rotor = weight / 4.0
    R = ROTOR_RADIUS
    c = ROTOR_CHORD
    theta = ROTOR_PITCH
    a = LIFT_SLOPE
    lam = INFLOW_RATIO
    solidity = 4.0 * c / (np.pi * R)
    Ct = solidity * a * (theta / 3.0 - lam / 2.0) / 2.0
    coefficient = Ct * AIR_DENSITY * np.pi * R**4
    omega = np.sqrt(thrust_per_rotor / coefficient)
    return omega * 30.0 / np.pi


def validate_physics():
    print("=" * 70)
    print("PHYSICS VALIDATION - Pre-Training Check")
    print("=" * 70)
    
    config = ExtendedEnvConfig()
    env = QuadrotorEnv(config=config, task=TaskType.HOVER)
    
    print(f"\n[CONFIG]")
    print(f"  hover_rpm:  {config.hover_rpm}")
    print(f"  rpm_range:  {config.rpm_range}")
    print(f"  mass:       {config.mass} kg")
    
    theoretical_hover = find_hover_rpm(config.mass)
    weight = config.mass * 9.81
    thrust_at_hover = 4 * compute_thrust_bet(config.hover_rpm)
    net_force = thrust_at_hover - weight
    accel = net_force / config.mass
    
    print(f"\n[PHYSICS CHECK]")
    print(f"  Theoretical hover RPM: {theoretical_hover:.0f}")
    print(f"  Configured hover RPM:  {config.hover_rpm:.0f}")
    print(f"  Thrust at hover_rpm:   {thrust_at_hover:.3f} N")
    print(f"  Weight:                {weight:.3f} N")
    print(f"  Net force:             {net_force:+.3f} N")
    print(f"  Acceleration:          {accel:+.3f} m/s²")
    
    if accel < -0.3:
        print(f"\n  ❌ FAIL: Drone will FALL with action=0 (accel={accel:.2f} m/s²)")
        return False
    elif accel > 1.5:
        print(f"\n  ⚠️  WARNING: Drone will RISE fast with action=0")
    else:
        print(f"\n  ✓ PASS: Physics calibration looks correct")
    
    print(f"\n[SIMULATION TEST]")
    obs, _ = env.reset()
    neutral_action = np.array([0.0, 0.0, 0.0, 0.0])
    
    heights = []
    for step in range(200):
        obs, reward, done, truncated, info = env.step(neutral_action)
        heights.append(info['position'][2])
        if done:
            print(f"  ❌ Crashed at step {step}: {info.get('crash_reason', 'unknown')}")
            break
    
    if not done:
        height_change = heights[-1] - heights[0]
        print(f"  Initial height: {heights[0]:.2f} m")
        print(f"  Final height:   {heights[-1]:.2f} m")
        print(f"  Height change:  {height_change:+.2f} m")
        
        if height_change < -0.5:
            print(f"\n  ❌ FAIL: Drone falls with action=0")
            return False
        else:
            print(f"\n  ✓ PASS: Drone stable with action=0")
    
    print(f"\n[ACTION RANGE TEST]")
    actions_to_test = [
        ("action=-1.0", np.array([-1.0, 0.0, 0.0, 0.0])),
        ("action=-0.5", np.array([-0.5, 0.0, 0.0, 0.0])),
        ("action= 0.0", np.array([0.0, 0.0, 0.0, 0.0])),
        ("action=+0.5", np.array([0.5, 0.0, 0.0, 0.0])),
        ("action=+1.0", np.array([1.0, 0.0, 0.0, 0.0])),
    ]
    
    for name, action in actions_to_test:
        obs, _ = env.reset()
        for _ in range(50):
            obs, _, done, _, info = env.step(action)
            if done:
                break
        if not done:
            rpm_mean = info['rpm_mean']
            print(f"  {name} -> RPM: {rpm_mean:.0f}")
        else:
            print(f"  {name} -> Crashed")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = validate_physics()
    sys.exit(0 if success else 1)
