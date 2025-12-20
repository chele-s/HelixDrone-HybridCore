import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))

from python_src.envs.drone_env import QuadrotorEnv, EnvConfig, TaskType

def compute_thrust_per_rotor(rpm, radius=0.127, chord=0.02, pitch_angle=0.26, 
                              lift_slope=5.7, inflow_ratio=0.05, air_density=1.225):
    if rpm <= 0:
        return 0
    omega = rpm * np.pi / 30.0
    R = radius
    c = chord
    theta = pitch_angle
    a = lift_slope
    lam = inflow_ratio
    
    solidity = 4.0 * c / (np.pi * R)
    Ct = solidity * a * (theta / 3.0 - lam / 2.0) / 2.0
    thrust = Ct * air_density * np.pi * R * R * (omega * R) * (omega * R)
    return thrust

mass = 1.0
gravity = 9.81
target_thrust = mass * gravity

print(f"Required thrust: {target_thrust:.2f} N")
print(f"Required thrust per rotor: {target_thrust/4:.2f} N")

for rpm in [5000, 6000, 7000, 8000, 9000, 10000]:
    thrust = compute_thrust_per_rotor(rpm)
    total = 4 * thrust
    print(f"RPM: {rpm}, Thrust/rotor: {thrust:.3f} N, Total: {total:.2f} N, Excess: {total - target_thrust:.2f} N")

print("\nFinding hover RPM...")
rpm_low, rpm_high = 1000, 20000
while rpm_high - rpm_low > 1:
    rpm_mid = (rpm_low + rpm_high) / 2
    total_thrust = 4 * compute_thrust_per_rotor(rpm_mid)
    if total_thrust < target_thrust:
        rpm_low = rpm_mid
    else:
        rpm_high = rpm_mid

hover_rpm = (rpm_low + rpm_high) / 2
print(f"Calculated hover RPM: {hover_rpm:.0f}")
print(f"Thrust at hover RPM: {4 * compute_thrust_per_rotor(hover_rpm):.3f} N")

print("\n--- Testing hover with default config ---")

config = EnvConfig(
    motor_dynamics=False,
    wind_enabled=False,
    domain_randomization=False,
    curriculum_enabled=False
)

env = QuadrotorEnv(config=config, task=TaskType.HOVER)
obs, info = env.reset()

print(f"Initial position: {info['position']}")
print(f"hover_rpm: {config.hover_rpm:.0f}")

action = np.zeros(4, dtype=np.float32)
total_reward = 0

for step in range(100):
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    state = env.get_drone_state()
    pos = np.array([state.position.x, state.position.y, state.position.z])
    euler = state.orientation.to_euler_zyx()
    roll, pitch = abs(euler.x), abs(euler.y)
    dist = np.linalg.norm(np.array([0.0, 0.0, 2.0]) - pos)
    
    if step % 10 == 0 or terminated:
        print(f"Step {step}: z={pos[2]:.3f}, vz={state.velocity.z:.4f}, dist={dist:.3f}, roll={roll:.3f}, pitch={pitch:.3f}, r={reward:.2f}")
    
    if terminated:
        if pos[2] < 0.03:
            print("  -> Crashed due to height")
        elif dist > 10.0:
            print("  -> Crashed due to distance")
        elif roll > 1.4 or pitch > 1.4:
            print("  -> Crashed due to angle")
        else:
            print("  -> Success!")
        break

print(f"\nFinal position: {info['position']}")
print(f"Total reward: {total_reward:.2f}")
