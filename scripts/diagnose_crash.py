import sys
sys.path.insert(0, '.')

import numpy as np
import yaml
from python_src.envs.drone_env import QuadrotorEnv, EnvConfig

with open('config/train_params.yaml') as f:
    cfg = yaml.safe_load(f)

env_cfg = EnvConfig()
env_cfg.dt = cfg['environment']['dt']
env_cfg.max_steps = 200
env_cfg.hover_rpm = cfg['environment']['hover_rpm']
env_cfg.min_rpm = cfg['environment']['min_rpm']
env_cfg.max_rpm = cfg['environment']['max_rpm']
env_cfg.rpm_range = cfg['environment']['rpm_range']
env_cfg.mass = cfg['environment']['mass']
env_cfg.use_sota_actuator = cfg['environment'].get('use_sota_actuator', True)
env_cfg.use_sub_stepping = cfg['environment'].get('use_sub_stepping', True)
env_cfg.physics_sub_steps = cfg['environment'].get('physics_sub_steps', 8)
env_cfg.domain_randomization = False
env_cfg.wind_enabled = False
env_cfg.motor_dynamics = False
env_cfg.crash_height = cfg['termination']['crash_height']
env_cfg.crash_distance = cfg['termination']['crash_distance']
env_cfg.crash_angle = cfg['termination']['crash_angle']
env_cfg.position_scale = cfg['observation']['position_scale']
env_cfg.velocity_scale = cfg['observation']['velocity_scale']
env_cfg.angular_velocity_scale = cfg['observation']['angular_velocity_scale']
env_cfg.reward_crash = cfg['rewards']['crash']
env_cfg.proximity_scale = cfg['rewards']['proximity_scale']

env = QuadrotorEnv(env_cfg)

print("=" * 70)
print("TEST 1: Zero action (hover)")
print("=" * 70)
obs, _ = env.reset()
for step in range(50):
    action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    pos = info.get('position', obs[:3] * env_cfg.position_scale)
    vel = info.get('velocity', obs[3:6] * env_cfg.velocity_scale) 
    if step % 5 == 0 or terminated:
        print(f"  Step {step:3d} | pos=[{pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}] "
              f"| reward={reward:+.2f} | term={terminated}")
    if terminated:
        print(f"  CRASHED at step {step}")
        break

print("\n" + "=" * 70)
print("TEST 2: Small random collective actions (like random exploration)")
print("=" * 70)
obs, _ = env.reset()
for step in range(50):
    collective = np.float32(np.random.randn() * 0.15)
    differential = np.random.randn(4).astype(np.float32) * 0.08
    action = np.clip(collective + differential, -1.0, 1.0)
    obs, reward, terminated, truncated, info = env.step(action)
    pos = info.get('position', obs[:3] * env_cfg.position_scale)
    if step % 5 == 0 or terminated:
        print(f"  Step {step:3d} | action=[{action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f},{action[3]:+.3f}] "
              f"| pos_z={pos[2]:+.4f} | reward={reward:+.2f} | term={terminated}")
    if terminated:
        print(f"  CRASHED at step {step}")
        break

print("\n" + "=" * 70)
print("TEST 3: Constant small differential (simulating biased actor)")
print("=" * 70)
obs, _ = env.reset()
for step in range(50):
    action = np.array([0.1, -0.1, 0.1, -0.1], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    pos = info.get('position', obs[:3] * env_cfg.position_scale)
    quat = info.get('quaternion', obs[6:10])
    if step % 2 == 0 or terminated:
        print(f"  Step {step:3d} | pos_z={pos[2]:+.4f} | quat=[{quat[0]:+.3f},{quat[1]:+.3f},{quat[2]:+.3f},{quat[3]:+.3f}] "
              f"| reward={reward:+.2f} | term={terminated}")
    if terminated:
        print(f"  CRASHED at step {step}")
        break

print("\n" + "=" * 70)
print("TEST 4: Observation values at reset")
print("=" * 70)
obs, _ = env.reset()
base_obs = obs[:20]
print(f"  Full obs dim: {obs.shape}")
print(f"  Base obs (first 20):")
for i in range(0, 20, 4):
    end = min(i+4, 20)
    vals = ', '.join(f'{base_obs[j]:+.4f}' for j in range(i, end))
    print(f"    [{i:2d}-{end-1:2d}]: {vals}")

print("\n" + "=" * 70)
print("TEST 5: What RPMs does action=[0,0,0,0] produce?")
print("=" * 70)
obs, _ = env.reset()
action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
hover = env_cfg.hover_rpm
rpm_range = env_cfg.rpm_range
rpm = hover + action * rpm_range
print(f"  hover_rpm={hover}, rpm_range={rpm_range}")
print(f"  action=[0,0,0,0] → RPMs = [{rpm[0]:.0f}, {rpm[1]:.0f}, {rpm[2]:.0f}, {rpm[3]:.0f}]")
action2 = np.array([0.1, -0.1, 0.1, -0.1], dtype=np.float32)
rpm2 = hover + action2 * rpm_range
print(f"  action=[0.1,-0.1,0.1,-0.1] → RPMs = [{rpm2[0]:.0f}, {rpm2[1]:.0f}, {rpm2[2]:.0f}, {rpm2[3]:.0f}]")

print("\n" + "=" * 70)
print("TEST 6: Step-by-step with zero action - detailed state")
print("=" * 70)
obs, _ = env.reset()
for step in range(30):
    action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    state = env._drone.get_state()
    pos = [state.position.x, state.position.y, state.position.z]
    vel = [state.velocity.x, state.velocity.y, state.velocity.z]
    omega = [state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z]
    quat = [state.orientation.w, state.orientation.x, state.orientation.y, state.orientation.z]
    
    print(f"  Step {step:2d} | z={pos[2]:+.5f} vel_z={vel[2]:+.5f} "
          f"ω=[{omega[0]:+.3f},{omega[1]:+.3f},{omega[2]:+.3f}] "
          f"q=[{quat[0]:+.3f},{quat[1]:+.3f},{quat[2]:+.3f},{quat[3]:+.3f}] "
          f"r={reward:+.3f} t={terminated}")
    if terminated:
        print(f"  >>> CRASHED at step {step}")
        break
