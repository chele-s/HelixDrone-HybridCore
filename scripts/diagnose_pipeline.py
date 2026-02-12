import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import yaml
from python_src.envs.drone_env import QuadrotorEnv, EnvConfig, VectorizedQuadrotorEnv
from python_src.agents.ddpg_agent import TD3LSTMAgent
from python_src.utils.helix_math import RunningMeanStd

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
env_cfg.domain_randomization = True
env_cfg.wind_enabled = True
env_cfg.motor_dynamics = True
env_cfg.crash_height = cfg['termination']['crash_height']
env_cfg.crash_distance = cfg['termination']['crash_distance']
env_cfg.crash_angle = cfg['termination']['crash_angle']
env_cfg.position_scale = cfg['observation']['position_scale']
env_cfg.velocity_scale = cfg['observation']['velocity_scale']
env_cfg.angular_velocity_scale = cfg['observation']['angular_velocity_scale']
env_cfg.reward_crash = cfg['rewards']['crash']
env_cfg.proximity_scale = cfg['rewards']['proximity_scale']

num_envs = 4
venv = VectorizedQuadrotorEnv(num_envs, env_cfg)

state_dim = venv.observation_space.shape[0]
base_obs_dim = 20
action_dim = 4

obs_normalizer = RunningMeanStd(shape=(state_dim,))

device = torch.device('cpu')
agent = TD3LSTMAgent(
    obs_dim=base_obs_dim,
    action_dim=action_dim,
    device=device,
    hidden_dim=cfg['agent']['hidden_dim'],
    lstm_hidden=cfg['lstm']['hidden_dim'],
    lstm_layers=cfg['lstm']['num_layers'],
    sequence_length=cfg['lstm']['sequence_length'],
    use_amp=False,
    compile_networks=False
)

obs_raw = venv.reset()
if isinstance(obs_raw, tuple):
    obs_raw = obs_raw[0]

print("=" * 70)
print("TEST A: Untrained actor output (raw obs)")
print("=" * 70)
agent.reset_hidden_states()
obs_base = obs_raw[:, :base_obs_dim]
print(f"  obs_base[0] = {obs_base[0][:8]}...")
action = agent.get_actions_batch(obs_base, add_noise=False)
print(f"  action[0] = {action[0]}")
print(f"  action range: [{action.min():.4f}, {action.max():.4f}]")
print(f"  action mean: {action.mean():.4f}, std: {action.std():.4f}")

print("\n" + "=" * 70)
print("TEST B: Untrained actor output (normalized obs)")
print("=" * 70)
agent.reset_hidden_states()
agent._obs_buffers = None
obs_normalizer.update(obs_raw)
obs_norm = obs_normalizer.normalize(obs_raw, clip=5.0)
obs_base_norm = obs_norm[:, :base_obs_dim]
print(f"  obs_base_norm[0] = {obs_base_norm[0][:8]}...")
print(f"  normalizer mean[:8] = {obs_normalizer.mean[:8]}")
print(f"  normalizer std[:8]  = {np.sqrt(obs_normalizer.var[:8] + 1e-8)}")
action_norm = agent.get_actions_batch(obs_base_norm, add_noise=False)
print(f"  action[0] = {action_norm[0]}")
print(f"  action range: [{action_norm.min():.4f}, {action_norm.max():.4f}]")

print("\n" + "=" * 70)
print("TEST C: Run 50 steps with untrained actor (raw obs)")
print("=" * 70)
obs_raw = venv.reset()
if isinstance(obs_raw, tuple):
    obs_raw = obs_raw[0]
agent.reset_hidden_states()
agent._obs_buffers = None
for step in range(50):
    obs_base = obs_raw[:, :base_obs_dim]
    action = agent.get_actions_batch(obs_base, add_noise=False)
    noise_c = np.random.randn(num_envs, 1).astype(np.float32) * 0.15
    noise_d = np.random.randn(num_envs, 4).astype(np.float32) * 0.06
    action = np.clip(action + noise_c + noise_d, -1.0, 1.0)
    obs_raw, reward, terminated, truncated, info = venv.step(action)
    if step % 5 == 0 or terminated.any():
        print(f"  Step {step:3d} | action[0]=[{action[0,0]:+.3f},{action[0,1]:+.3f},{action[0,2]:+.3f},{action[0,3]:+.3f}] "
              f"| r=[{reward[0]:+.2f}] | term={terminated}")
    if terminated.all():
        print(f"  ALL CRASHED at step {step}")
        break

print("\n" + "=" * 70)
print("TEST D: Run 50 steps with untrained actor (normalized obs)")
print("=" * 70)
obs_raw = venv.reset()
if isinstance(obs_raw, tuple):
    obs_raw = obs_raw[0]
agent.reset_hidden_states()
agent._obs_buffers = None
obs_normalizer = RunningMeanStd(shape=(state_dim,))
for step in range(200):
    obs_normalizer.update(obs_raw)
    obs_norm = obs_normalizer.normalize(obs_raw, clip=5.0)
    obs_base_norm = obs_norm[:, :base_obs_dim]
    action = agent.get_actions_batch(obs_base_norm, add_noise=False)
    noise_c = np.random.randn(num_envs, 1).astype(np.float32) * 0.15
    noise_d = np.random.randn(num_envs, 4).astype(np.float32) * 0.06
    action = np.clip(action + noise_c + noise_d, -1.0, 1.0)
    obs_raw, reward, terminated, truncated, info = venv.step(action)
    if step % 10 == 0 or terminated.any():
        print(f"  Step {step:3d} | action[0]=[{action[0,0]:+.3f},{action[0,1]:+.3f},{action[0,2]:+.3f},{action[0,3]:+.3f}] "
              f"| r=[{reward[0]:+.2f}] | term={terminated}")
    if terminated.all():
        print(f"  ALL CRASHED at step {step}")
        break

print("\n" + "=" * 70)
print("TEST E: What does normalizer do to obs after warmup?")
print("=" * 70)
obs_raw = venv.reset()
if isinstance(obs_raw, tuple):
    obs_raw = obs_raw[0]
normalizer2 = RunningMeanStd(shape=(state_dim,))
for _ in range(100):
    a = np.random.randn(num_envs, 1).astype(np.float32) * 0.15 + np.random.randn(num_envs, 4).astype(np.float32) * 0.08
    obs_raw, _, _, _, _ = venv.step(np.clip(a, -1, 1))
    normalizer2.update(obs_raw)
obs_n = normalizer2.normalize(obs_raw, clip=5.0)
print(f"  Raw obs[0][:10] = {obs_raw[0][:10]}")
print(f"  Norm obs[0][:10] = {obs_n[0][:10]}")
print(f"  Mean[:10] = {normalizer2.mean[:10]}")
print(f"  Std[:10]  = {np.sqrt(normalizer2.var[:10] + 1e-8)}")
