import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build', 'Release'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

from python_src.envs.drone_env import QuadrotorEnv, EnvConfig, TaskType
from python_src.agents import TD3Agent, TD3LSTMAgent

def compare_agents():
    print("=" * 80)
    print("LSTM vs MLP AGENT DIAGNOSTIC")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    config = EnvConfig()
    config.use_sota_actuator = True
    config.hover_rpm = 2600.0
    config.rpm_range = 3600.0
    config.domain_randomization = False
    config.wind_enabled = False
    config.curriculum_enabled = False
    
    env = QuadrotorEnv(config=config, task=TaskType.HOVER)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    base_obs_dim = 20
    
    print(f"\n[ENV INFO]")
    print(f"  Full observation dim: {state_dim}")
    print(f"  Base observation dim: {base_obs_dim}")
    print(f"  Action dim: {action_dim}")
    
    mlp_agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        hidden_dim=256
    )
    
    lstm_agent = TD3LSTMAgent(
        obs_dim=base_obs_dim,
        action_dim=action_dim,
        device=device,
        hidden_dim=256,
        lstm_hidden=128,
        lstm_layers=2,
        sequence_length=16
    )
    lstm_agent.reset_hidden_states()
    
    print("\n" + "=" * 80)
    print("TEST 1: Compare untrained agent outputs")
    print("=" * 80)
    
    obs, _ = env.reset()
    obs_base = env._get_base_obs()
    
    print(f"\n[INPUT OBSERVATION]")
    print(f"  Full obs shape: {obs.shape}")
    print(f"  Full obs range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  Base obs shape: {obs_base.shape}")
    print(f"  Base obs range: [{obs_base.min():.3f}, {obs_base.max():.3f}]")
    
    mlp_action = mlp_agent.get_action(obs, add_noise=False)
    lstm_action = lstm_agent.get_action(obs_base, add_noise=False)
    
    print(f"\n[UNTRAINED ACTIONS (no noise)]")
    print(f"  MLP action:  {mlp_action}")
    print(f"  LSTM action: {lstm_action}")
    print(f"  MLP action range:  [{mlp_action.min():.4f}, {mlp_action.max():.4f}]")
    print(f"  LSTM action range: [{lstm_action.min():.4f}, {lstm_action.max():.4f}]")
    
    print("\n" + "=" * 80)
    print("TEST 2: Run 20 steps and compare action patterns")
    print("=" * 80)
    
    env.reset()
    lstm_agent.reset_hidden_states()
    
    mlp_actions_log = []
    lstm_actions_log = []
    
    for step in range(20):
        obs = env._get_obs()
        obs_base = env._get_base_obs()
        
        mlp_action = mlp_agent.get_action(obs, add_noise=True)
        lstm_action = lstm_agent.get_action(obs_base, add_noise=True)
        
        mlp_actions_log.append(mlp_action.copy())
        lstm_actions_log.append(lstm_action.copy())
        
        obs, reward, terminated, truncated, info = env.step(lstm_action)
        
        if step < 5 or step >= 15:
            print(f"  Step {step:02d}: MLP={mlp_action.mean():.4f} LSTM={lstm_action.mean():.4f}")
    
    mlp_actions = np.array(mlp_actions_log)
    lstm_actions = np.array(lstm_actions_log)
    
    print(f"\n[ACTION STATISTICS over 20 steps]")
    print(f"  MLP  mean: {mlp_actions.mean():.4f}, std: {mlp_actions.std():.4f}")
    print(f"  LSTM mean: {lstm_actions.mean():.4f}, std: {lstm_actions.std():.4f}")
    print(f"  MLP  range: [{mlp_actions.min():.4f}, {mlp_actions.max():.4f}]")
    print(f"  LSTM range: [{lstm_actions.min():.4f}, {lstm_actions.max():.4f}]")
    
    print("\n" + "=" * 80)
    print("TEST 3: Run full episode with LSTM (no noise)")
    print("=" * 80)
    
    obs, _ = env.reset()
    lstm_agent.reset_hidden_states()
    
    print(f"\n{'Step':>4} | {'Action Mean':>11} | {'Action Std':>10} | {'Z':>7} | {'Tilt°':>7} | {'Reward':>8}")
    print("-" * 70)
    
    for step in range(50):
        obs_base = env._get_base_obs()
        lstm_action = lstm_agent.get_action(obs_base, add_noise=False)
        
        obs, reward, terminated, truncated, info = env.step(lstm_action)
        
        state = env._drone.get_state()
        euler = state.orientation.to_euler_zyx()
        tilt = np.degrees(np.sqrt(euler.x**2 + euler.y**2))
        
        if step % 5 == 0 or terminated:
            print(f"{step:>4} | {lstm_action.mean():>11.4f} | {lstm_action.std():>10.4f} | {state.position.z:>7.3f} | {tilt:>7.2f} | {reward:>8.2f}")
        
        if terminated:
            print(f"\n>>> CRASHED at step {step}")
            break
    
    print("\n" + "=" * 80)
    print("TEST 4: Check LSTM hidden state behavior")
    print("=" * 80)
    
    obs, _ = env.reset()
    lstm_agent.reset_hidden_states()
    
    obs_base = env._get_base_obs()
    
    action1 = lstm_agent.get_action(obs_base, add_noise=False)
    action2 = lstm_agent.get_action(obs_base, add_noise=False)
    action3 = lstm_agent.get_action(obs_base, add_noise=False)
    
    print(f"\n[Same obs 3 times - should differ due to LSTM state]")
    print(f"  Action 1: {action1}")
    print(f"  Action 2: {action2}")
    print(f"  Action 3: {action3}")
    
    are_same = np.allclose(action1, action2) and np.allclose(action2, action3)
    if are_same:
        print("  ⚠️  WARNING: All actions identical - LSTM state not updating?")
    else:
        print("  ✓ Actions differ as expected for LSTM")
    
    print("\n" + "=" * 80)
    print("TEST 5: Check observation buffer")
    print("=" * 80)
    
    lstm_agent.reset_hidden_states()
    print(f"\n[Observation buffer state]")
    if hasattr(lstm_agent, '_single_obs_buffer'):
        print(f"  Buffer shape: {lstm_agent._single_obs_buffer.shape}")
        print(f"  Buffer ptr: {lstm_agent._single_obs_ptr}")
        print(f"  Buffer full: {lstm_agent._single_obs_full}")
        print(f"  Buffer values (last entry): {lstm_agent._single_obs_buffer[0][:5]}...")
    else:
        print("  No _single_obs_buffer found")
    
    obs_base = env._get_base_obs()
    _ = lstm_agent.get_action(obs_base, add_noise=False)
    
    print(f"\n[After 1 step]")
    if hasattr(lstm_agent, '_single_obs_buffer'):
        print(f"  Buffer ptr: {lstm_agent._single_obs_ptr}")
        print(f"  Buffer values (entry 0): {lstm_agent._single_obs_buffer[0][:5]}...")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    compare_agents()
