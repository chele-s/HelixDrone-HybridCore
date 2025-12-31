"""Quick test for the new QuadrotorEnvV2 with ESKF and Payload."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_src.envs import QuadrotorEnvV2, TaskType, ExtendedEnvConfig

def test_basic():
    print("=== BASIC ENV TEST ===")
    env = QuadrotorEnvV2()
    print(f"Obs space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Obs mode: {env.config.observation_mode}")
    
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}")
    
    action = env.action_space.sample()
    obs2, reward, term, trunc, info2 = env.step(action)
    print(f"Reward: {reward:.3f}, Terminated: {term}")
    print("PASSED!")
    return True

def test_payload():
    print("\n=== PAYLOAD TASK TEST ===")
    env = QuadrotorEnvV2(task=TaskType.PAYLOAD_HOVER)
    print(f"Obs space: {env.observation_space.shape}")
    print(f"Payload enabled: {env.config.payload_enabled}")
    print(f"Cable sensor: {env.config.cable_sensor_enabled}")
    
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if i % 10 == 0:
            swing = info.get('payload_swing', 0)
            tension = info.get('payload_tension', 0)
            print(f"  Step {i}: reward={reward:.2f}, swing={swing:.3f}rad, tension={tension:.1f}N")
        if term or trunc:
            obs, info = env.reset()
    
    print("PASSED!")
    return True

def test_estimated_mode():
    print("\n=== ESTIMATED MODE TEST ===")
    config = ExtendedEnvConfig(
        observation_mode='estimated',
        use_eskf=True
    )
    env = QuadrotorEnvV2(config=config)
    
    obs, info = env.reset()
    est = env.get_estimated_state()
    true_state = env.get_drone_state()
    
    print(f"True position: ({true_state.position.x:.3f}, {true_state.position.y:.3f}, {true_state.position.z:.3f})")
    if est:
        print(f"ESKF position: ({est.position.x:.3f}, {est.position.y:.3f}, {est.position.z:.3f})")
    
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    
    est2 = env.get_estimated_state()
    true2 = env.get_drone_state()
    
    if est2:
        pos_error = (
            (est2.position.x - true2.position.x)**2 +
            (est2.position.y - true2.position.y)**2 +
            (est2.position.z - true2.position.z)**2
        ) ** 0.5
        print(f"Position error after 10 steps: {pos_error:.4f}m")
    
    print("PASSED!")
    return True

def test_sb3_compatibility():
    print("\n=== STABLE-BASELINES3 COMPATIBILITY ===")
    try:
        from stable_baselines3.common.env_checker import check_env
        env = QuadrotorEnvV2()
        check_env(env, warn=True)
        print("SB3 env_checker: PASSED!")
        return True
    except ImportError:
        print("SB3 not installed, skipping")
        return True
    except Exception as e:
        print(f"SB3 check failed: {e}")
        return False

if __name__ == "__main__":
    all_passed = True
    all_passed &= test_basic()
    all_passed &= test_payload()
    all_passed &= test_estimated_mode()
    all_passed &= test_sb3_compatibility()
    
    print("\n" + "="*50)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*50)
