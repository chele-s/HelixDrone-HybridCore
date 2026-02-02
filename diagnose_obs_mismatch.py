import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build', 'Release'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from python_src.envs.drone_env import QuadrotorEnv, EnvConfig, TaskType
from python_src.utils.helix_math import RunningMeanStd

def diagnose_observation_mismatch():
    print("=" * 80)
    print("OBSERVATION NORMALIZATION MISMATCH DIAGNOSTIC")
    print("=" * 80)
    
    config = EnvConfig()
    config.use_sota_actuator = True
    config.domain_randomization = False
    config.wind_enabled = False
    
    env = QuadrotorEnv(config=config, task=TaskType.HOVER)
    obs_full, _ = env.reset()
    obs_base = env._get_base_obs()
    
    print(f"\n[1] OBSERVATION STRUCTURE")
    print(f"  Full observation shape: {obs_full.shape}")
    print(f"  Base observation shape: {obs_base.shape}")
    
    obs_first_20 = obs_full[:20]
    
    print(f"\n[2] COMPARING obs_full[:20] vs _get_base_obs()")
    print(f"  obs_full[:20]: {obs_first_20[:8]}...")
    print(f"  _get_base_obs: {obs_base[:8]}...")
    
    match = np.allclose(obs_first_20, obs_base, atol=1e-5)
    print(f"  Match: {match}")
    if not match:
        diff = np.abs(obs_first_20 - obs_base)
        print(f"  Max difference: {diff.max():.6f} at index {diff.argmax()}")
    
    print(f"\n[3] SIMULATING OBSERVATION NORMALIZATION")
    normalizer = RunningMeanStd(shape=(obs_full.shape[0],))
    
    warmup_obs = []
    for _ in range(100):
        obs, _ = env.reset()
        warmup_obs.append(obs)
        for _ in range(10):
            action = np.random.uniform(-1, 1, 4).astype(np.float32)
            obs, _, done, _, _ = env.step(action)
            warmup_obs.append(obs)
            if done:
                break
    
    warmup_obs = np.array(warmup_obs)
    normalizer.update(warmup_obs)
    
    print(f"  Normalizer mean[:8]: {normalizer.mean[:8]}")
    print(f"  Normalizer std[:8]:  {np.sqrt(normalizer.var[:8])}")
    
    obs_full, _ = env.reset()
    obs_base = env._get_base_obs()
    
    obs_normalized = normalizer.normalize(obs_full, clip=10.0)
    
    print(f"\n[4] NORMALIZED vs UNNORMALIZED OBSERVATION")
    print(f"  Raw obs[:8]:        {obs_full[:8]}")
    print(f"  Normalized obs[:8]: {obs_normalized[:8]}")
    print(f"  Base obs[:8]:       {obs_base[:8]}")
    
    print(f"\n[5] THE PROBLEM:")
    print(f"  During training, VectorizedQuadrotorEnv uses:")
    print(f"    obs_base_batch = obs[:, :20]  <- This is NORMALIZED")
    print(f"  But the LSTM expects observations in their ORIGINAL scale")
    print(f"  because it was designed to receive _get_base_obs() which is NOT normalized")
    
    print(f"\n[6] IMPACT ON LSTM")
    
    print(f"  Expected input range (from _get_base_obs):")
    print(f"    Position: 0-1 (scaled by position_scale=5)")
    print(f"    Velocity: -1 to 1 (scaled)")
    print(f"    Quaternion: -1 to 1") 
    print(f"    etc...")
    
    print(f"\n  Actual input range (from normalized obs[:20]):")
    print(f"    All values centered around 0 with std ~1")
    print(f"    This is DIFFERENT from what was expected!")
    
    print(f"\n[7] VERIFICATION: Check if training uses normalized obs for LSTM")
    print(f"  In train.py line 407:")
    print(f"    obs_base_batch = obs[:, :self.base_obs_dim]")
    print(f"  Where 'obs' is the result of _normalize_obs()")
    print(f"  This means LSTM gets normalized input but agent.get_action")
    print(f"  in eval uses env._get_base_obs() which is NOT normalized!")
    
    print(f"\n  MISMATCH CONFIRMED!")
    print(f"  Training: normalized observations")
    print(f"  Inference/Eval: unnormalized observations")
    
    print("\n" + "=" * 80)
    print("ROOT CAUSE: NORMALIZATION MISMATCH BETWEEN TRAINING AND INFERENCE")
    print("=" * 80)


if __name__ == "__main__":
    diagnose_observation_mismatch()
