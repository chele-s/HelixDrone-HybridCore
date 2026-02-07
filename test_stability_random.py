import numpy as np
import gymnasium as gym
import drone_env
import time

def test_stability():
    print("Initializing environment for Stability Test...")
    config = drone_env.EnvConfig()
    config.use_sota_actuator = True
    config.hover_rpm = 2600.0
    config.rpm_range = 3600.0
    config.mass = 0.6
    
    env = drone_env.QuadrotorEnv(config=config)
    
    num_episodes = 5
    max_steps = 200
    
    print(f"\n--- RUNNING STABILITY TEST ({num_episodes} Episodes) ---")
    print(f"Config: Hover={config.hover_rpm}, Range={config.rpm_range}, Mass={config.mass}")
    
    total_steps = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_steps = 0
        
        for i in range(max_steps):
            # Random action: Simulates untrained agent
            # Using conservative random noise to see if it survives 'normal' exploration
            action = np.random.uniform(-1.0, 1.0, 4)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            
            if terminated:
                break
        
        total_steps += episode_steps
        print(f"Episode {episode+1}: Survived {episode_steps} steps")
    
    avg_steps = total_steps / num_episodes
    print(f"\nAverage Survival Steps: {avg_steps:.1f}")
    
    if avg_steps > 30:
        print("PASS: System is stable enough for learning.")
    else:
        print("FAIL: System is still crashing too quickly.")

if __name__ == "__main__":
    test_stability()
