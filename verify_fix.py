import numpy as np
import gymnasium as gym
import drone_env
import time

def verify_motor_dynamics():
    print("Initializing environment with SOTA Actuator...")
    config = drone_env.EnvConfig()
    config.use_sota_actuator = True
    config.motor_dynamics = True # This should be overridden to False internally
    
    env = drone_env.QuadrotorEnv(config=config)
    env.reset()
    
    # Step 1: Send a step input
    print("\nSending Step Input (Full Throttle)...")
    action = np.ones(4) # Max RPM
    
    # We will track how fast the drone state updates
    
    obs, _, _, _, info = env.step(action)
    
    drone_state = env.get_drone_state()
    # The SOTA actuator has a delay. Let's see if the C++ side *adds* more delay.
    # Ideally, after delay_ms, the motors should spin up.
    
    print(f"Step 1 RPM: {drone_state.motorRPM}")
    
    # We are looking for immediate reaction after the SOTA delay, without the slow "0.02" creep
    # If double filtering is active, the RPM will be very low for many steps.
    
    for i in range(10):
        obs, _, _, _, info = env.step(action)
        drone_state = env.get_drone_state()
        print(f"Step {i+2} RPM: {np.mean(drone_state.motorRPM):.2f}")

if __name__ == "__main__":
    verify_motor_dynamics()
