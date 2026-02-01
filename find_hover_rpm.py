import numpy as np
import gymnasium as gym
import drone_env
import time
import matplotlib.pyplot as plt

def find_hover_rpm():
    print("Initializing environment for RPM calibration...")
    # Use standard config but disable SOTA for direct RPM control during calibration
    config = drone_env.EnvConfig()
    config.use_sota_actuator = False 
    config.motor_dynamics = False # Instant response for calibration
    config.mass = 0.6
    
    # Range to test
    min_rpm = 2000
    max_rpm = 8000
    step = 50
    
    rpms = range(min_rpm, max_rpm + step, step)
    vertical_accels = []
    
    env = drone_env.QuadrotorEnv(config=config)
    
    print(f"Testing RPM range: {min_rpm} to {max_rpm}")
    
    best_rpm = -1
    min_accel = float('inf')
    
    for rpm in rpms:
        env.reset()
        
        # Override state to be at rest
        # We need to access the internal drone state if possible, or usually reset puts it at rest.
        # But we need to force the motor RPMs immediately.
        
        # Since we disabled motor dynamics, the command should be instant.
        # Action space is -1 to 1. We need to map RPM to action.
        # But wait, action mapping depends on hover_rpm which is what we are trying to find!
        # So we should cheat and set the motors directly if possible, OR
        # define a temporary hover_rpm in config so we can map 0.0 to checking value.
        
        # Actually, simpler:
        # Action -1 = min_rpm
        # Action +1 = max_rpm
        # Action 0 = hover_rpm
        # We can calculate the exact action needed for a specific target RPM.
        
        target_rpm = float(rpm)
        
        # action = ... (complex inverse mapping)
        # Instead, let's use the fact that we can set the action to produce exactly target_rpm
        # provided we know the scaling.
        # Normalized = (target_rpm - hover_rpm) / (max_rpm - min_rpm) * 2 ?? No.
        
        # Let's look at `_process_action` in drone_env.py or similar.
        # Usually: pwm = action * 0.5 + 0.5 ...
        # Let's just modify the env to take raw RPM? No.
        
        # Let's use the C++ binding directly if we can, or just infer from step result.
        # Actually, if we set SOTA=False, the action is usually mapped linearly.
        
        # Let's Try a Binary Search or Sweep using the Physics Engine directly if exposed?
        # Initializing the environment allocates a drone.
        
        # Let's use a trick:
        # Set min_rpm = target_rpm, max_rpm = target_rpm.
        # Then any action results in target_rpm.
        
        curr_config = drone_env.EnvConfig()
        curr_config.use_sota_actuator = False
        curr_config.motor_dynamics = False
        curr_config.mass = 0.6
        curr_config.min_rpm = target_rpm
        curr_config.max_rpm = target_rpm
        curr_config.hover_rpm = target_rpm
        
        temp_env = drone_env.QuadrotorEnv(config=curr_config)
        temp_env.reset()
        
        # Apply action 0 (which will be target_rpm because min=max)
        obs, reward, terminated, truncated, info = temp_env.step(np.zeros(4))
        
        # Get state
        state = temp_env.get_drone_state()
        
        # Check vertical acceleration
        # acceleration = (v_current - v_prev) / dt
        # But verify_fix.py showed we can get state.
        
        # Simulating one step (dt=0.01 or similar)
        # Verify Z velocity. Positive Z is UP? Or DOWN?
        # Usually in NED, Z is Down. So positive accel is falling.
        # We want Z accel to be 0. (Hover)
        # Gravity is 9.81.
        
        # Let's look at the velocity change.
        # Initial velocity is 0.
        # After 1 step, velocity is V_new.
        # Accel ~ V_new / dt.
        
        vel_z = state.velocity.z
        
        # In NED frame:
        # If Thrust < Weight, we fall (positive Z accel)
        # If Thrust > Weight, we climb (negative Z accel)
        # We want vel_z to be close to 0 (actually it starts at 0, so after 1 step relative to gravity...)
        # Wait, if we start at 0, and gravity acts, we fall.
        # Ideally, we want the NET force to be 0.
        
        vertical_accels.append(vel_z)
        
        # Ideally we want the RPM that yields the SMALLEST abs(vel_z) after 1 step starting from rest.
        if abs(vel_z) < min_accel:
            min_accel = abs(vel_z)
            best_rpm = rpm
            
        # print(f"RPM: {rpm}, VelZ: {vel_z:.6f}")
        temp_env.close()

    print(f"\n--- CALIBRATION COMPLETE ---")
    print(f"Optimal Hover RPM: {best_rpm}")
    print(f"Vertical Drift at Optimal: {min_accel:.6f} m/s per step")
    print("\nPlease update 'hover_rpm' in train_params.yaml and defaults with this value.")

if __name__ == "__main__":
    find_hover_rpm()
