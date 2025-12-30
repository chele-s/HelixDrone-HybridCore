import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'Release'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'build'))

import drone_core
import numpy as np

def find_hover_rpm():
    cfg = drone_core.QuadrotorConfig()
    cfg.mass = 0.6
    cfg.arm_length = 0.25
    cfg.integration_method = drone_core.IntegrationMethod.RK4
    cfg.motor_config = drone_core.MotorConfiguration.X
    cfg.enable_ground_effect = False
    cfg.enable_wind_disturbance = False
    cfg.enable_motor_dynamics = True
    cfg.enable_battery_dynamics = False
    cfg.enable_blade_flapping = True
    cfg.enable_advanced_aero = True
    
    cfg.rotor.radius = 0.127
    cfg.rotor.chord = 0.02
    cfg.rotor.pitch_angle = 0.26
    cfg.rotor.flapping.enabled = True
    cfg.rotor.flapping.lock_number = 8.0
    
    cfg.motor.kv = 2300
    cfg.motor.max_current = 30
    cfg.motor.max_rpm = 35000.0
    cfg.motor.esc.nonlinear_gamma = 1.2
    
    cfg.aero.air_density = 1.225
    
    print("=" * 60)
    print("Physics Debug - Finding Correct Hover RPM")
    print("=" * 60)
    print(f"Mass: {cfg.mass} kg")
    print(f"Target thrust for hover: {cfg.mass * 9.81:.4f} N")
    print("=" * 60)
    
    drone = drone_core.Quadrotor(cfg)
    
    results = []
    
    for rpm in range(500, 15000, 250):
        drone.reset()
        drone.set_position(drone_core.Vec3(0, 0, 5.0))
        drone.set_velocity(drone_core.Vec3(0, 0, 0))
        drone.set_angular_velocity(drone_core.Vec3(0, 0, 0))
        drone.set_orientation(drone_core.Quaternion(1, 0, 0, 0))
        
        for _ in range(20):
            cmd = drone_core.MotorCommand(float(rpm), float(rpm), float(rpm), float(rpm))
            drone.step(cmd, 0.002)
        
        cmd = drone_core.MotorCommand(float(rpm), float(rpm), float(rpm), float(rpm))
        drone.step(cmd, 0.01)
        
        state = drone.get_state()
        vel_z = state.velocity.z
        
        acc_z = vel_z / 0.01
        
        net_force = cfg.mass * acc_z
        thrust = cfg.mass * 9.81 + net_force
        
        results.append((rpm, vel_z, acc_z, thrust))
        
        if len(results) > 1:
            prev_vel = results[-2][1]
            if prev_vel <= 0 and vel_z > 0:
                print(f">>> HOVER POINT FOUND between RPM {results[-2][0]} and {rpm}")
    
    print("\n" + "=" * 60)
    print("RPM vs Vertical Velocity (negative = falling, positive = rising)")
    print("=" * 60)
    print(f"{'RPM':>8} | {'Vel_Z (m/s)':>12} | {'Acc_Z (m/s²)':>12} | {'Thrust (N)':>10}")
    print("-" * 60)
    
    hover_rpm = None
    for rpm, vel_z, acc_z, thrust in results:
        marker = ""
        if hover_rpm is None and vel_z > 0:
            hover_rpm = rpm
            marker = " <<< HOVER"
        print(f"{rpm:>8} | {vel_z:>12.4f} | {acc_z:>12.4f} | {thrust:>10.4f}{marker}")
    
    print("=" * 60)
    if hover_rpm:
        print(f"\n>>> RECOMMENDED hover_rpm: {hover_rpm - 125} to {hover_rpm}")
        print(f">>> Current config uses: 5500")
        print(f">>> Difference: hover_rpm should be ~{hover_rpm} instead of 5500")
    else:
        print("WARNING: Could not find hover point in tested range")
    
    return hover_rpm

if __name__ == '__main__':
    find_hover_rpm()
