import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'Release'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'build'))

import drone_core
import numpy as np

def create_config():
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
    cfg.sub_step.enable_sub_stepping = False
    cfg.rotor.radius = 0.127
    cfg.rotor.chord = 0.02
    cfg.rotor.pitch_angle = 0.18
    cfg.rotor.flapping.enabled = True
    cfg.rotor.flapping.lock_number = 8.0
    cfg.motor.kv = 2300
    cfg.motor.max_current = 30
    cfg.motor.max_rpm = 35000.0
    cfg.motor.esc.nonlinear_gamma = 1.2
    cfg.aero.air_density = 1.225
    return cfg

print("=" * 60)
print("VERIFICATION: pitch_angle = 0.18")
print("=" * 60)

cfg = create_config()
hover_rpm = 2750.0
target_z = 2.0
dt = 0.01

print(f"\n>>> TEST 1: Hover at {hover_rpm} RPM")
drone = drone_core.Quadrotor(cfg)
drone.set_position(drone_core.Vec3(0, 0, target_z))
drone.set_velocity(drone_core.Vec3(0, 0, 0))
drone.set_angular_velocity(drone_core.Vec3(0, 0, 0))
drone.set_orientation(drone_core.Quaternion(1, 0, 0, 0))

warmup_cmd = drone_core.MotorCommand(hover_rpm, hover_rpm, hover_rpm, hover_rpm)
for _ in range(10):
    drone.step(warmup_cmd, dt)
drone.set_velocity(drone_core.Vec3(0, 0, 0))

cmd = drone_core.MotorCommand(hover_rpm, hover_rpm, hover_rpm, hover_rpm)
for i in range(100):
    drone.step(cmd, dt)

s = drone.get_state()
print(f"   Hover test: Z={s.position.z:.4f}, Vz={s.velocity.z:+.4f}")
if abs(s.velocity.z) < 1.0 and abs(s.position.z - target_z) < 1.0:
    print("   >>> HOVERING OK!")
else:
    print("   >>> NOT HOVERING")

print(f"\n>>> TEST 2: 2000 RPM (should fall)")
drone2 = drone_core.Quadrotor(cfg)
drone2.set_position(drone_core.Vec3(0, 0, target_z))
drone2.set_velocity(drone_core.Vec3(0, 0, 0))
drone2.set_angular_velocity(drone_core.Vec3(0, 0, 0))
drone2.set_orientation(drone_core.Quaternion(1, 0, 0, 0))

warmup_cmd = drone_core.MotorCommand(hover_rpm, hover_rpm, hover_rpm, hover_rpm)
for _ in range(10):
    drone2.step(warmup_cmd, dt)
drone2.set_velocity(drone_core.Vec3(0, 0, 0))

cmd = drone_core.MotorCommand(2000, 2000, 2000, 2000)
for i in range(100):
    drone2.step(cmd, dt)

s = drone2.get_state()
print(f"   2000 RPM test: Z={s.position.z:.4f}, Vz={s.velocity.z:+.4f}")
if s.velocity.z < 0:
    print("   >>> FALLING (CORRECT!)")
else:
    print("   >>> RISING (bug)")

print(f"\n>>> TEST 3: 3500 RPM (should rise)")
drone3 = drone_core.Quadrotor(cfg)
drone3.set_position(drone_core.Vec3(0, 0, target_z))
drone3.set_velocity(drone_core.Vec3(0, 0, 0))
drone3.set_angular_velocity(drone_core.Vec3(0, 0, 0))
drone3.set_orientation(drone_core.Quaternion(1, 0, 0, 0))

warmup_cmd = drone_core.MotorCommand(hover_rpm, hover_rpm, hover_rpm, hover_rpm)
for _ in range(10):
    drone3.step(warmup_cmd, dt)
drone3.set_velocity(drone_core.Vec3(0, 0, 0))

cmd = drone_core.MotorCommand(3500, 3500, 3500, 3500)
for _ in range(100):
    drone3.step(cmd, dt)

s = drone3.get_state()
print(f"   3500 RPM test: Z={s.position.z:.4f}, Vz={s.velocity.z:+.4f}")
if s.velocity.z > 0:
    print("   >>> RISING (CORRECT!)")
else:
    print("   >>> FALLING (bug)")

print("\n" + "=" * 60)
