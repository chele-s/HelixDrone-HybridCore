import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import drone_core
import numpy as np


def make_drone():
    cfg = drone_core.QuadrotorConfig()
    cfg.integration_method = drone_core.IntegrationMethod.RK4
    cfg.motor_config = drone_core.MotorConfiguration.X
    cfg.enable_ground_effect = True
    cfg.enable_wind_disturbance = False
    cfg.enable_motor_dynamics = True
    cfg.enable_battery_dynamics = False
    cfg.enable_blade_flapping = True
    cfg.enable_advanced_aero = True
    cfg.sub_step.physics_sub_steps = 4
    cfg.sub_step.enable_sub_stepping = True
    cfg.sub_step.min_sub_step_dt = 0.0001
    cfg.sub_step.max_sub_step_dt = 0.005
    cfg.mass = 0.6
    cfg.arm_length = 0.25
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
    cfg.aero.ground_effect_coeff = 0.5
    return drone_core.Quadrotor(cfg)


def test_hover(rpm, duration=2.0, dt=0.01, start_z=2.0):
    drone = make_drone()
    drone.reset()
    drone.set_position(drone_core.Vec3(0, 0, start_z))
    drone.set_velocity(drone_core.Vec3(0, 0, 0))

    cmd = drone_core.MotorCommand(rpm, rpm, rpm, rpm)
    for _ in range(10):
        drone.step_with_sub_stepping(cmd, dt)
    drone.set_velocity(drone_core.Vec3(0, 0, 0))
    drone.set_angular_velocity(drone_core.Vec3(0, 0, 0))

    steps = int(duration / dt)
    positions = []
    velocities = []

    for i in range(steps):
        drone.step_with_sub_stepping(cmd, dt)
        s = drone.get_state()
        positions.append(s.position.z)
        velocities.append(s.velocity.z)

        if s.position.z < 0.03:
            return {
                "rpm": rpm,
                "crashed": True,
                "crash_step": i,
                "crash_time": i * dt,
                "final_z": s.position.z,
                "final_vz": s.velocity.z,
                "max_drift": start_z - min(positions),
            }

    final_z = positions[-1]
    drift = final_z - start_z
    avg_vz = np.mean(velocities[-50:])

    return {
        "rpm": rpm,
        "crashed": False,
        "final_z": final_z,
        "drift": drift,
        "avg_vz_last_50": avg_vz,
        "max_z": max(positions),
        "min_z": min(positions),
    }


def find_hover_rpm():
    print("=" * 70)
    print("HOVER RPM CALIBRATION")
    print("=" * 70)
    print(f"{'RPM':>8}  {'Status':>10}  {'Final Z':>10}  {'Drift':>10}  {'Avg Vz':>10}")
    print("-" * 70)

    best_rpm = None
    best_drift = float("inf")

    for rpm in range(1000, 10001, 200):
        result = test_hover(rpm)
        if result["crashed"]:
            print(f"{rpm:>8}  {'CRASHED':>10}  {result['final_z']:>10.4f}  {'---':>10}  {'---':>10}  (t={result['crash_time']:.2f}s)")
        else:
            drift = abs(result["drift"])
            status = "RISING" if result["drift"] > 0.1 else "FALLING" if result["drift"] < -0.1 else "HOVER"
            print(f"{rpm:>8}  {status:>10}  {result['final_z']:>10.4f}  {result['drift']:>+10.4f}  {result['avg_vz_last_50']:>+10.6f}")

            if drift < best_drift:
                best_drift = drift
                best_rpm = rpm

    print("-" * 70)
    print(f"\nBest hover RPM (coarse): {best_rpm}")

    print(f"\nFine-tuning around {best_rpm}...")
    print("-" * 70)

    for rpm in range(best_rpm - 200, best_rpm + 201, 25):
        result = test_hover(rpm, duration=5.0)
        if not result["crashed"]:
            drift = abs(result["drift"])
            status = "RISING" if result["drift"] > 0.05 else "FALLING" if result["drift"] < -0.05 else "HOVER"
            print(f"{rpm:>8}  {status:>10}  {result['final_z']:>10.4f}  {result['drift']:>+10.4f}  {result['avg_vz_last_50']:>+10.6f}")

            if drift < best_drift:
                best_drift = drift
                best_rpm = rpm
        else:
            print(f"{rpm:>8}  {'CRASHED':>10}")

    print("=" * 70)
    print(f"OPTIMAL HOVER RPM: {best_rpm}")
    print(f"Drift over 5s: {best_drift:.6f}m")
    print("=" * 70)

    print(f"\nDiagnostic: 19-step crash test at RPM={best_rpm} from z=2.0")
    drone = make_drone()
    drone.reset()
    drone.set_position(drone_core.Vec3(0, 0, 2.0))
    cmd = drone_core.MotorCommand(best_rpm, best_rpm, best_rpm, best_rpm)
    for _ in range(10):
        drone.step_with_sub_stepping(cmd, 0.01)
    drone.set_velocity(drone_core.Vec3(0, 0, 0))
    drone.set_angular_velocity(drone_core.Vec3(0, 0, 0))

    print(f"{'Step':>6}  {'Z':>10}  {'Vz':>10}  {'Roll':>10}  {'Pitch':>10}")
    for i in range(25):
        drone.step_with_sub_stepping(cmd, 0.01)
        s = drone.get_state()
        e = s.orientation.to_euler_zyx()
        print(f"{i+1:>6}  {s.position.z:>10.6f}  {s.velocity.z:>+10.6f}  {e.x:>+10.6f}  {e.y:>+10.6f}")
        if s.position.z < 0.03:
            print("  >>> CRASHED")
            break

    return best_rpm


if __name__ == "__main__":
    find_hover_rpm()
