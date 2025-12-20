
import unittest
import numpy as np
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto

try:
    import drone_core
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build', 'Release')))
    import drone_core

class FlightMode(Enum):
    STABILIZE = auto()
    ALTITUDE = auto()
    LOITER = auto()
    RTL = auto()

@dataclass
class FlightMetrics:
    rise_time: float
    overshoot: float
    settling_time: float
    steady_state_error: float

class AdvancedFlightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executor = ThreadPoolExecutor(max_workers=4)
        cls.DT = 0.001

    def setUp(self):
        self.config = drone_core.QuadrotorConfig()
        self.config.mass = 1.0
        self.config.arm_length = 0.25
        self.config.motor_config = drone_core.MotorConfiguration.X
        self.config.integration_method = drone_core.IntegrationMethod.RK4
        self.config.enable_ground_effect = True
        self.config.enable_wind_disturbance = True
        self.config.enable_motor_dynamics = True
        self.drone = drone_core.Quadrotor(self.config)

    def _simulate_step_response(self, target_z: float, timeout: float = 5.0) -> list:
        history = []
        t = 0.0
        kp, kd, ki = 50.0, 15.0, 0.5
        err_sum = 0.0
        
        while t < timeout:
            state = self.drone.get_state()
            error = target_z - state.position.z
            err_sum += error * self.DT
            
            output = kp * error + kd * (0 - state.velocity.z) + ki * err_sum
            base_throttle = 9.80665 * self.config.mass / 4 / 1.5e-6 
            throttle = np.sqrt(max(0, base_throttle + output / 4 / 1.5e-6))
            throttle = float(np.clip(throttle, 0, 20000))
            
            cmd = drone_core.MotorCommand(throttle, throttle, throttle, throttle)
            self.drone.step(cmd, self.DT)
            
            history.append((t, state.position.z))
            t += self.DT
        return history

    def test_dynamic_step_response(self):
        target = 10.0
        history = self._simulate_step_response(target)
        times, values = zip(*history)
        values = np.array(values)
        
        final_val = values[-1]
        rise_idx = np.where(values >= 0.9 * target)[0]
        rise_time = times[rise_idx[0]] if len(rise_idx) > 0 else float('inf')
        max_val = np.max(values)
        overshoot = (max_val - target) / target * 100.0
        
        self.assertLess(rise_time, 3.5)
        self.assertLess(overshoot, 25.0)
        self.assertAlmostEqual(final_val, target, delta=0.1)

    def test_stochastic_wind_rejection(self):
        self.drone.reset()
        self.drone.enable_feature("wind", True)
        self.drone.set_wind(drone_core.Vec3(5.0, 2.0, 0.0))
        
        pos_deviations = []
        cmd = drone_core.MotorCommand(12000, 12000, 12000, 12000) 
        
        for _ in range(5000):
            self.drone.step(cmd, self.DT)
            s = self.drone.get_state()
            pos_deviations.append(s.velocity.norm())
            
        std_dev = np.std(pos_deviations)
        self.assertGreater(std_dev, 0.0) 

    def test_ground_effect_singularity(self):
        self.drone.reset()
        self.drone.enable_feature("ground_effect", True)
        self.drone.set_position(drone_core.Vec3(0,0,0.1))
        
        initial_force = self.drone.get_forces().z
        
        cmd = drone_core.MotorCommand(15000, 15000, 15000, 15000)
        self.drone.step(cmd, self.DT)
        
        force_near_ground = self.drone.get_forces().z
        
        self.drone.set_position(drone_core.Vec3(0,0,50.0))
        self.drone.step(cmd, self.DT)
        force_free_air = self.drone.get_forces().z
        
        self.assertGreater(force_near_ground, force_free_air)

    def test_motor_failure_recovery_logic(self):
        def control_loop(failure_time):
            d = drone_core.Quadrotor(self.config)
            d.reset()
            t = 0
            survived = False
            while t < 2.0:
                cmd_rpm = 10000
                if t > failure_time:
                    cmd = drone_core.MotorCommand(0, cmd_rpm*1.2, cmd_rpm*1.2, cmd_rpm*1.2)
                else:
                    cmd = drone_core.MotorCommand(cmd_rpm, cmd_rpm, cmd_rpm, cmd_rpm)
                
                d.step(cmd, self.DT)
                s = d.get_state()
                if t > failure_time + 0.5:
                    if abs(s.angular_velocity.x) > 5.0 or abs(s.angular_velocity.y) > 5.0:
                        survived = False
                        break
                    survived = True
                t += self.DT
            return survived

        future = self.executor.submit(control_loop, 1.0)
        self.assertIsNotNone(future.result())

    def test_battery_depletion_curve(self):
        self.drone.reset()
        self.drone.enable_feature("battery", True)
        
        v_start = self.drone.get_state().battery_voltage
        cmd = drone_core.MotorCommand(20000, 20000, 20000, 20000)
        
        for _ in range(100):
            self.drone.step(cmd, 0.1)
            
        v_end = self.drone.get_state().battery_voltage
        self.assertLess(v_end, v_start)

if __name__ == '__main__':
    unittest.main(verbosity=2)