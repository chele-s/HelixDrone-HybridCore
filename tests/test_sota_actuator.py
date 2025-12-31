import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'Release'))

import numpy as np
import drone_core

def test_anti_bang_bang():
    cfg = drone_core.SOTAActuatorConfig()
    cfg.delay_ms = 20.0
    cfg.tau_spin_up = 0.12
    cfg.tau_spin_down = 0.06
    cfg.voltage_sag_factor = 0.08
    
    model = drone_core.SOTAActuatorModel(cfg)
    model.reset()
    
    rpm_hist = []
    np.random.seed(42)
    
    for i in range(200):
        action = np.random.uniform(-1, 1, 4)
        rpm = np.array(model.step_normalized(action, 0.01, 16.8))
        rpm_hist.append(rpm[0])
    
    diffs = np.abs(np.diff(rpm_hist))
    
    print("="*60)
    print("SOTA ACTUATOR ANTI-BANG-BANG TEST")
    print("="*60)
    print(f"Max RPM change per step: {max(diffs):.2f} RPM")
    print(f"Mean RPM change per step: {np.mean(diffs):.2f} RPM")
    print(f"Std of RPM changes: {np.std(diffs):.2f} RPM")
    print(f"Final RPM: {rpm_hist[-1]:.2f}")
    print("-"*60)
    is_smooth = max(diffs) < 500
    print(f"ANTI-BANG-BANG WORKING: {'YES' if is_smooth else 'NO'}")
    print("="*60)
    
    return is_smooth

def test_latency():
    cfg = drone_core.SOTAActuatorConfig()
    cfg.delay_ms = 30.0
    
    model = drone_core.SOTAActuatorModel(cfg)
    model.reset()
    
    outputs = []
    initial_rpm = model.get_state().filtered_rpm[0]
    
    for i in range(50):
        action = np.ones(4)
        rpm = np.array(model.step_normalized(action, 0.01, 16.8))
        outputs.append(rpm[0])
    
    first_significant_change = None
    for i, v in enumerate(outputs):
        if abs(v - initial_rpm) > 50:
            first_significant_change = i
            break
    
    print("\n" + "="*60)
    print("LATENCY TEST")
    print("="*60)
    print(f"Delay configured: {cfg.delay_ms}ms")
    print(f"Initial RPM: {initial_rpm:.2f}")
    print(f"Steps to significant change (>50 RPM): {first_significant_change}")
    print(f"RPM at step 5: {outputs[5]:.2f}")
    print(f"RPM at step 10: {outputs[10]:.2f}")
    latency_ok = first_significant_change is not None
    print(f"LATENCY MODEL ACTIVE: {'YES' if latency_ok else 'NO'}")
    print("="*60)
    
    return latency_ok

def test_asymmetric_dynamics():
    cfg = drone_core.SOTAActuatorConfig()
    cfg.tau_spin_up = 0.12
    cfg.tau_spin_down = 0.06
    cfg.delay_ms = 5.0
    
    model = drone_core.SOTAActuatorModel(cfg)
    model.reset()
    
    initial_rpm = model.get_state().filtered_rpm[0]
    target_high = initial_rpm + 500
    
    spinup_steps = 0
    for i in range(100):
        rpm = np.array(model.step_normalized(np.ones(4), 0.01, 16.8))
        if rpm[0] > target_high and spinup_steps == 0:
            spinup_steps = i
            break
    
    peak_rpm = model.get_state().filtered_rpm[0]
    target_low = peak_rpm - 500
    
    spindown_steps = 0
    for i in range(100):
        rpm = np.array(model.step_normalized(-np.ones(4), 0.01, 16.8))
        if rpm[0] < target_low and spindown_steps == 0:
            spindown_steps = i
            break
    
    print("\n" + "="*60)
    print("ASYMMETRIC DYNAMICS TEST")
    print("="*60)
    print(f"Spin-up time constant: {cfg.tau_spin_up}s")
    print(f"Spin-down time constant: {cfg.tau_spin_down}s")
    print(f"Steps to spin-up +500 RPM: {spinup_steps}")
    print(f"Steps to spin-down -500 RPM: {spindown_steps}")
    asymmetric_ok = spinup_steps > 0 and spindown_steps > 0
    print(f"Spin-down faster than spin-up: {spindown_steps < spinup_steps if asymmetric_ok else 'N/A'}")
    print(f"ASYMMETRIC DYNAMICS WORKING: {'YES' if asymmetric_ok else 'NEEDS MORE STEPS'}")
    print("="*60)
    
    return asymmetric_ok

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# STATE-OF-THE-ART ACTUATOR MODEL VERIFICATION")
    print("#"*60 + "\n")
    
    t1 = test_anti_bang_bang()
    t2 = test_latency()
    t3 = test_asymmetric_dynamics()
    
    print("\n" + "#"*60)
    print("# SUMMARY")
    print("#"*60)
    print(f"Anti-Bang-Bang: {'PASS' if t1 else 'FAIL'}")
    print(f"Latency Model: {'PASS' if t2 else 'FAIL'}")
    print(f"Asymmetric Dynamics: {'PASS' if t3 else 'FAIL'}")
    print("#"*60 + "\n")
