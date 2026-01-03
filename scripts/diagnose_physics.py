"""
Physics Diagnostic Script for HelixDrone
Uses EXACT same formulas as C++ BladeElementTheory
"""
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'Release'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'build'))

import numpy as np
from python_src.envs.drone_env import ExtendedEnvConfig

# Constants from PhysicsEngine.h RotorConfig defaults
ROTOR_RADIUS = 0.127      # meters
ROTOR_CHORD = 0.02        # meters  
ROTOR_PITCH = 0.18        # radians
LIFT_SLOPE = 5.7          # lift curve slope
INFLOW_RATIO = 0.05       # lambda
AIR_DENSITY = 1.225       # kg/m^3

def compute_thrust_bet(rpm: float) -> float:
    """
    Exact copy of BladeElementTheory::computeThrust from PhysicsEngine.cpp
    Returns thrust in Newtons for ONE rotor
    """
    if rpm <= 0:
        return 0.0
    
    omega = rpm * np.pi / 30.0  # RPM to rad/s
    R = ROTOR_RADIUS
    c = ROTOR_CHORD
    theta = ROTOR_PITCH
    a = LIFT_SLOPE
    lam = INFLOW_RATIO
    
    solidity = 4.0 * c / (np.pi * R)
    Ct = solidity * a * (theta / 3.0 - lam / 2.0) / 2.0
    
    thrust = Ct * AIR_DENSITY * np.pi * R * R * (omega * R) * (omega * R)
    return thrust

def find_hover_rpm_analytical(mass_kg: float) -> float:
    """Find hover RPM using analytical solution"""
    weight = mass_kg * 9.81
    thrust_per_rotor = weight / 4.0
    
    # From compute_thrust_bet, we have:
    # thrust = Ct * rho * pi * R^2 * (omega * R)^2
    # thrust = Ct * rho * pi * R^4 * omega^2
    # omega = sqrt(thrust / (Ct * rho * pi * R^4))
    
    R = ROTOR_RADIUS
    c = ROTOR_CHORD
    theta = ROTOR_PITCH
    a = LIFT_SLOPE
    lam = INFLOW_RATIO
    rho = AIR_DENSITY
    
    solidity = 4.0 * c / (np.pi * R)
    Ct = solidity * a * (theta / 3.0 - lam / 2.0) / 2.0
    
    coefficient = Ct * rho * np.pi * R**4
    omega = np.sqrt(thrust_per_rotor / coefficient)
    rpm = omega * 30.0 / np.pi
    
    return rpm

def main():
    print("="*70)
    print("HELIX DRONE PHYSICS DIAGNOSTIC - Using Exact BET Formula")
    print("="*70)
    
    config = ExtendedEnvConfig()
    
    print(f"\n[1] Environment Configuration:")
    print(f"    mass:      {config.mass} kg")
    print(f"    hover_rpm: {config.hover_rpm} RPM")
    print(f"    rpm_range: {config.rpm_range} RPM")
    print(f"    max_rpm:   {config.max_rpm} RPM")
    print(f"    min_rpm:   {config.min_rpm} RPM")
    
    print(f"\n[2] Rotor Configuration (from RotorConfig defaults in PhysicsEngine.h):")
    print(f"    radius:       {ROTOR_RADIUS} m")
    print(f"    chord:        {ROTOR_CHORD} m")
    print(f"    pitch_angle:  {ROTOR_PITCH} rad ({np.rad2deg(ROTOR_PITCH):.1f}°)")
    print(f"    lift_slope:   {LIFT_SLOPE}")
    print(f"    inflow_ratio: {INFLOW_RATIO}")
    print(f"    air_density:  {AIR_DENSITY} kg/m³")
    
    # Calculate coefficients
    solidity = 4.0 * ROTOR_CHORD / (np.pi * ROTOR_RADIUS)
    Ct = solidity * LIFT_SLOPE * (ROTOR_PITCH / 3.0 - INFLOW_RATIO / 2.0) / 2.0
    print(f"\n[3] Derived Coefficients:")
    print(f"    solidity (sigma):   {solidity:.4f}")
    print(f"    thrust coeff (Ct):  {Ct:.6f}")
    
    # Weight analysis
    weight = config.mass * 9.81
    thrust_per_rotor_needed = weight / 4.0
    print(f"\n[4] Force Requirements:")
    print(f"    Weight:                 {weight:.4f} N")
    print(f"    Thrust/rotor needed:    {thrust_per_rotor_needed:.4f} N")
    
    # Find theoretical hover RPM
    theoretical_hover_rpm = find_hover_rpm_analytical(config.mass)
    print(f"\n[5] THEORETICAL HOVER RPM (from BET formula):")
    print(f"    >>> {theoretical_hover_rpm:.1f} RPM <<<")
    
    # Verify by computing thrust at that RPM
    thrust_at_theoretical = compute_thrust_bet(theoretical_hover_rpm)
    total_thrust_theoretical = 4 * thrust_at_theoretical
    print(f"\n    Verification:")
    print(f"    Thrust/rotor at {theoretical_hover_rpm:.0f} RPM: {thrust_at_theoretical:.4f} N")
    print(f"    Total thrust:                     {total_thrust_theoretical:.4f} N")
    print(f"    Weight:                           {weight:.4f} N")
    print(f"    Error:                            {100*(total_thrust_theoretical-weight)/weight:+.2f}%")
    
    # Compare with configured hover_rpm
    print(f"\n[6] CONFIGURATION COMPARISON:")
    print(f"    Configured hover_rpm:   {config.hover_rpm:.0f} RPM")
    print(f"    Theoretical hover_rpm:  {theoretical_hover_rpm:.1f} RPM")
    diff = abs(config.hover_rpm - theoretical_hover_rpm)
    diff_pct = 100 * diff / theoretical_hover_rpm
    print(f"    Difference:             {diff:.0f} RPM ({diff_pct:.1f}%)")
    
    # What thrust does configured hover_rpm produce?
    thrust_at_config = compute_thrust_bet(config.hover_rpm)
    total_thrust_config = 4 * thrust_at_config
    print(f"\n[7] THRUST AT CONFIGURED hover_rpm ({config.hover_rpm} RPM):")
    print(f"    Thrust/rotor:  {thrust_at_config:.4f} N")
    print(f"    Total thrust:  {total_thrust_config:.4f} N")
    print(f"    Weight:        {weight:.4f} N")
    net_force = total_thrust_config - weight
    accel = net_force / config.mass
    print(f"    Net force:     {net_force:+.4f} N")
    print(f"    Acceleration:  {accel:+.4f} m/s²")
    
    if net_force < -0.1:
        print(f"\n    🚨 PROBLEM: At hover_rpm={config.hover_rpm}, the drone will FALL!")
        print(f"       Acceleration: {accel:.2f} m/s² (should be 0)")
    elif net_force > 0.1:
        print(f"\n    ⚠️  At hover_rpm={config.hover_rpm}, the drone will RISE")
    else:
        print(f"\n    ✓ hover_rpm is correctly configured for hover")
    
    # Action range analysis
    print(f"\n[8] ACTION RANGE ANALYSIS:")
    action_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    print(f"    Action -> RPM -> Thrust/rotor -> Total -> Net Force")
    print(f"    " + "-"*66)
    for action in action_values:
        rpm = config.hover_rpm + config.rpm_range * action
        thrust = compute_thrust_bet(rpm)
        total = 4 * thrust
        net = total - weight
        effect = "FALL" if net < -0.5 else ("RISE" if net > 0.5 else "HOVER")
        print(f"    {action:+.1f} -> {rpm:6.0f} -> {thrust:6.3f} N -> {total:6.3f} N -> {net:+7.3f} N  [{effect}]")
    
    # Recommendation
    print(f"\n" + "="*70)
    print("DIAGNOSIS & RECOMMENDATION")
    print("="*70)
    
    if diff_pct > 20:
        print(f"""
🚨 CRITICAL PHYSICS MISMATCH DETECTED!

The configured hover_rpm ({config.hover_rpm}) is WRONG for this rotor configuration.

At {config.hover_rpm} RPM:
- Total thrust = {total_thrust_config:.3f} N
- Weight = {weight:.3f} N  
- Deficit = {weight - total_thrust_config:.3f} N ({100*(weight-total_thrust_config)/weight:.1f}% too weak)

The drone CANNOT hover at this RPM - it will always fall.

FIX REQUIRED in python_src/envs/drone_env.py ExtendedEnvConfig:
  hover_rpm: float = {theoretical_hover_rpm:.0f}  # was {config.hover_rpm}
  rpm_range: float = {theoretical_hover_rpm * 0.3:.0f}  # ~30% for control authority
  
This explains why the agent learned to output max action - it's desperately
trying to generate more thrust but even at action=+1.0, thrust is insufficient!
""")
    else:
        print(f"\n✓ Physics configuration appears correct.")
        print(f"  The hover_rpm of {config.hover_rpm} should produce stable hover.")

if __name__ == "__main__":
    main()
