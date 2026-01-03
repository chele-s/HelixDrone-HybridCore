import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))

import numpy as np
from python_src.envs.motor_mixer import MotorMixer, MotorMixerConfig


def test_motor_mixer():
    print("=" * 60)
    print("MOTOR MIXER VERIFICATION")
    print("=" * 60)
    
    mixer = MotorMixer(MotorMixerConfig())
    
    tests = [
        ("Pure thrust +1", [1.0, 0.0, 0.0, 0.0]),
        ("Pure roll +0.5", [0.0, 0.5, 0.0, 0.0]),
        ("Pure pitch +0.5", [0.0, 0.0, 0.5, 0.0]),
        ("Pure yaw +0.5", [0.0, 0.0, 0.0, 0.5]),
        ("Hover command", [0.5, 0.0, 0.0, 0.0]),
        ("Thrust + roll", [0.5, 0.3, 0.0, 0.0]),
    ]
    
    for name, cmd in tests:
        motors = mixer.mix(np.array(cmd))
        print(f"\n{name}: {cmd}")
        print(f"  Motors: [{motors[0]:+.2f}, {motors[1]:+.2f}, {motors[2]:+.2f}, {motors[3]:+.2f}]")
        print(f"  Range: {max(motors) - min(motors):.2f}")
    
    print("\n" + "=" * 60)
    print("DIFFERENTIAL CONTROL CHECK")
    print("=" * 60)
    
    roll_cmd = mixer.mix(np.array([0.5, 0.3, 0.0, 0.0]))
    print(f"\nRoll +0.3 command:")
    print(f"  M1 (front-left): {roll_cmd[0]:+.2f}")
    print(f"  M2 (front-right): {roll_cmd[1]:+.2f}")
    print(f"  M3 (back-right): {roll_cmd[2]:+.2f}")
    print(f"  M4 (back-left): {roll_cmd[3]:+.2f}")
    
    if roll_cmd[1] > roll_cmd[0] and roll_cmd[2] > roll_cmd[3]:
        print("  >>> PASS: Right motors higher for positive roll")
    else:
        print("  >>> FAIL: Differential not correct!")
    
    pitch_cmd = mixer.mix(np.array([0.5, 0.0, 0.3, 0.0]))
    print(f"\nPitch +0.3 command:")
    print(f"  M1 (front-left): {pitch_cmd[0]:+.2f}")
    print(f"  M2 (front-right): {pitch_cmd[1]:+.2f}")
    print(f"  M3 (back-right): {pitch_cmd[2]:+.2f}")
    print(f"  M4 (back-left): {pitch_cmd[3]:+.2f}")
    
    if pitch_cmd[2] > pitch_cmd[0] and pitch_cmd[3] > pitch_cmd[1]:
        print("  >>> PASS: Back motors higher for positive pitch")
    else:
        print("  >>> FAIL: Differential not correct!")


if __name__ == "__main__":
    test_motor_mixer()
