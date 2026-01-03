"""Analyze motor differential in replay data"""
import csv

with open('replays/unity_replay_ep0.csv') as f:
    reader = csv.DictReader(f)
    data = list(reader)

print("="*70)
print("MOTOR DIFFERENTIAL ANALYSIS")
print("="*70)
print("\nStep | M1     | M2     | M3     | M4     | MaxDiff | Comment")
print("-"*70)

total_diff = 0
count = 0

for i, r in enumerate(data[5:]):  # Skip first 5 (warmup)
    m1 = float(r['rpm_0'])
    m2 = float(r['rpm_1'])
    m3 = float(r['rpm_2'])
    m4 = float(r['rpm_3'])
    
    if m1 < 100:  # Skip if motors off
        continue
    
    avg = (m1 + m2 + m3 + m4) / 4
    max_diff = max(abs(m1-avg), abs(m2-avg), abs(m3-avg), abs(m4-avg))
    total_diff += max_diff
    count += 1
    
    comment = ""
    if max_diff < 10:
        comment = "SYMMETRIC!"
    elif max_diff < 30:
        comment = "low diff"
    else:
        comment = "good diff"
    
    if i < 25:  # Print first 25
        print(f"{i+5:4d} | {m1:6.0f} | {m2:6.0f} | {m3:6.0f} | {m4:6.0f} | {max_diff:7.1f} | {comment}")

print("-"*70)
print(f"\nAverage max differential: {total_diff/count:.1f} RPM")

# Check if actor is outputting symmetric actions
print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if total_diff/count < 20:
    print("""
🚨 SYMMETRIC MOTOR OUTPUT DETECTED!

The 4 motors are running almost identically. This means:
1. The Actor network is outputting the SAME value for all 4 motors
2. Without differential, the drone CANNOT correct roll/pitch
3. This is why it flips over and crashes

CAUSE: The Actor learned that symmetric output is "safe" because:
- Different outputs caused erratic behavior during early training
- The reward function doesn't sufficiently penalize symmetric outputs
- The Motor Mixer might not be working correctly
""")
else:
    print(f"Motor differential looks healthy at {total_diff/count:.1f} RPM average.")

# Also check if RPM is stuck at ceiling
rpms = [float(r['rpm_0']) for r in data[10:] if float(r['rpm_0']) > 100]
max_rpm = max(rpms)
min_rpm = min(rpms)
print(f"\nRPM Range in replay: {min_rpm:.0f} - {max_rpm:.0f}")
print(f"Expected max (action=+1): 4500 RPM")
print(f"Actual max achieved: {max_rpm:.0f} RPM")

if max_rpm < 3000:
    print("""
🚨 RPM CEILING DETECTED!

The drone never exceeds ~2500 RPM even though it SHOULD be able to reach 4500.
This means the Agent learned to output action ≈ 0 and never learned to increase thrust.

POSSIBLE CAUSES:
1. Early crashes taught the agent that increasing RPM = bad
2. The Critic values "do nothing" higher than "try to fly"
3. The observation doesn't give enough info about falling
""")
