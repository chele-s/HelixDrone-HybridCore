import csv

with open('replays/unity_replay_ep0.csv') as f:
    reader = csv.DictReader(f)
    data = list(reader)

print("Target Analysis:")
print("="*60)
print(f"{'Step':>4} | {'Target X':>8} | {'Target Y':>8} | {'Target Z':>8} | {'Drone Z':>7}")
print("-"*60)

for i, r in enumerate(data[:25]):
    tx = float(r['target_x'])
    ty = float(r['target_y'])
    tz = float(r['target_z'])
    dz = float(r['pos_z'])
    print(f"{i:4d} | {tx:8.2f} | {ty:8.2f} | {tz:8.2f} | {dz:7.2f}")

print("\n" + "="*60)
print("Summary:")
targets = [(float(r['target_x']), float(r['target_y']), float(r['target_z'])) for r in data]
tx_range = (min(t[0] for t in targets), max(t[0] for t in targets))
ty_range = (min(t[1] for t in targets), max(t[1] for t in targets))
tz_range = (min(t[2] for t in targets), max(t[2] for t in targets))

print(f"Target X range: {tx_range[0]:.2f} to {tx_range[1]:.2f}")
print(f"Target Y range: {ty_range[0]:.2f} to {ty_range[1]:.2f}")
print(f"Target Z range: {tz_range[0]:.2f} to {tz_range[1]:.2f}")

if abs(tx_range[0]) < 0.01 and abs(tx_range[1]) < 0.01:
    print("\n⚠️  Target X is always 0 - might be a problem")
if abs(ty_range[0] - 2.0) < 0.01 and abs(ty_range[1] - 2.0) < 0.01:
    print("⚠️  Target Y is always 2.0 - static target")
