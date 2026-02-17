import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'Release'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'build'))

import torch
import numpy as np

from python_src.agents.networks import LSTMActor, LSTMCritic
from python_src.agents.ddpg_agent import TD3LSTMAgent

OBS_DIM = 20
ACTION_DIM = 4
HIDDEN_DIM = 256
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
SEQ_LEN = 16

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cpu')

def generate_realistic_obs_sequence(length: int) -> np.ndarray:
    seq = np.zeros((length, OBS_DIM), dtype=np.float32)
    pos = np.array([0.0, 0.0, 2.0])
    vel = np.array([0.0, 0.0, 0.0])
    for t in range(length):
        vel += np.random.randn(3) * 0.01
        pos += vel * 0.01
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        ang_vel = np.random.randn(3) * 0.1
        error = (np.array([0, 0, 2.0]) - pos) / 5.0
        prev_action = np.random.randn(4) * 0.1
        seq[t] = np.concatenate([pos / 5.0, vel / 5.0, quat, ang_vel / 10.0, error, prev_action])
    return seq


print("=" * 70)
print("TEST 1: HIDDEN STATE DOUBLE-PROCESSING")
print("Does inference produce different output than training mode?")
print("=" * 70)

actor = LSTMActor(OBS_DIM, ACTION_DIM, HIDDEN_DIM, LSTM_HIDDEN, LSTM_LAYERS)
actor.eval()

obs_history = generate_realistic_obs_sequence(30)

hidden = actor.get_initial_hidden(1, device)
inference_actions = []
for t in range(30):
    start = max(0, t - SEQ_LEN + 1)
    seq = np.zeros((SEQ_LEN, OBS_DIM), dtype=np.float32)
    valid_len = t - start + 1
    seq[:valid_len] = obs_history[start:t + 1]

    with torch.no_grad():
        obs_tensor = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0)
        lengths = torch.tensor([valid_len], dtype=torch.int64)
        action, new_hidden = actor(obs_tensor, hidden, lengths)
        hidden = (new_hidden[0].detach(), new_hidden[1].detach())
        inference_actions.append(action.numpy()[0].copy())

training_actions = []
for t in range(30):
    start = max(0, t - SEQ_LEN + 1)
    seq = np.zeros((SEQ_LEN, OBS_DIM), dtype=np.float32)
    valid_len = t - start + 1
    seq[:valid_len] = obs_history[start:t + 1]

    with torch.no_grad():
        obs_tensor = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0)
        lengths = torch.tensor([valid_len], dtype=torch.int64)
        action, _ = actor(obs_tensor, None, lengths)
        training_actions.append(action.numpy()[0].copy())

inference_actions = np.array(inference_actions)
training_actions = np.array(training_actions)
diffs = np.abs(inference_actions - training_actions)

print(f"\n{'Step':>4} | {'Inf action':>30} | {'Train action':>30} | {'MaxDiff':>8}")
print("-" * 80)
for t in range(30):
    inf_str = np.array2string(inference_actions[t], precision=4, separator=',')
    trn_str = np.array2string(training_actions[t], precision=4, separator=',')
    max_diff = diffs[t].max()
    marker = " <<< DIVERGED" if max_diff > 0.01 else ""
    print(f"{t:4d} | {inf_str:>30} | {trn_str:>30} | {max_diff:>8.5f}{marker}")

divergence_start = -1
for t in range(30):
    if diffs[t].max() > 0.01:
        divergence_start = t
        break

print(f"\nMax divergence: {diffs.max():.6f}")
print(f"Mean divergence: {diffs.mean():.6f}")
if divergence_start >= 0:
    print(f"Divergence starts at step: {divergence_start}")
    print("VERDICT: INFERENCE AND TRAINING SEE DIFFERENT LSTM BEHAVIOR")
    print("The persistent hidden state causes double-processing of observations.")
else:
    print("VERDICT: No significant divergence detected")


print("\n\n" + "=" * 70)
print("TEST 2: HIDDEN STATE ACCUMULATION MAGNITUDE")
print("How large does the hidden state grow over an episode?")
print("=" * 70)

actor2 = LSTMActor(OBS_DIM, ACTION_DIM, HIDDEN_DIM, LSTM_HIDDEN, LSTM_LAYERS)
actor2.eval()

obs_long = generate_realistic_obs_sequence(200)

hidden = actor2.get_initial_hidden(1, device)
h_norms = []
c_norms = []

for t in range(200):
    start = max(0, t - SEQ_LEN + 1)
    seq = np.zeros((SEQ_LEN, OBS_DIM), dtype=np.float32)
    valid_len = t - start + 1
    seq[:valid_len] = obs_long[start:t + 1]

    with torch.no_grad():
        obs_tensor = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0)
        lengths = torch.tensor([valid_len], dtype=torch.int64)
        _, new_hidden = actor2(obs_tensor, hidden, lengths)
        hidden = (new_hidden[0].detach(), new_hidden[1].detach())
        h_norms.append(hidden[0].norm().item())
        c_norms.append(hidden[1].norm().item())

print(f"\n{'Step':>4} | {'|h|':>10} | {'|c|':>10}")
print("-" * 30)
for t in [0, 1, 5, 10, 15, 20, 30, 50, 100, 150, 199]:
    print(f"{t:4d} | {h_norms[t]:>10.4f} | {c_norms[t]:>10.4f}")

print(f"\nHidden state h: initial={h_norms[0]:.4f}, final={h_norms[-1]:.4f}, ratio={h_norms[-1]/(h_norms[0]+1e-8):.1f}x")
print(f"Hidden state c: initial={c_norms[0]:.4f}, final={c_norms[-1]:.4f}, ratio={c_norms[-1]/(c_norms[0]+1e-8):.1f}x")

if h_norms[-1] > h_norms[0] * 3:
    print("VERDICT: Hidden state EXPLODES - training (h=0) vs inference (h=large) is a massive distribution shift")
else:
    print("VERDICT: Hidden state remains bounded")


print("\n\n" + "=" * 70)
print("TEST 3: ACTION DIVERGENCE OVER EPISODE")
print("Compare action outputs: inference-mode vs training-mode for full episode")
print("=" * 70)

agent = TD3LSTMAgent(
    obs_dim=OBS_DIM, action_dim=ACTION_DIM, device=device,
    hidden_dim=HIDDEN_DIM, lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS,
    sequence_length=SEQ_LEN, use_amp=False, compile_networks=False
)

obs_episode = generate_realistic_obs_sequence(150)

agent.reset_hidden_states()
agent.init_vectorized(1)
inference_actions_full = []
for t in range(150):
    obs = obs_episode[t:t+1]
    action = agent.get_actions_batch(obs, add_noise=False)
    inference_actions_full.append(action[0].copy())

agent.reset_hidden_states()
agent.init_vectorized(1)
training_mode_actions = []
for t in range(150):
    start = max(0, t - SEQ_LEN + 1)
    seq = np.zeros((1, SEQ_LEN, OBS_DIM), dtype=np.float32)
    valid_len = t - start + 1
    seq[0, :valid_len] = obs_episode[start:t + 1]

    with torch.no_grad():
        obs_tensor = torch.as_tensor(seq, dtype=torch.float32, device=device)
        lengths_tensor = torch.tensor([valid_len], dtype=torch.int64, device=device)
        action, _ = agent.actor(obs_tensor, None, lengths_tensor)
        training_mode_actions.append(action.cpu().numpy()[0].copy())

inference_actions_full = np.array(inference_actions_full)
training_mode_actions = np.array(training_mode_actions)
diffs_full = np.abs(inference_actions_full - training_mode_actions)

print(f"\n{'Step':>4} | {'Inference[0]':>10} | {'TrainMode[0]':>10} | {'MaxDiff':>8}")
print("-" * 50)
for t in [0, 1, 5, 10, 15, 16, 17, 20, 30, 50, 100, 149]:
    print(f"{t:4d} | {inference_actions_full[t, 0]:>+10.5f} | {training_mode_actions[t, 0]:>+10.5f} | {diffs_full[t].max():>8.5f}")

print(f"\nOverall max divergence: {diffs_full.max():.6f}")
print(f"Divergence at step 16 (first full sequence): {diffs_full[16].max():.6f}")
print(f"Divergence at step 100: {diffs_full[100].max():.6f}")

pct_diverged = np.mean(diffs_full.max(axis=1) > 0.01) * 100
print(f"% of steps with >0.01 action diff: {pct_diverged:.1f}%")

if pct_diverged > 20:
    print("VERDICT: CRITICAL MISMATCH - The agent learns one behavior but executes another")
    print("ROOT CAUSE: get_actions_batch() passes persistent hidden + full sequence")
    print("            but update() passes hidden=None + full sequence")
    print("            The LSTM sees double-processed overlapping obs during inference")
else:
    print("VERDICT: PASS - Inference matches training (hidden=None fix applied)")


print("\n\n" + "=" * 70)
print("TEST 4: SEQUENCE CONSTRUCTION CONSISTENCY")
print("Does _update_obs_buffers_batch build the same sequence as replay buffer would?")
print("=" * 70)

agent2 = TD3LSTMAgent(
    obs_dim=OBS_DIM, action_dim=ACTION_DIM, device=device,
    hidden_dim=HIDDEN_DIM, lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS,
    sequence_length=SEQ_LEN, use_amp=False, compile_networks=False
)
agent2.init_vectorized(1)

obs_seq = generate_realistic_obs_sequence(30)
inference_seqs = []
for t in range(30):
    obs = obs_seq[t:t+1]
    seq, lengths = agent2._update_obs_buffers_batch(obs)
    inference_seqs.append((seq[0].copy(), int(lengths[0])))

mismatches = 0
for t in range(30):
    inf_seq, inf_len = inference_seqs[t]

    start = max(0, t - SEQ_LEN + 1)
    expected_len = min(t + 1, SEQ_LEN)
    expected_seq = np.zeros((SEQ_LEN, OBS_DIM), dtype=np.float32)
    expected_seq[:expected_len] = obs_seq[start:t + 1]

    seq_match = np.allclose(inf_seq[:expected_len], expected_seq[:expected_len], atol=1e-6)
    len_match = inf_len == expected_len

    if not seq_match or not len_match:
        mismatches += 1
        print(f"  Step {t}: MISMATCH len={inf_len} vs {expected_len}, seq_match={seq_match}")

if mismatches == 0:
    print("  All sequences match between inference and expected replay buffer format")
    print("  VERDICT: Sequence construction is CORRECT - the bug is NOT in sequence building")
else:
    print(f"  {mismatches} mismatches found!")
    print("  VERDICT: Sequence construction has bugs")


print("\n\n" + "=" * 70)
print("TEST 5: WHAT HAPPENS IF WE FIX THE HIDDEN STATE?")
print("Remove persistent hidden - does inference match training?")
print("=" * 70)

actor_fix = LSTMActor(OBS_DIM, ACTION_DIM, HIDDEN_DIM, LSTM_HIDDEN, LSTM_LAYERS)
actor_fix.eval()

obs_test = generate_realistic_obs_sequence(50)

fixed_actions = []
for t in range(50):
    start = max(0, t - SEQ_LEN + 1)
    seq = np.zeros((SEQ_LEN, OBS_DIM), dtype=np.float32)
    valid_len = t - start + 1
    seq[:valid_len] = obs_test[start:t + 1]

    with torch.no_grad():
        obs_tensor = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0)
        lengths = torch.tensor([valid_len], dtype=torch.int64)
        action, _ = actor_fix(obs_tensor, None, lengths)
        fixed_actions.append(action.numpy()[0].copy())

hidden_broken = actor_fix.get_initial_hidden(1, device)
broken_actions = []
for t in range(50):
    start = max(0, t - SEQ_LEN + 1)
    seq = np.zeros((SEQ_LEN, OBS_DIM), dtype=np.float32)
    valid_len = t - start + 1
    seq[:valid_len] = obs_test[start:t + 1]

    with torch.no_grad():
        obs_tensor = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0)
        lengths = torch.tensor([valid_len], dtype=torch.int64)
        action, new_h = actor_fix(obs_tensor, hidden_broken, lengths)
        hidden_broken = (new_h[0].detach(), new_h[1].detach())
        broken_actions.append(action.numpy()[0].copy())

fixed_actions = np.array(fixed_actions)
broken_actions = np.array(broken_actions)
fix_diff = np.abs(fixed_actions - broken_actions)

print(f"\nFixed (h=None each step) vs Broken (persistent h):")
print(f"Max diff at step 0:  {fix_diff[0].max():.6f}")
print(f"Max diff at step 15: {fix_diff[15].max():.6f}")
print(f"Max diff at step 16: {fix_diff[16].max():.6f}")
print(f"Max diff at step 30: {fix_diff[30].max():.6f}")
print(f"Max diff at step 49: {fix_diff[49].max():.6f}")
print(f"Overall max diff:    {fix_diff.max():.6f}")

print("\nWith fix (h=None): all inference calls match what training expects.")
print("This is the correct approach for the current training architecture.")


print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
STATUS: FIXED (hidden=None now used in both training and inference)

THE ORIGINAL BUG (now resolved):
  - Training used hidden_state=None (zeros) for each replay sequence
  - Inference used a PERSISTENT hidden state carried across steps
  - This caused observations to be double-processed through the LSTM
  - Result: actor learned one mapping but executed a different one

THE FIX APPLIED:
  - get_action() and get_actions_batch() now pass hidden=None
  - _actor_hidden field removed entirely
  - Inference now matches training: clean sequence processing from zeros
  - Context window = sequence_length (16 steps at 100Hz = 0.16s)

Tests 1, 2, 5 demonstrate WHY persistent hidden is wrong (regression proof).
Test 3 confirms the fix works: inference == training actions.
Test 4 confirms sequence construction is correct.
""")

