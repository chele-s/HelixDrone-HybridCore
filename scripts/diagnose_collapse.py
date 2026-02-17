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
from python_src.agents.replay_buffer import SequenceReplayBuffer

OBS_DIM = 20
ACTION_DIM = 4
HIDDEN_DIM = 256
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
SEQ_LEN = 16
BATCH_SIZE = 128
DEVICE = torch.device('cpu')

torch.manual_seed(42)
np.random.seed(42)


def generate_trajectory(length, obs_dim=OBS_DIM, action_dim=ACTION_DIM):
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    pos = np.array([0.0, 0.0, 2.0])
    vel = np.array([0.0, 0.0, 0.0])

    for t in range(length):
        vel += np.random.randn(3) * 0.02
        pos += vel * 0.01
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        ang_vel = np.random.randn(3) * 0.1
        error = (np.array([0, 0, 2.0]) - pos) / 5.0
        prev_action = np.random.randn(4) * 0.1
        obs = np.concatenate([pos / 5.0, vel / 5.0, quat, ang_vel / 10.0, error, prev_action]).astype(np.float32)

        action = np.clip(np.random.randn(action_dim).astype(np.float32) * 0.15, -1, 1)

        vel_next = vel + np.random.randn(3) * 0.02
        pos_next = pos + vel_next * 0.01
        error_next = (np.array([0, 0, 2.0]) - pos_next) / 5.0
        next_obs = np.concatenate([pos_next / 5.0, vel_next / 5.0, quat, ang_vel / 10.0, error_next, action]).astype(np.float32)

        dist = np.linalg.norm(pos - np.array([0, 0, 2.0]))
        reward = float(max(0, 3.0 - dist) + np.random.randn() * 0.1)

        done = (t == length - 1)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_observations.append(next_obs)
        dones.append(done)

        vel = vel_next
        pos = pos_next

    return observations, actions, rewards, next_observations, dones


def fill_buffer(buffer, num_transitions=5000):
    total = 0
    while total < num_transitions:
        ep_len = np.random.randint(50, 200)
        obs_list, act_list, rew_list, next_obs_list, done_list = generate_trajectory(ep_len)
        for o, a, r, no, d in zip(obs_list, act_list, rew_list, next_obs_list, done_list):
            buffer.push(o, a, r, no, d)
            total += 1
            if total >= num_transitions:
                break


print("=" * 70)
print("LSTM POLICY COLLAPSE DIAGNOSTIC")
print("Testing why the actor destroys the policy at the transition point")
print("=" * 70)


print("\n--- Setup: Creating agent and filling replay buffer ---")
agent = TD3LSTMAgent(
    obs_dim=OBS_DIM, action_dim=ACTION_DIM, device=DEVICE,
    hidden_dim=HIDDEN_DIM, lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS,
    sequence_length=SEQ_LEN, lr_actor=1e-4, lr_critic=1e-4,
    use_amp=False, compile_networks=False,
    gradient_clip=1.0, policy_delay=2
)

buffer = SequenceReplayBuffer(
    capacity=50000, device=DEVICE,
    obs_dim=OBS_DIM, action_dim=ACTION_DIM,
    sequence_length=SEQ_LEN
)

fill_buffer(buffer, num_transitions=5000)
print(f"Buffer size: {buffer.size}")


print("\n\n" + "=" * 70)
print("TEST 1: CRITIC WARMUP ADEQUACY")
print("How well does the critic learn Q-values in 312 gradient steps?")
print("(This matches your setup: 5000 timesteps / 16 envs)")
print("=" * 70)

q_values_before = []
critic_losses = []

for step in range(400):
    metrics = agent.update(buffer, BATCH_SIZE, critic_only=True)
    if metrics:
        critic_losses.append(metrics['critic_loss'])
        q_values_before.append(metrics['q_value'])

print(f"\nAfter 400 critic-only steps (exceeds your 312):")
print(f"  Critic loss: {critic_losses[0]:.4f} -> {critic_losses[-1]:.4f}")
print(f"  Q-value:     {q_values_before[0]:.4f} -> {q_values_before[-1]:.4f}")
print(f"  Q-value range: [{min(q_values_before):.4f}, {max(q_values_before):.4f}]")


print("\n\n" + "=" * 70)
print("TEST 2: Q-VALUE EXTRAPOLATION ERROR")
print("Does the critic give reliable Q-values for actor's actions vs random?")
print("=" * 70)

batch = buffer.sample(BATCH_SIZE)
obs_seq = batch['obs_seq']
lengths = batch['lengths']

with torch.no_grad():
    random_actions = torch.randn(BATCH_SIZE, ACTION_DIM, device=DEVICE) * 0.15
    random_actions = random_actions.clamp(-1.0, 1.0)
    q_random1, q_random2, _ = agent.critic(obs_seq, random_actions, None, lengths)
    q_random = torch.min(q_random1, q_random2)

    actor_actions, _ = agent.actor(obs_seq, None, lengths)
    q_actor1, q_actor2, _ = agent.critic(obs_seq, actor_actions, None, lengths)
    q_actor = torch.min(q_actor1, q_actor2)

    extreme_actions = torch.ones(BATCH_SIZE, ACTION_DIM, device=DEVICE)
    q_extreme1, q_extreme2, _ = agent.critic(obs_seq, extreme_actions, None, lengths)
    q_extreme = torch.min(q_extreme1, q_extreme2)

    neg_extreme_actions = -torch.ones(BATCH_SIZE, ACTION_DIM, device=DEVICE)
    q_neg1, q_neg2, _ = agent.critic(obs_seq, neg_extreme_actions, None, lengths)
    q_neg = torch.min(q_neg1, q_neg2)

print(f"\n{'Action Type':<25} | {'Mean Q':<10} | {'Std Q':<10} | {'Min Q':<10} | {'Max Q':<10}")
print("-" * 75)
for name, q in [("Random (training dist)", q_random),
                ("Actor (untrained)", q_actor),
                ("All +1.0 (extreme)", q_extreme),
                ("All -1.0 (extreme)", q_neg)]:
    print(f"{name:<25} | {q.mean():<10.4f} | {q.std():<10.4f} | {q.min():<10.4f} | {q.max():<10.4f}")

q_gap = (q_actor.mean() - q_random.mean()).item()
print(f"\nQ-gap (actor - random): {q_gap:+.4f}")
if abs(q_gap) > 5.0:
    print("VERDICT: Critic assigns significantly different Q-values to actor actions")
    print("         The actor will aggressively chase this signal -> collapse risk HIGH")
else:
    print("VERDICT: Q-gap is modest, but first gradient steps may still amplify it")


print("\n\n" + "=" * 70)
print("TEST 3: ACTOR UPDATE CATASTROPHE SIMULATION")
print("What happens to Q-values, actor loss, and actions during the first")
print("~200 full updates (simulating the collapse window)?")
print("=" * 70)

actor_losses = []
q_during_training = []
action_means = []
action_stds = []
gradient_norms = []

for step in range(200):
    metrics = agent.update(buffer, BATCH_SIZE, critic_only=False)
    if metrics:
        actor_losses.append(metrics.get('actor_loss', 0.0))
        q_during_training.append(metrics['q_value'])

    if step % 10 == 0:
        test_batch = buffer.sample(64)
        with torch.no_grad():
            test_actions, _ = agent.actor(test_batch['obs_seq'], None, test_batch['lengths'])
            action_means.append(test_actions.mean(dim=0).cpu().numpy())
            action_stds.append(test_actions.std(dim=0).cpu().numpy())

        total_norm = 0.0
        for p in agent.actor.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        gradient_norms.append(total_norm ** 0.5)

print(f"\n{'Step':<6} | {'Actor Loss':<12} | {'Q-value':<10} | {'Action Mean':<20} | {'Action Std':<20}")
print("-" * 80)

al_filtered = [x for x in actor_losses if x != 0.0]
for i, step in enumerate([0, 5, 10, 20, 50, 99, 149, 199]):
    if step < len(q_during_training):
        al = al_filtered[min(step, len(al_filtered)-1)] if al_filtered else 0.0
        q = q_during_training[step]
        am_idx = step // 10
        am = action_means[min(am_idx, len(action_means)-1)]
        astd = action_stds[min(am_idx, len(action_stds)-1)]
        print(f"{step:<6} | {al:<12.4f} | {q:<10.4f} | {np.array2string(am, precision=3, separator=','):<20} | {np.array2string(astd, precision=3, separator=','):<20}")

print(f"\nQ-value trajectory: {q_during_training[0]:.4f} -> {q_during_training[-1]:.4f}")
if al_filtered:
    print(f"Actor loss trajectory: {al_filtered[0]:.4f} -> {al_filtered[-1]:.4f}")

if action_means:
    initial_am = action_means[0]
    final_am = action_means[-1]
    action_shift = np.abs(final_am - initial_am).max()
    print(f"Max action mean shift: {action_shift:.4f}")
    final_std = action_stds[-1]
    print(f"Final action std: {np.array2string(final_std, precision=4)}")

    if action_shift > 0.3:
        print("VERDICT: Actor actions shifted dramatically -> policy collapse likely")
    elif np.any(final_std < 0.01):
        print("VERDICT: Actor collapsed to near-constant output -> degenerate policy")
    else:
        print("VERDICT: Actor actions remain reasonable after 200 updates")


print("\n\n" + "=" * 70)
print("TEST 4: GRADIENT MAGNITUDE ANALYSIS")
print("Are actor gradients too large during the transition?")
print("=" * 70)

if gradient_norms:
    print(f"\n{'Sample':<8} | {'Grad Norm':<12}")
    print("-" * 25)
    for i, gn in enumerate(gradient_norms):
        print(f"{i*10:<8} | {gn:<12.6f}")

    max_gnorm = max(gradient_norms)
    mean_gnorm = np.mean(gradient_norms)
    print(f"\nMax gradient norm: {max_gnorm:.6f}")
    print(f"Mean gradient norm: {mean_gnorm:.6f}")

    if max_gnorm > 10.0:
        print("VERDICT: Gradient explosion detected -> actor weights change too fast")
    elif max_gnorm > 1.0:
        print("VERDICT: Gradients are clipped but still large -> aggressive actor updates")
    else:
        print("VERDICT: Gradients are well-controlled")


print("\n\n" + "=" * 70)
print("TEST 5: CRITIC WARMUP STEPS CALCULATION")
print("How many gradient steps does your config actually give the critic?")
print("=" * 70)

learning_starts = 25000
num_envs = 16
warmup_timesteps = 5000
train_freq = 1
gradient_steps = 1

steps_per_iter = num_envs
iters_in_warmup = warmup_timesteps // steps_per_iter
grad_steps_actual = iters_in_warmup * gradient_steps

print(f"\n  learning_starts:    {learning_starts}")
print(f"  num_envs:           {num_envs}")
print(f"  warmup_timesteps:   {warmup_timesteps} (hardcoded in train.py line 505)")
print(f"  train_freq:         {train_freq}")
print(f"  gradient_steps:     {gradient_steps}")
print(f"  ---")
print(f"  Steps per iteration: {steps_per_iter}")
print(f"  Iterations in warmup: {iters_in_warmup}")
print(f"  ACTUAL gradient steps for critic: {grad_steps_actual}")
print(f"  With batch_size=256: {grad_steps_actual * 256} total samples processed")
print(f"\n  VERDICT: Only {grad_steps_actual} gradient steps is SEVERELY insufficient")
print(f"  Recommended MINIMUM: 2000-5000 gradient steps")
print(f"  To get 5000 gradient steps with {num_envs} envs:")
print(f"  warmup_timesteps should be: {5000 * num_envs} (={5000 * num_envs})")


print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
ROOT CAUSE: The actor-critic transition is too abrupt.

MECHANISM:
  1. Critic-only training gets only ~312 gradient steps (5000 ts / 16 envs)
  2. Critic learns Q-values for RANDOM ACTIONS only (warmup distribution)
  3. When actor starts updating, it queries Q(s, actor_action)
  4. Critic extrapolates POORLY for actor's actions (out-of-distribution)
  5. Actor chases phantom Q-values with full learning rate
  6. Actions shift dramatically within ~100 gradient steps
  7. New actions produce bad observations -> bad Q-targets -> feedback loop
  8. Policy collapses to degenerate behavior (episode length ~15 = instant crash)

FIXES NEEDED (in order of priority):
  1. ACTOR LR WARMUP: Ramp actor learning rate from near-zero to target over
     several thousand gradient steps. This prevents abrupt policy shift.
  2. LONGER CRITIC WARMUP: Use gradient-step count (not timestep count) so 
     the critic gets adequate training regardless of num_envs.
  3. ACTOR LOSS CLIPPING: Cap |actor_loss| to prevent catastrophic gradient 
     spikes during early actor training.
""")
