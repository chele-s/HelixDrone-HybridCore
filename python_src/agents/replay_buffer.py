import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
from collections import deque
import random


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        if state_dim is not None and action_dim is not None:
            self.states = np.zeros((capacity, state_dim), dtype=np.float32)
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
            self.rewards = np.zeros((capacity, 1), dtype=np.float32)
            self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
            self.dones = np.zeros((capacity, 1), dtype=np.float32)
            self._preallocated = True
        else:
            self.buffer = deque(maxlen=capacity)
            self._preallocated = False
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        if self._preallocated:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.dones[self.ptr] = float(done)
            
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        else:
            self.buffer.append((state, action, reward, next_state, done))
    
    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> None:
        batch_size = states.shape[0]
        
        if self._preallocated:
            end_ptr = self.ptr + batch_size
            
            if end_ptr <= self.capacity:
                self.states[self.ptr:end_ptr] = states
                self.actions[self.ptr:end_ptr] = actions
                self.rewards[self.ptr:end_ptr] = rewards.reshape(-1, 1)
                self.next_states[self.ptr:end_ptr] = next_states
                self.dones[self.ptr:end_ptr] = dones.reshape(-1, 1).astype(np.float32)
            else:
                first_part = self.capacity - self.ptr
                second_part = end_ptr - self.capacity
                
                self.states[self.ptr:] = states[:first_part]
                self.states[:second_part] = states[first_part:]
                
                self.actions[self.ptr:] = actions[:first_part]
                self.actions[:second_part] = actions[first_part:]
                
                self.rewards[self.ptr:] = rewards[:first_part].reshape(-1, 1)
                self.rewards[:second_part] = rewards[first_part:].reshape(-1, 1)
                
                self.next_states[self.ptr:] = next_states[:first_part]
                self.next_states[:second_part] = next_states[first_part:]
                
                self.dones[self.ptr:] = dones[:first_part].reshape(-1, 1).astype(np.float32)
                self.dones[:second_part] = dones[first_part:].reshape(-1, 1).astype(np.float32)
            
            self.ptr = end_ptr % self.capacity
            self.size = min(self.size + batch_size, self.capacity)
        else:
            for i in range(batch_size):
                self.buffer.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))
    
    def sample(
        self, 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._preallocated:
            indices = np.random.randint(0, self.size, size=batch_size)
            
            return (
                torch.FloatTensor(self.states[indices]).to(self.device),
                torch.FloatTensor(self.actions[indices]).to(self.device),
                torch.FloatTensor(self.rewards[indices]).to(self.device),
                torch.FloatTensor(self.next_states[indices]).to(self.device),
                torch.FloatTensor(self.dones[indices]).to(self.device)
            )
        else:
            batch = random.sample(list(self.buffer), batch_size)
            state, action, reward, next_state, done = zip(*batch)
            
            return (
                torch.FloatTensor(np.array(state)).to(self.device),
                torch.FloatTensor(np.array(action)).to(self.device),
                torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device),
                torch.FloatTensor(np.array(next_state)).to(self.device),
                torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)
            )
    
    def __len__(self) -> int:
        if self._preallocated:
            return self.size
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
    
    def update(self, tree_idx: int, priority: float) -> None:
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def update_batch(self, tree_indices: np.ndarray, priorities: np.ndarray) -> None:
        for tree_idx, priority in zip(tree_indices, priorities):
            self.update(tree_idx, priority)
    
    def add(self, priority: float) -> int:
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        
        data_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        return data_idx
    
    def add_batch(self, priorities: np.ndarray) -> np.ndarray:
        batch_size = len(priorities)
        data_indices = np.zeros(batch_size, dtype=np.int32)
        
        for i, priority in enumerate(priorities):
            data_indices[i] = self.add(priority)
        
        return data_indices
    
    def get(self, value: float) -> Tuple[int, float, int]:
        parent_idx = 0
        
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            
            if left_idx >= len(self.tree):
                break
            
            if value <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                value -= self.tree[left_idx]
                parent_idx = right_idx
        
        data_idx = parent_idx - self.capacity + 1
        return parent_idx, self.tree[parent_idx], data_idx
    
    @property
    def total_priority(self) -> float:
        return self.tree[0]
    
    @property
    def max_priority(self) -> float:
        return np.max(self.tree[self.capacity - 1:])
    
    @property
    def min_priority(self) -> float:
        leaves = self.tree[self.capacity - 1:]
        non_zero = leaves[leaves > 0]
        return np.min(non_zero) if len(non_zero) > 0 else 1.0


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6
    ):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1
        
        self.tree = SumTree(capacity)
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.size = 0
        self.max_priority = 1.0
    
    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        priority = self.max_priority ** self.alpha
        
        data_idx = self.tree.add(priority)
        
        self.states[data_idx] = state
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.next_states[data_idx] = next_state
        self.dones[data_idx] = float(done)
        
        self.size = min(self.size + 1, self.capacity)
    
    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> None:
        batch_size = states.shape[0]
        priority = self.max_priority ** self.alpha
        priorities = np.full(batch_size, priority, dtype=np.float64)
        
        data_indices = self.tree.add_batch(priorities)
        
        self.states[data_indices] = states
        self.actions[data_indices] = actions
        self.rewards[data_indices] = rewards.reshape(-1, 1)
        self.next_states[data_indices] = next_states
        self.dones[data_indices] = dones.reshape(-1, 1).astype(np.float32)
        
        self.size = min(self.size + batch_size, self.capacity)
    
    def sample(
        self, 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
               torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float64)
        tree_indices = np.zeros(batch_size, dtype=np.int32)
        
        segment = self.tree.total_priority / batch_size
        
        segment_starts = np.arange(batch_size) * segment
        segment_ends = segment_starts + segment
        values = np.random.uniform(segment_starts, segment_ends)
        
        for i in range(batch_size):
            tree_idx, priority, data_idx = self.tree.get(values[i])
            tree_indices[i] = tree_idx
            priorities[i] = max(priority, 1e-8)
            indices[i] = data_idx % self.capacity
        
        probabilities = priorities / max(self.tree.total_priority, 1e-8)
        weights = (self.size * probabilities) ** (-self.beta)
        weights = weights / (weights.max() + 1e-8)
        
        self.frame += 1
        
        states_batch = torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device)
        actions_batch = torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device)
        rewards_batch = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        next_states_batch = torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=self.device)
        dones_batch = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        weights_batch = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        return (
            states_batch,
            actions_batch,
            rewards_batch,
            next_states_batch,
            dones_batch,
            weights_batch,
            tree_indices
        )
    
    def update_priorities(
        self, 
        tree_indices: np.ndarray, 
        td_errors: np.ndarray
    ) -> None:
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        self.tree.update_batch(tree_indices, priorities)
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size


class NStepReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        n_step: int = 3,
        gamma: float = 0.99,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None
    ):
        self.capacity = capacity
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        
        self.n_step_buffer = deque(maxlen=n_step)
        
        self.main_buffer = ReplayBuffer(
            capacity, device, state_dim, action_dim
        )
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step:
            n_step_return = 0
            for i in range(self.n_step):
                _, _, r, _, d = self.n_step_buffer[i]
                n_step_return += (self.gamma ** i) * r
                if d:
                    break
            
            state_0, action_0, _, _, _ = self.n_step_buffer[0]
            _, _, _, next_state_n, done_n = self.n_step_buffer[-1]
            
            self.main_buffer.push(
                state_0, action_0, n_step_return, next_state_n, done_n
            )
        
        if done:
            while len(self.n_step_buffer) > 0:
                n_step_return = 0
                for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                    n_step_return += (self.gamma ** i) * r
                    if d:
                        break
                
                state_0, action_0, _, _, _ = self.n_step_buffer[0]
                _, _, _, next_state_n, done_n = list(self.n_step_buffer)[-1]
                
                self.main_buffer.push(
                    state_0, action_0, n_step_return, next_state_n, done_n
                )
                
                self.n_step_buffer.popleft()
    
    def sample(
        self, 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.main_buffer.sample(batch_size)
    
    def __len__(self) -> int:
        return len(self.main_buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        return self.main_buffer.is_ready(batch_size)


try:
    import drone_core as _cpp_core
    _HAS_CPP_BUFFER = hasattr(_cpp_core, 'PrioritizedReplayBuffer')
    _HAS_CPP_SEQ_BUFFER = hasattr(_cpp_core, 'SequenceReplayBuffer')
except ImportError:
    _HAS_CPP_BUFFER = False
    _HAS_CPP_SEQ_BUFFER = False


class CppPrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if _HAS_CPP_BUFFER:
            self._cpp_buffer = _cpp_core.PrioritizedReplayBuffer(
                capacity, state_dim, action_dim, alpha, beta_start, beta_frames, epsilon
            )
            self._use_cpp = True
        else:
            self._py_buffer = PrioritizedReplayBuffer(
                capacity, device, state_dim, action_dim, alpha, beta_start, beta_frames, epsilon
            )
            self._use_cpp = False
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        if self._use_cpp:
            self._cpp_buffer.push(
                state.astype(np.float32),
                action.astype(np.float32),
                float(reward),
                next_state.astype(np.float32),
                float(done)
            )
        else:
            self._py_buffer.push(state, action, reward, next_state, done)
    
    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> None:
        if self._use_cpp:
            self._cpp_buffer.push_batch(
                states.astype(np.float32),
                actions.astype(np.float32),
                rewards.astype(np.float32).flatten(),
                next_states.astype(np.float32),
                dones.astype(np.float32).flatten()
            )
        else:
            self._py_buffer.push_batch(states, actions, rewards, next_states, dones)
    
    def sample(
        self, 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
               torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        if self._use_cpp:
            states, actions, rewards, next_states, dones, weights, tree_indices = \
                self._cpp_buffer.sample(batch_size)
            return (
                torch.as_tensor(states, dtype=torch.float32, device=self.device),
                torch.as_tensor(actions, dtype=torch.float32, device=self.device),
                torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
                torch.as_tensor(next_states, dtype=torch.float32, device=self.device),
                torch.as_tensor(dones, dtype=torch.float32, device=self.device),
                torch.as_tensor(weights, dtype=torch.float32, device=self.device),
                tree_indices
            )
        else:
            return self._py_buffer.sample(batch_size)
    
    def update_priorities(
        self, 
        tree_indices: np.ndarray, 
        td_errors: np.ndarray
    ) -> None:
        if self._use_cpp:
            self._cpp_buffer.update_priorities(
                tree_indices.astype(np.int32),
                td_errors.astype(np.float64)
            )
        else:
            self._py_buffer.update_priorities(tree_indices, td_errors)
    
    def __len__(self) -> int:
        if self._use_cpp:
            return self._cpp_buffer.size()
        return len(self._py_buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        if self._use_cpp:
            return self._cpp_buffer.is_ready(batch_size)
        return self._py_buffer.is_ready(batch_size)
    
    @property
    def using_cpp(self) -> bool:
        return self._use_cpp


class CppSequenceReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        obs_dim: int,
        action_dim: int,
        sequence_length: int = 16,
        burn_in_length: int = 0
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        
        if False:
            self._cpp = _cpp_core.SequenceReplayBuffer(capacity, obs_dim, action_dim, sequence_length)
            self._use_cpp = True
        else:
            self._py = SequenceReplayBuffer(capacity, device, obs_dim, action_dim, sequence_length, burn_in_length)
            self._use_cpp = False
    
    def push(self, obs: np.ndarray, action: np.ndarray, reward: float, 
             next_obs: np.ndarray, done: bool) -> None:
        if self._use_cpp:
            self._cpp.push(obs.astype(np.float32), action.astype(np.float32), 
                          float(reward), next_obs.astype(np.float32), float(done))
        else:
            self._py.push(obs, action, reward, next_obs, done)
    
    def push_batch(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                   next_states: np.ndarray, dones: np.ndarray) -> None:
        if self._use_cpp:
            self._cpp.push_batch(states.astype(np.float32), actions.astype(np.float32),
                                rewards.astype(np.float32).flatten(), next_states.astype(np.float32),
                                dones.astype(np.float32).flatten())
        else:
            self._py.push_batch(states, actions, rewards, next_states, dones)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self._use_cpp:
            result = self._cpp.sample(batch_size)
            return {
                'obs_seq': torch.as_tensor(result['obs_seq'], dtype=torch.float32, device=self.device),
                'next_obs_seq': torch.as_tensor(result['next_obs_seq'], dtype=torch.float32, device=self.device),
                'action_seq': torch.as_tensor(result['action_seq'], dtype=torch.float32, device=self.device),
                'actions': torch.as_tensor(result['actions'], dtype=torch.float32, device=self.device),
                'rewards': torch.as_tensor(result['rewards'], dtype=torch.float32, device=self.device),
                'dones': torch.as_tensor(result['dones'], dtype=torch.float32, device=self.device),
                'masks': torch.as_tensor(result['masks'], dtype=torch.float32, device=self.device)
            }
        return self._py.sample(batch_size)
    
    def __len__(self) -> int:
        return self._cpp.size() if self._use_cpp else len(self._py)
    
    def is_ready(self, batch_size: int) -> bool:
        return self._cpp.is_ready(batch_size) if self._use_cpp else self._py.is_ready(batch_size)
    
    @property
    def using_cpp(self) -> bool:
        return self._use_cpp


class CppSequencePrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        obs_dim: int,
        action_dim: int,
        sequence_length: int = 16,
        burn_in_length: int = 0,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        
        if False:
            self._cpp = _cpp_core.SequencePrioritizedReplayBuffer(
                capacity, obs_dim, action_dim, sequence_length, alpha, beta_start, beta_frames, epsilon)
            self._use_cpp = True
        else:
            self._py = SequencePrioritizedReplayBuffer(
                capacity, device, obs_dim, action_dim, sequence_length, burn_in_length, alpha, beta_start, beta_frames, epsilon)
            self._use_cpp = False
    
    def push(self, obs: np.ndarray, action: np.ndarray, reward: float,
             next_obs: np.ndarray, done: bool) -> None:
        if self._use_cpp:
            self._cpp.push(obs.astype(np.float32), action.astype(np.float32),
                          float(reward), next_obs.astype(np.float32), float(done))
        else:
            self._py.push(obs, action, reward, next_obs, done)
    
    def push_batch(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                   next_states: np.ndarray, dones: np.ndarray) -> None:
        if self._use_cpp:
            self._cpp.push_batch(states.astype(np.float32), actions.astype(np.float32),
                                rewards.astype(np.float32).flatten(), next_states.astype(np.float32),
                                dones.astype(np.float32).flatten())
        else:
            self._py.push_batch(states, actions, rewards, next_states, dones)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        if self._use_cpp:
            result = self._cpp.sample(batch_size)
            return {
                'obs_seq': torch.as_tensor(result['obs_seq'], dtype=torch.float32, device=self.device),
                'next_obs_seq': torch.as_tensor(result['next_obs_seq'], dtype=torch.float32, device=self.device),
                'action_seq': torch.as_tensor(result['action_seq'], dtype=torch.float32, device=self.device),
                'actions': torch.as_tensor(result['actions'], dtype=torch.float32, device=self.device),
                'rewards': torch.as_tensor(result['rewards'], dtype=torch.float32, device=self.device),
                'dones': torch.as_tensor(result['dones'], dtype=torch.float32, device=self.device),
                'masks': torch.as_tensor(result['masks'], dtype=torch.float32, device=self.device),
                'weights': torch.as_tensor(result['weights'], dtype=torch.float32, device=self.device),
                'sequence_indices': result['sequence_indices'].astype(np.int64)
            }
        return self._py.sample(batch_size)
    
    def update_priorities(self, sequence_indices: np.ndarray, td_errors: np.ndarray) -> None:
        if self._use_cpp:
            self._cpp.update_priorities(sequence_indices.astype(np.uint64), td_errors.astype(np.float64))
        else:
            self._py.update_priorities(sequence_indices, td_errors)
    
    def __len__(self) -> int:
        return self._cpp.size() if self._use_cpp else len(self._py)
    
    def is_ready(self, batch_size: int) -> bool:
        return self._cpp.is_ready(batch_size) if self._use_cpp else self._py.is_ready(batch_size)
    
    @property
    def using_cpp(self) -> bool:
        return self._use_cpp


class SequenceReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        obs_dim: int,
        action_dim: int,
        sequence_length: int = 16,
        burn_in_length: int = 0
    ):
        self.capacity = capacity
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.total_sequence_length = sequence_length + burn_in_length
        
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.episode_ids = np.zeros((capacity,), dtype=np.int64)
        self.step_in_episode = np.zeros((capacity,), dtype=np.int32)
        
        self.ptr = 0
        self.size = 0
        self.current_episode_id = 0
        self._current_ep_step = 0
        self._env_ep_steps = {}
        self._env_episode_ids = {}
    
    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> None:
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.episode_ids[self.ptr] = self.current_episode_id
        self.step_in_episode[self.ptr] = self._current_ep_step
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._current_ep_step += 1
        
        if done:
            self.current_episode_id += 1
            self._current_ep_step = 0
    
    def push_episode(self, episode_data: Dict[str, np.ndarray]) -> None:
        obs_seq = episode_data['observations']
        actions_seq = episode_data['actions']
        rewards_seq = episode_data['rewards']
        next_obs_seq = episode_data['next_observations']
        dones_seq = episode_data['dones']
        
        for i in range(len(obs_seq)):
            self.push(
                obs_seq[i],
                actions_seq[i],
                rewards_seq[i],
                next_obs_seq[i],
                bool(dones_seq[i])
            )
    
    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> None:
        batch_size = states.shape[0]
        
        for i in range(batch_size):
            if i not in self._env_ep_steps:
                self._env_ep_steps[i] = 0
                self._env_episode_ids[i] = self.current_episode_id
                self.current_episode_id += 1
            
            self.observations[self.ptr] = states[i]
            self.actions[self.ptr] = actions[i]
            self.rewards[self.ptr] = float(rewards[i]) if rewards.ndim > 0 else float(rewards)
            self.next_observations[self.ptr] = next_states[i]
            self.dones[self.ptr] = float(dones[i]) if dones.ndim > 0 else float(dones)
            self.episode_ids[self.ptr] = self._env_episode_ids[i]
            self.step_in_episode[self.ptr] = self._env_ep_steps[i]
            
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            self._env_ep_steps[i] += 1
            
            is_done = bool(dones[i]) if dones.ndim > 0 else bool(dones)
            if is_done:
                self._env_episode_ids[i] = self.current_episode_id
                self.current_episode_id += 1
                self._env_ep_steps[i] = 0

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        
        obs_sequences = np.zeros((batch_size, self.total_sequence_length, self.obs_dim), dtype=np.float32)
        next_obs_sequences = np.zeros((batch_size, self.total_sequence_length, self.obs_dim), dtype=np.float32)
        action_sequences = np.zeros((batch_size, self.total_sequence_length, self.action_dim), dtype=np.float32)
        reward_sequences = np.zeros((batch_size, self.total_sequence_length), dtype=np.float32)
        done_sequences = np.zeros((batch_size, self.total_sequence_length), dtype=np.float32)
        masks = np.zeros((batch_size, self.total_sequence_length), dtype=np.float32)
        
        for b, end_idx in enumerate(indices):
            ep_id = self.episode_ids[end_idx]
            step_num = self.step_in_episode[end_idx]
            
            available_steps = step_num + 1
            seq_len_to_copy = min(available_steps, self.total_sequence_length)
            pad_len = self.total_sequence_length - seq_len_to_copy
            
            start_idx = (end_idx - seq_len_to_copy + 1) % self.capacity
            
            for t in range(seq_len_to_copy):
                src_idx = (start_idx + t) % self.capacity
                dst_idx = pad_len + t
                obs_sequences[b, dst_idx] = self.observations[src_idx]
                next_obs_sequences[b, dst_idx] = self.next_observations[src_idx]
                action_sequences[b, dst_idx] = self.actions[src_idx]
                reward_sequences[b, dst_idx] = self.rewards[src_idx]
                done_sequences[b, dst_idx] = self.dones[src_idx]
                masks[b, dst_idx] = 1.0
        
        burn_in_obs = obs_sequences[:, :self.burn_in_length, :] if self.burn_in_length > 0 else None
        burn_in_next_obs = next_obs_sequences[:, :self.burn_in_length, :] if self.burn_in_length > 0 else None
        burn_in_actions = action_sequences[:, :self.burn_in_length, :] if self.burn_in_length > 0 else None
        burn_in_masks = masks[:, :self.burn_in_length] if self.burn_in_length > 0 else None
        
        train_obs = obs_sequences[:, self.burn_in_length:, :]
        train_next_obs = next_obs_sequences[:, self.burn_in_length:, :]
        train_actions = action_sequences[:, self.burn_in_length:, :]
        train_rewards = reward_sequences[:, self.burn_in_length:]
        train_dones = done_sequences[:, self.burn_in_length:]
        train_masks = masks[:, self.burn_in_length:]
        
        result = {
            'obs_seq': torch.as_tensor(train_obs, dtype=torch.float32, device=self.device),
            'next_obs_seq': torch.as_tensor(train_next_obs, dtype=torch.float32, device=self.device),
            'action_seq': torch.as_tensor(train_actions, dtype=torch.float32, device=self.device),
            'actions': torch.as_tensor(train_actions[:, -1, :], dtype=torch.float32, device=self.device),
            'rewards': torch.as_tensor(train_rewards[:, -1], dtype=torch.float32, device=self.device).unsqueeze(-1),
            'dones': torch.as_tensor(train_dones[:, -1], dtype=torch.float32, device=self.device).unsqueeze(-1),
            'masks': torch.as_tensor(train_masks, dtype=torch.float32, device=self.device)
        }
        
        if self.burn_in_length > 0:
            result['burn_in_obs'] = torch.as_tensor(burn_in_obs, dtype=torch.float32, device=self.device)
            result['burn_in_next_obs'] = torch.as_tensor(burn_in_next_obs, dtype=torch.float32, device=self.device)
            result['burn_in_actions'] = torch.as_tensor(burn_in_actions, dtype=torch.float32, device=self.device)
            result['burn_in_masks'] = torch.as_tensor(burn_in_masks, dtype=torch.float32, device=self.device)
        
        return result
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


class SequencePrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        obs_dim: int,
        action_dim: int,
        sequence_length: int = 16,
        burn_in_length: int = 0,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6
    ):
        self.capacity = capacity
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.total_sequence_length = sequence_length + burn_in_length
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1
        
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.episode_ids = np.zeros((capacity,), dtype=np.int64)
        self.step_in_episode = np.zeros((capacity,), dtype=np.int32)
        
        self.priorities = np.zeros((capacity,), dtype=np.float64)
        self.max_priority = 1.0
        
        self.ptr = 0
        self.size = 0
        self.current_episode_id = 0
        self._current_ep_step = 0
        self._env_ep_steps = {}
        self._env_episode_ids = {}
    
    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> None:
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.episode_ids[self.ptr] = self.current_episode_id
        self.step_in_episode[self.ptr] = self._current_ep_step
        self.priorities[self.ptr] = self.max_priority ** self.alpha
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._current_ep_step += 1
        
        if done:
            self.current_episode_id += 1
            self._current_ep_step = 0
    
    def push_episode(self, episode_data: Dict[str, np.ndarray]) -> None:
        obs_seq = episode_data['observations']
        actions_seq = episode_data['actions']
        rewards_seq = episode_data['rewards']
        next_obs_seq = episode_data['next_observations']
        dones_seq = episode_data['dones']
        
        for i in range(len(obs_seq)):
            self.push(
                obs_seq[i],
                actions_seq[i],
                rewards_seq[i],
                next_obs_seq[i],
                bool(dones_seq[i])
            )
    
    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> None:
        batch_size = states.shape[0]
        
        for i in range(batch_size):
            if i not in self._env_ep_steps:
                self._env_ep_steps[i] = 0
                self._env_episode_ids[i] = self.current_episode_id
                self.current_episode_id += 1
            
            self.observations[self.ptr] = states[i]
            self.actions[self.ptr] = actions[i]
            self.rewards[self.ptr] = float(rewards[i]) if rewards.ndim > 0 else float(rewards)
            self.next_observations[self.ptr] = next_states[i]
            self.dones[self.ptr] = float(dones[i]) if dones.ndim > 0 else float(dones)
            self.episode_ids[self.ptr] = self._env_episode_ids[i]
            self.step_in_episode[self.ptr] = self._env_ep_steps[i]
            self.priorities[self.ptr] = self.max_priority ** self.alpha
            
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            self._env_ep_steps[i] += 1
            
            is_done = bool(dones[i]) if dones.ndim > 0 else bool(dones)
            if is_done:
                self._env_episode_ids[i] = self.current_episode_id
                self.current_episode_id += 1
                self._env_ep_steps[i] = 0

    def sample(self, batch_size: int) -> Dict[str, Any]:
        probs = self.priorities[:self.size] ** self.alpha
        probs_sum = probs.sum()
        if probs_sum < 1e-10:
            probs = np.ones(self.size) / self.size
        else:
            probs = probs / probs_sum
        
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=True)
        
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / (weights.max() + 1e-10)
        
        self.frame += 1
        
        obs_sequences = np.zeros((batch_size, self.total_sequence_length, self.obs_dim), dtype=np.float32)
        next_obs_sequences = np.zeros((batch_size, self.total_sequence_length, self.obs_dim), dtype=np.float32)
        action_sequences = np.zeros((batch_size, self.total_sequence_length, self.action_dim), dtype=np.float32)
        reward_sequences = np.zeros((batch_size, self.total_sequence_length), dtype=np.float32)
        done_sequences = np.zeros((batch_size, self.total_sequence_length), dtype=np.float32)
        masks = np.zeros((batch_size, self.total_sequence_length), dtype=np.float32)
        
        for b, end_idx in enumerate(indices):
            step_num = self.step_in_episode[end_idx]
            
            available_steps = step_num + 1
            seq_len_to_copy = min(available_steps, self.total_sequence_length)
            pad_len = self.total_sequence_length - seq_len_to_copy
            
            start_idx = (end_idx - seq_len_to_copy + 1) % self.capacity
            
            for t in range(seq_len_to_copy):
                src_idx = (start_idx + t) % self.capacity
                dst_idx = pad_len + t
                obs_sequences[b, dst_idx] = self.observations[src_idx]
                next_obs_sequences[b, dst_idx] = self.next_observations[src_idx]
                action_sequences[b, dst_idx] = self.actions[src_idx]
                reward_sequences[b, dst_idx] = self.rewards[src_idx]
                done_sequences[b, dst_idx] = self.dones[src_idx]
                masks[b, dst_idx] = 1.0
        
        burn_in_obs = obs_sequences[:, :self.burn_in_length, :] if self.burn_in_length > 0 else None
        burn_in_next_obs = next_obs_sequences[:, :self.burn_in_length, :] if self.burn_in_length > 0 else None
        burn_in_actions = action_sequences[:, :self.burn_in_length, :] if self.burn_in_length > 0 else None
        burn_in_masks = masks[:, :self.burn_in_length] if self.burn_in_length > 0 else None
        
        train_obs = obs_sequences[:, self.burn_in_length:, :]
        train_next_obs = next_obs_sequences[:, self.burn_in_length:, :]
        train_actions = action_sequences[:, self.burn_in_length:, :]
        train_rewards = reward_sequences[:, self.burn_in_length:]
        train_dones = done_sequences[:, self.burn_in_length:]
        train_masks = masks[:, self.burn_in_length:]
        
        result = {
            'obs_seq': torch.as_tensor(train_obs, dtype=torch.float32, device=self.device),
            'next_obs_seq': torch.as_tensor(train_next_obs, dtype=torch.float32, device=self.device),
            'action_seq': torch.as_tensor(train_actions, dtype=torch.float32, device=self.device),
            'actions': torch.as_tensor(train_actions[:, -1, :], dtype=torch.float32, device=self.device),
            'rewards': torch.as_tensor(train_rewards[:, -1], dtype=torch.float32, device=self.device).unsqueeze(-1),
            'dones': torch.as_tensor(train_dones[:, -1], dtype=torch.float32, device=self.device).unsqueeze(-1),
            'masks': torch.as_tensor(train_masks, dtype=torch.float32, device=self.device),
            'weights': torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(-1),
            'sequence_indices': indices
        }
        
        if self.burn_in_length > 0:
            result['burn_in_obs'] = torch.as_tensor(burn_in_obs, dtype=torch.float32, device=self.device)
            result['burn_in_next_obs'] = torch.as_tensor(burn_in_next_obs, dtype=torch.float32, device=self.device)
            result['burn_in_actions'] = torch.as_tensor(burn_in_actions, dtype=torch.float32, device=self.device)
            result['burn_in_masks'] = torch.as_tensor(burn_in_masks, dtype=torch.float32, device=self.device)
        
        return result
    
    def update_priorities(
        self,
        sequence_indices: np.ndarray,
        td_errors: np.ndarray
    ) -> None:
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        self.priorities[sequence_indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size