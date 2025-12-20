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
    
    def add(self, priority: float) -> int:
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        
        data_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        return data_idx
    
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
        
        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update(tree_idx, priority)
        
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