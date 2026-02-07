import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import copy

from .networks import Actor, Critic, DeepActor, LSTMActor, LSTMCritic
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, CppPrioritizedReplayBuffer, SequenceReplayBuffer, SequencePrioritizedReplayBuffer


class OUNoise:
    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 0.02
    ):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        
        self.state = np.ones(action_dim) * mu
        self.reset()
    
    def reset(self) -> None:
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        dx = (
            self.theta * (self.mu - self.state) * self.dt +
            self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        )
        self.state = self.state + dx
        return self.state.astype(np.float32)
    
    def __call__(self) -> np.ndarray:
        return self.sample()


class GaussianNoise:
    def __init__(
        self,
        action_dim: int,
        sigma: float = 0.1,
        sigma_min: float = 0.01,
        decay: float = 0.9999
    ):
        self.action_dim = action_dim
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.decay = decay
    
    def sample(self) -> np.ndarray:
        noise = np.random.randn(self.action_dim) * self.sigma
        self.sigma = max(self.sigma_min, self.sigma * self.decay)
        return noise.astype(np.float32)
    
    def reset(self) -> None:
        pass
    
    def __call__(self) -> np.ndarray:
        return self.sample()


class DDPGAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        max_action: float = 1.0,
        gradient_clip: float = 1.0
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.gradient_clip = gradient_clip
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        self.noise = OUNoise(action_dim)
        
        self._train_steps = 0
        self._actor_loss = 0.0
        self._critic_loss = 0.0
    
    def get_action(
        self, 
        state: np.ndarray, 
        add_noise: bool = True,
        noise_scale: float = 1.0
    ) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if add_noise and noise_scale > 0:
            noise = self.noise.sample() * noise_scale
            action = action + noise
        
        return np.clip(action, -self.max_action, self.max_action)
    
    def get_actions_batch(
        self, 
        states: np.ndarray, 
        add_noise: bool = True,
        noise_scale: float = 1.0
    ) -> np.ndarray:
        with torch.no_grad():
            states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions = self.actor(states_tensor).cpu().numpy()
        
        if add_noise and noise_scale > 0:
            noise = np.random.randn(*actions.shape) * noise_scale
            actions = actions + noise
        
        return np.clip(actions, -self.max_action, self.max_action)
    
    def update(
        self, 
        replay_buffer: Union[ReplayBuffer, PrioritizedReplayBuffer],
        batch_size: int = 256
    ) -> Dict[str, float]:
        if not replay_buffer.is_ready(batch_size):
            return {}
        
        is_per = isinstance(replay_buffer, (PrioritizedReplayBuffer, CppPrioritizedReplayBuffer))
        
        if is_per:
            states, actions, rewards, next_states, dones, weights, indices = \
                replay_buffer.sample(batch_size)
        else:
            states, actions, rewards, next_states, dones = \
                replay_buffer.sample(batch_size)
            weights = torch.ones(batch_size, 1).to(self.device)
        
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1, current_q2 = self.critic(states, actions)
        
        td_errors1 = target_q - current_q1
        td_errors2 = target_q - current_q2
        
        critic_loss = (weights * (td_errors1 ** 2)).mean() + \
                      (weights * (td_errors2 ** 2)).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        
        actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        if is_per:
            td_errors = (td_errors1.abs() + td_errors2.abs()) / 2
            replay_buffer.update_priorities(
                indices, 
                td_errors.detach().cpu().numpy().flatten()
            )
        
        self._train_steps += 1
        self._actor_loss = actor_loss.item()
        self._critic_loss = critic_loss.item()
        
        return {
            'actor_loss': self._actor_loss,
            'critic_loss': self._critic_loss,
            'q_value': current_q1.mean().item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(
                self.tau * src_param.data + (1.0 - self.tau) * tgt_param.data
            )
    
    def reset_noise(self) -> None:
        self.noise.reset()
    
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'train_steps': self._train_steps
        }, path / 'ddpg_checkpoint.pt')
    
    def load(self, path: Union[str, Path]) -> None:
        checkpoint = torch.load(Path(path) / 'ddpg_checkpoint.pt', map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self._train_steps = checkpoint['train_steps']


class TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        max_action: float = 1.0,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        gradient_clip: float = 1.0
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.gradient_clip = gradient_clip
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        self.exploration_noise = GaussianNoise(action_dim, sigma=0.1)
        
        self._train_steps = 0
        self._actor_loss = 0.0
        self._critic_loss = 0.0
    
    def get_action(
        self, 
        state: np.ndarray, 
        add_noise: bool = True
    ) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if add_noise:
            noise = self.exploration_noise.sample()
            action = action + noise
        
        return np.clip(action, -self.max_action, self.max_action)
    
    def get_actions_batch(
        self, 
        states: np.ndarray, 
        add_noise: bool = True,
        noise_scale: float = 1.0
    ) -> np.ndarray:
        with torch.no_grad():
            states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions = self.actor(states_tensor).cpu().numpy()
        
        if add_noise and noise_scale > 0:
            noise = np.random.randn(*actions.shape) * noise_scale
            actions = actions + noise
        
        return np.clip(actions, -self.max_action, self.max_action)
    
    def update(
        self, 
        replay_buffer: Union[ReplayBuffer, PrioritizedReplayBuffer],
        batch_size: int = 256
    ) -> Dict[str, float]:
        if not replay_buffer.is_ready(batch_size):
            return {}
        
        self._train_steps += 1
        
        is_per = isinstance(replay_buffer, (PrioritizedReplayBuffer, CppPrioritizedReplayBuffer))
        
        if is_per:
            states, actions, rewards, next_states, dones, weights, indices = \
                replay_buffer.sample(batch_size)
        else:
            states, actions, rewards, next_states, dones = \
                replay_buffer.sample(batch_size)
            weights = torch.ones(batch_size, 1).to(self.device)
        
        with torch.no_grad():
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = (
                self.actor_target(next_states) + noise
            ).clamp(-self.max_action, self.max_action)
            
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1, current_q2 = self.critic(states, actions)
        
        td_errors1 = target_q - current_q1
        td_errors2 = target_q - current_q2
        
        critic_loss = (weights * (td_errors1 ** 2)).mean() + \
                      (weights * (td_errors2 ** 2)).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        
        self._critic_loss = critic_loss.item()
        
        if is_per:
            td_errors = (td_errors1.abs() + td_errors2.abs()) / 2
            replay_buffer.update_priorities(
                indices, 
                td_errors.detach().cpu().numpy().flatten()
            )
        
        metrics = {
            'critic_loss': self._critic_loss,
            'q_value': current_q1.mean().item()
        }
        
        if self._train_steps % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
            self.actor_optimizer.step()
            
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            
            self._actor_loss = actor_loss.item()
            metrics['actor_loss'] = self._actor_loss
        
        return metrics
    
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(
                self.tau * src_param.data + (1.0 - self.tau) * tgt_param.data
            )
    
    def reset_noise(self) -> None:
        self.exploration_noise.reset()
    
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'train_steps': self._train_steps
        }, path / 'td3_checkpoint.pt')
    
    def load(self, path: Union[str, Path]) -> None:
        checkpoint = torch.load(Path(path) / 'td3_checkpoint.pt', map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self._train_steps = checkpoint['train_steps']
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'tau': self.tau,
            'max_action': self.max_action,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip,
            'policy_delay': self.policy_delay,
            'gradient_clip': self.gradient_clip
        }


class TD3LSTMAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        sequence_length: int = 16,
        burn_in_length: int = 0,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        max_action: float = 1.0,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        gradient_clip: float = 1.0,
        use_amp: bool = True,
        compile_networks: bool = True
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp and device.type == 'cuda'
        
        self.actor = LSTMActor(
            obs_dim, action_dim, hidden_dim, lstm_hidden, lstm_layers, max_action
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = LSTMCritic(
            obs_dim, action_dim, hidden_dim, lstm_hidden, lstm_layers
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        import sys as _sys
        _is_linux_cuda = device.type == 'cuda' and not _sys.platform.startswith('win')
        if compile_networks and _is_linux_cuda and hasattr(torch, 'compile'):
            try:
                self.actor = torch.compile(self.actor, mode='reduce-overhead')
                self.actor_target = torch.compile(self.actor_target, mode='reduce-overhead')
                self.critic = torch.compile(self.critic, mode='reduce-overhead')
                self.critic_target = torch.compile(self.critic_target, mode='reduce-overhead')
            except Exception:
                pass
        
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        self.exploration_noise = GaussianNoise(action_dim, sigma=0.1)
        
        self._num_envs = 1
        self._obs_buffers = None
        self._obs_buffer_ptrs = None
        self._obs_buffers_full = None
        self._actor_hidden = None
        
        self._train_steps = 0
        self._actor_loss = 0.0
        self._critic_loss = 0.0
        
        self.use_lstm = True
    
    def init_vectorized(self, num_envs: int) -> None:
        self._num_envs = num_envs
        self._obs_buffers = np.zeros((num_envs, self.sequence_length, self.obs_dim), dtype=np.float32)
        self._obs_buffer_ptrs = np.zeros(num_envs, dtype=np.int32)
        self._obs_buffers_full = np.zeros(num_envs, dtype=bool)
        self._actor_hidden = None
    
    def reset_hidden_states(self, env_indices: Optional[Union[int, np.ndarray]] = None) -> None:
        if self._num_envs == 1 or env_indices is None:
            self._obs_buffers = np.zeros((self._num_envs, self.sequence_length, self.obs_dim), dtype=np.float32) if self._num_envs > 1 else None
            self._obs_buffer_ptrs = np.zeros(self._num_envs, dtype=np.int32) if self._num_envs > 1 else None
            self._obs_buffers_full = np.zeros(self._num_envs, dtype=bool) if self._num_envs > 1 else None
            self._actor_hidden = None
            self._single_obs_buffer = np.zeros((self.sequence_length, self.obs_dim), dtype=np.float32)
            self._single_obs_ptr = 0
            self._single_obs_full = False
        else:
            if isinstance(env_indices, int):
                env_indices = [env_indices]
            for idx in env_indices:
                self._obs_buffers[idx] = 0.0
                self._obs_buffer_ptrs[idx] = 0
                self._obs_buffers_full[idx] = False
            if self._actor_hidden is not None:
                h, c = self._actor_hidden
                for idx in env_indices:
                    h[:, idx, :] = 0.0
                    c[:, idx, :] = 0.0
    
    def _update_obs_buffer_single(self, obs: np.ndarray) -> Tuple[np.ndarray, int]:
        if not hasattr(self, '_single_obs_buffer'):
            self._single_obs_buffer = np.zeros((self.sequence_length, self.obs_dim), dtype=np.float32)
            self._single_obs_ptr = 0
            self._single_obs_full = False
        
        self._single_obs_buffer[self._single_obs_ptr] = obs
        self._single_obs_ptr = (self._single_obs_ptr + 1) % self.sequence_length
        
        if self._single_obs_ptr == 0:
            self._single_obs_full = True
        
        if self._single_obs_full:
            indices = [(self._single_obs_ptr + i) % self.sequence_length for i in range(self.sequence_length)]
            return self._single_obs_buffer[indices], self.sequence_length
        else:
            result = np.zeros((self.sequence_length, self.obs_dim), dtype=np.float32)
            valid_count = self._single_obs_ptr
            result[:valid_count] = self._single_obs_buffer[:valid_count]
            return result, valid_count
    
    def _update_obs_buffers_batch(self, obs_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = obs_batch.shape[0]
        
        ptrs = self._obs_buffer_ptrs
        self._obs_buffers[np.arange(batch_size), ptrs] = obs_batch
        
        new_ptrs = (ptrs + 1) % self.sequence_length
        self._obs_buffers_full |= (new_ptrs == 0)
        self._obs_buffer_ptrs = new_ptrs
        
        sequences = np.zeros((batch_size, self.sequence_length, self.obs_dim), dtype=np.float32)
        lengths = np.zeros(batch_size, dtype=np.int64)
        
        for i in range(batch_size):
            if self._obs_buffers_full[i]:
                start = new_ptrs[i]
                indices = [(start + t) % self.sequence_length for t in range(self.sequence_length)]
                sequences[i] = self._obs_buffers[i, indices]
                lengths[i] = self.sequence_length
            else:
                valid_count = new_ptrs[i]
                sequences[i, :valid_count] = self._obs_buffers[i, :valid_count]
                lengths[i] = max(valid_count, 1)
        
        return sequences, lengths
    
    def get_action(
        self,
        obs: np.ndarray,
        add_noise: bool = True,
        reset_hidden: bool = False
    ) -> np.ndarray:
        if reset_hidden or self._actor_hidden is None:
            self._actor_hidden = self.actor.get_initial_hidden(1, self.device)
        
        obs_seq, length = self._update_obs_buffer_single(obs)
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
            obs_tensor = torch.as_tensor(
                obs_seq, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            lengths_tensor = torch.tensor([length], dtype=torch.int64, device=self.device)
            
            action, new_hidden = self.actor(obs_tensor, self._actor_hidden, lengths_tensor)
            self._actor_hidden = (new_hidden[0].detach(), new_hidden[1].detach())
            action = action.cpu().numpy()[0]
        
        if add_noise:
            noise = self.exploration_noise.sample()
            action = action + noise
        
        return np.clip(action, -self.max_action, self.max_action)
    
    def get_actions_batch(
        self,
        obs_batch: np.ndarray,
        add_noise: bool = True,
        noise_scale: float = 1.0
    ) -> np.ndarray:
        batch_size = obs_batch.shape[0]
        if self._obs_buffers is None or self._obs_buffers.shape[0] != batch_size:
            self.init_vectorized(batch_size)
        
        if self._actor_hidden is None or self._actor_hidden[0].size(1) != batch_size:
            self._actor_hidden = self.actor.get_initial_hidden(batch_size, self.device)
        
        obs_sequences, lengths = self._update_obs_buffers_batch(obs_batch)
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
            obs_tensor = torch.as_tensor(
                obs_sequences, dtype=torch.float32, device=self.device
            )
            lengths_tensor = torch.as_tensor(lengths, dtype=torch.int64, device=self.device)
            actions, new_hidden = self.actor(obs_tensor, self._actor_hidden, lengths_tensor)
            self._actor_hidden = (new_hidden[0].detach(), new_hidden[1].detach())
            actions = actions.cpu().numpy()
        
        if add_noise and noise_scale > 0:
            noise = np.random.randn(*actions.shape) * noise_scale
            actions = actions + noise
        
        return np.clip(actions, -self.max_action, self.max_action)
    
    def update(
        self,
        replay_buffer: Union[SequenceReplayBuffer, SequencePrioritizedReplayBuffer],
        batch_size: int = 256
    ) -> Dict[str, float]:
        if not replay_buffer.is_ready(batch_size):
            return {}
        
        self._train_steps += 1
        
        batch = replay_buffer.sample(batch_size)
        
        obs_seq = batch['obs_seq']
        next_obs_seq = batch['next_obs_seq']
        rewards = batch['rewards']
        dones = batch['dones']
        actions = batch['actions']
        lengths = batch['lengths']
        next_lengths = batch['next_lengths']
        
        is_per = isinstance(replay_buffer, SequencePrioritizedReplayBuffer)
        if is_per:
            weights = batch['weights']
            sequence_indices = batch['sequence_indices']
        else:
            weights = torch.ones(batch_size, 1, device=self.device)
        
        actor_burn_in_hidden = None
        actor_burn_in_hidden_next = None
        critic_burn_in_hidden = None
        critic_burn_in_hidden_next = None
        burn_in_lengths = None
        
        if self.burn_in_length > 0 and 'burn_in_obs' in batch:
            burn_in_obs = batch['burn_in_obs']
            burn_in_next_obs = batch['burn_in_next_obs']
            burn_in_actions = batch['burn_in_actions']
            burn_in_lengths = batch.get('burn_in_lengths', None)
            dummy_action = burn_in_actions[:, -1, :]
            
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
                _, actor_burn_in_hidden = self.actor(burn_in_obs, None, burn_in_lengths)
                _, actor_burn_in_hidden_next = self.actor_target(burn_in_next_obs, None, burn_in_lengths)
                
                _, _, critic_burn_in_hidden = self.critic(burn_in_obs, dummy_action, None, burn_in_lengths)
                _, _, critic_burn_in_hidden_next = self.critic_target(burn_in_next_obs, dummy_action, None, burn_in_lengths)
                
                actor_burn_in_hidden = (actor_burn_in_hidden[0].detach(), actor_burn_in_hidden[1].detach())
                actor_burn_in_hidden_next = (actor_burn_in_hidden_next[0].detach(), actor_burn_in_hidden_next[1].detach())
                critic_burn_in_hidden = (critic_burn_in_hidden[0].detach(), critic_burn_in_hidden[1].detach())
                critic_burn_in_hidden_next = (critic_burn_in_hidden_next[0].detach(), critic_burn_in_hidden_next[1].detach())
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            with torch.no_grad():
                next_actions, _ = self.actor_target(next_obs_seq, actor_burn_in_hidden_next, next_lengths)
                
                noise = (
                    torch.randn_like(next_actions) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
                
                target_q1, target_q2, _ = self.critic_target(next_obs_seq, next_actions, critic_burn_in_hidden_next, next_lengths)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards + (1 - dones) * self.gamma * target_q
            
            current_q1, current_q2, _ = self.critic(obs_seq, actions, critic_burn_in_hidden, lengths)
            
            td_errors1 = target_q - current_q1
            td_errors2 = target_q - current_q2
            
            critic_loss = (weights * (td_errors1 ** 2)).mean() + \
                          (weights * (td_errors2 ** 2)).mean()
        
        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.scaler.step(self.critic_optimizer)
        
        self._critic_loss = critic_loss.item()
        
        if is_per:
            td_errors = (td_errors1.abs() + td_errors2.abs()) / 2
            replay_buffer.update_priorities(
                sequence_indices,
                td_errors.detach().cpu().numpy().flatten()
            )
        
        metrics = {
            'critic_loss': self._critic_loss,
            'q_value': current_q1.mean().item()
        }
        
        if self._train_steps % self.policy_delay == 0:
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                actor_actions, _ = self.actor(obs_seq, actor_burn_in_hidden, lengths)
                actor_q1, _ = self.critic.q1_forward(obs_seq, actor_actions, critic_burn_in_hidden, lengths)
                actor_loss = -actor_q1.mean()
            
            self.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
            self.scaler.step(self.actor_optimizer)
            
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            
            self._actor_loss = actor_loss.item()
            metrics['actor_loss'] = self._actor_loss
        
        self.scaler.update()
        
        return metrics
    
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(
                self.tau * src_param.data + (1.0 - self.tau) * tgt_param.data
            )
    
    def reset_noise(self) -> None:
        self.exploration_noise.reset()
    
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'train_steps': self._train_steps,
            'config': self.get_config()
        }, path / 'td3_lstm_checkpoint.pt')
    
    def load(self, path: Union[str, Path]) -> None:
        checkpoint = torch.load(Path(path) / 'td3_lstm_checkpoint.pt', map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self._train_steps = checkpoint['train_steps']
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'lstm_hidden': self.lstm_hidden,
            'lstm_layers': self.lstm_layers,
            'sequence_length': self.sequence_length,
            'burn_in_length': self.burn_in_length,
            'gamma': self.gamma,
            'tau': self.tau,
            'max_action': self.max_action,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip,
            'policy_delay': self.policy_delay,
            'gradient_clip': self.gradient_clip
        }


def create_agent(
    agent_type: str,
    state_dim: int,
    action_dim: int,
    device: torch.device,
    **kwargs
) -> Union[DDPGAgent, TD3Agent, TD3LSTMAgent]:
    if agent_type.lower() == 'ddpg':
        return DDPGAgent(state_dim, action_dim, device, **kwargs)
    elif agent_type.lower() == 'td3':
        return TD3Agent(state_dim, action_dim, device, **kwargs)
    elif agent_type.lower() == 'td3_lstm':
        return TD3LSTMAgent(state_dim, action_dim, device, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")