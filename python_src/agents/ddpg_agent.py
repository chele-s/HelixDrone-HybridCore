import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import copy

from .networks import Actor, Critic, DeepActor
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


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
        
        is_per = isinstance(replay_buffer, PrioritizedReplayBuffer)
        
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
        
        is_per = isinstance(replay_buffer, PrioritizedReplayBuffer)
        
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


def create_agent(
    agent_type: str,
    state_dim: int,
    action_dim: int,
    device: torch.device,
    **kwargs
) -> Union[DDPGAgent, TD3Agent]:
    if agent_type.lower() == 'ddpg':
        return DDPGAgent(state_dim, action_dim, device, **kwargs)
    elif agent_type.lower() == 'td3':
        return TD3Agent(state_dim, action_dim, device, **kwargs)
    else:
        raise ValueError(f"Tipo de agente desconocido: {agent_type}")