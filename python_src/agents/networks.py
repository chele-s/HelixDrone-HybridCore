import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def orthogonal_init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:

    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


class Actor(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256,
        max_action: float = 1.0
    ):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        orthogonal_init(self.fc1, gain=np.sqrt(2))
        orthogonal_init(self.fc2, gain=np.sqrt(2))
        orthogonal_init(self.fc3, gain=np.sqrt(2))
        orthogonal_init(self.fc_out, gain=0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        
        return torch.tanh(self.fc_out(x)) * self.max_action


class Critic(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256
    ):
        super(Critic, self).__init__()
        
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_ln1 = nn.LayerNorm(hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_ln2 = nn.LayerNorm(hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_ln3 = nn.LayerNorm(hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_ln1 = nn.LayerNorm(hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_ln2 = nn.LayerNorm(hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_ln3 = nn.LayerNorm(hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.q1_fc1, self.q1_fc2, self.q1_fc3,
                       self.q2_fc1, self.q2_fc2, self.q2_fc3]:
            orthogonal_init(module, gain=np.sqrt(2))
        
        orthogonal_init(self.q1_out, gain=1.0)
        orthogonal_init(self.q2_out, gain=1.0)
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        
        q1 = F.relu(self.q1_ln1(self.q1_fc1(sa)))
        q1 = F.relu(self.q1_ln2(self.q1_fc2(q1)))
        q1 = F.relu(self.q1_ln3(self.q1_fc3(q1)))
        q1 = self.q1_out(q1)
        
        q2 = F.relu(self.q2_ln1(self.q2_fc1(sa)))
        q2 = F.relu(self.q2_ln2(self.q2_fc2(q2)))
        q2 = F.relu(self.q2_ln3(self.q2_fc3(q2)))
        q2 = self.q2_out(q2)
        
        return q1, q2
    
    def q1_forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        
        q1 = F.relu(self.q1_ln1(self.q1_fc1(sa)))
        q1 = F.relu(self.q1_ln2(self.q1_fc2(q1)))
        q1 = F.relu(self.q1_ln3(self.q1_fc3(q1)))
        
        return self.q1_out(q1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super(ResidualBlock, self).__init__()
        
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        return F.relu(x + residual)


class DeepActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_residual_blocks: int = 2,
        max_action: float = 1.0
    ):
        super(DeepActor, self).__init__()
        
        self.max_action = max_action
        
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)]
        )
        
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
        orthogonal_init(self.input_layer[0])
        orthogonal_init(self.output_layer, gain=0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(state)
        x = self.residual_blocks(x)
        return torch.tanh(self.output_layer(x)) * self.max_action


class NoisyLinear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        sigma_init: float = 0.5
    ):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.sigma_init = sigma_init
        self._reset_parameters()
        self._reset_noise()
    
    def _reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def _reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Tuple[int, ...] = (256, 256),
    activation: nn.Module = nn.ReLU,
    use_layer_norm: bool = True,
    output_activation: Optional[nn.Module] = None
) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    
    if output_activation is not None:
        layers.append(output_activation())
    
    net = nn.Sequential(*layers)
    
    for i, layer in enumerate(net):
        if isinstance(layer, nn.Linear):
            if i == len(net) - 1 or (i == len(net) - 2 and output_activation):
                orthogonal_init(layer, gain=0.01)
            else:
                orthogonal_init(layer)
    
    return net


class LSTMActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        num_layers: int = 2,
        max_action: float = 1.0,
        dropout: float = 0.0
    ):
        super(LSTMActor, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        self.max_action = max_action
        
        self.input_proj = nn.Linear(obs_dim, lstm_hidden)
        self.input_ln = nn.LayerNorm(lstm_hidden)
        
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.post_lstm_ln = nn.LayerNorm(lstm_hidden)
        
        self.fc1 = nn.Linear(lstm_hidden, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        orthogonal_init(self.input_proj, gain=np.sqrt(2))
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
        
        orthogonal_init(self.fc1, gain=np.sqrt(2))
        orthogonal_init(self.fc2, gain=np.sqrt(2))
        orthogonal_init(self.fc_out, gain=0.01)
    
    @torch.compiler.disable
    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = obs_seq.size(0)
        
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, obs_seq.device)
        
        self.lstm.flatten_parameters()
        x = self.input_proj(obs_seq)
        x = self.input_ln(x)
        x = F.relu(x)
        
        lstm_out, new_hidden = self.lstm(x, hidden)
        
        if lengths is not None:
            idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, self.lstm_hidden)
            x = torch.gather(lstm_out, dim=1, index=idx).squeeze(1)
        else:
            x = lstm_out[:, -1, :]
        
        x = self.post_lstm_ln(x)
        
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        action = torch.tanh(self.fc_out(x)) * self.max_action
        
        return action, new_hidden
    
    def get_initial_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        
        h = torch.zeros(self.num_layers, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.lstm_hidden, device=device)
        return (h, c)


class LSTMCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        super(LSTMCritic, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(obs_dim, lstm_hidden)
        self.input_ln = nn.LayerNorm(lstm_hidden)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.post_lstm_ln = nn.LayerNorm(lstm_hidden)
        
        self.q1_fc1 = nn.Linear(lstm_hidden + action_dim, hidden_dim)
        self.q1_ln1 = nn.LayerNorm(hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_ln2 = nn.LayerNorm(hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        self.q2_fc1 = nn.Linear(lstm_hidden + action_dim, hidden_dim)
        self.q2_ln1 = nn.LayerNorm(hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_ln2 = nn.LayerNorm(hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        orthogonal_init(self.input_proj, gain=np.sqrt(2))
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
        
        for module in [self.q1_fc1, self.q1_fc2, self.q2_fc1, self.q2_fc2]:
            orthogonal_init(module, gain=np.sqrt(2))
        
        orthogonal_init(self.q1_out, gain=1.0)
        orthogonal_init(self.q2_out, gain=1.0)
    
    def _process_lstm(
        self,
        obs_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]],
        lengths: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = obs_seq.size(0)
        
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, obs_seq.device)
        
        self.lstm.flatten_parameters()
        
        x = self.input_proj(obs_seq)
        x = self.input_ln(x)
        x = F.relu(x)
        
        lstm_out, new_hidden = self.lstm(x, hidden)
        
        if lengths is not None:
            idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, self.lstm_hidden)
            x = torch.gather(lstm_out, dim=1, index=idx).squeeze(1)
        else:
            x = lstm_out[:, -1, :]
        
        x = self.post_lstm_ln(x)
        return x, new_hidden
    
    @torch.compiler.disable
    def forward(
        self,
        obs_seq: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, new_hidden = self._process_lstm(obs_seq, hidden, lengths)
        
        sa = torch.cat([x, action], dim=-1)
        
        q1 = F.relu(self.q1_ln1(self.q1_fc1(sa)))
        q1 = F.relu(self.q1_ln2(self.q1_fc2(q1)))
        q1 = self.q1_out(q1)
        
        q2 = F.relu(self.q2_ln1(self.q2_fc1(sa)))
        q2 = F.relu(self.q2_ln2(self.q2_fc2(q2)))
        q2 = self.q2_out(q2)
        
        return q1, q2, new_hidden
    
    @torch.compiler.disable
    def q1_forward(
        self,
        obs_seq: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, new_hidden = self._process_lstm(obs_seq, hidden, lengths)
        
        x = torch.cat([x, action], dim=-1)
        x = F.relu(self.q1_ln1(self.q1_fc1(x)))
        x = F.relu(self.q1_ln2(self.q1_fc2(x)))
        q1 = self.q1_out(x)
        
        return q1, new_hidden
    
    def get_initial_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        
        h = torch.zeros(self.num_layers, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.lstm_hidden, device=device)
        
        return (h, c)