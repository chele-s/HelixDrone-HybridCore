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
        with torch.no_grad():
            self.fc_out.bias.data[0] = 0.15
    
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
            orthogonal_init(module, gain=1.0)
        
        for module in [self.q1_out, self.q2_out]:
            nn.init.uniform_(module.weight, -0.003, 0.003)
            nn.init.zeros_(module.bias)
    
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


class TemporalConv1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        use_residual: bool = True
    ):
        super(TemporalConv1D, self).__init__()
        
        self.use_residual = use_residual
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.ln = nn.LayerNorm(out_channels)
        
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None
        
        orthogonal_init(self.conv)
        if self.residual_proj is not None:
            orthogonal_init(self.residual_proj)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv(x)
        out = out.transpose(1, 2)
        out = self.ln(out)
        out = out.transpose(1, 2)
        out = F.relu(out)
        
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            out = out + residual
        
        return out


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        frame_dim: int,
        n_frames: int,
        hidden_dim: int = 64,
        n_layers: int = 2
    ):
        super(TemporalEncoder, self).__init__()
        
        self.frame_dim = frame_dim
        self.n_frames = n_frames
        
        layers = []
        in_ch = frame_dim
        
        for i in range(n_layers):
            out_ch = hidden_dim * (2 ** i)
            dilation = 2 ** i
            layers.append(TemporalConv1D(in_ch, out_ch, kernel_size=3, dilation=dilation))
            in_ch = out_ch
        
        self.conv_layers = nn.ModuleList(layers)
        self.output_dim = in_ch
        
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        if x.dim() == 2:
            x = x.view(batch_size, self.n_frames, self.frame_dim)
        
        x = x.transpose(1, 2)
        
        for conv in self.conv_layers:
            x = conv(x)
        
        x = self.pool(x).squeeze(-1)
        
        return x


class TemporalActor(nn.Module):
    def __init__(
        self,
        frame_dim: int,
        n_frames: int,
        action_dim: int,
        hidden_dim: int = 256,
        temporal_hidden: int = 64,
        temporal_layers: int = 2,
        max_action: float = 1.0
    ):
        super(TemporalActor, self).__init__()
        
        self.max_action = max_action
        self.frame_dim = frame_dim
        self.n_frames = n_frames
        
        self.temporal_encoder = TemporalEncoder(
            frame_dim=frame_dim,
            n_frames=n_frames,
            hidden_dim=temporal_hidden,
            n_layers=temporal_layers
        )
        
        encoder_out = self.temporal_encoder.output_dim
        
        self.fc1 = nn.Linear(encoder_out, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)
        
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc_out, gain=0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.temporal_encoder(state)
        
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        
        return torch.tanh(self.fc_out(x)) * self.max_action


class TemporalCritic(nn.Module):
    def __init__(
        self,
        frame_dim: int,
        n_frames: int,
        action_dim: int,
        hidden_dim: int = 256,
        temporal_hidden: int = 64,
        temporal_layers: int = 2
    ):
        super(TemporalCritic, self).__init__()
        
        self.frame_dim = frame_dim
        self.n_frames = n_frames
        
        self.temporal_encoder_q1 = TemporalEncoder(
            frame_dim=frame_dim,
            n_frames=n_frames,
            hidden_dim=temporal_hidden,
            n_layers=temporal_layers
        )
        self.temporal_encoder_q2 = TemporalEncoder(
            frame_dim=frame_dim,
            n_frames=n_frames,
            hidden_dim=temporal_hidden,
            n_layers=temporal_layers
        )
        
        encoder_out = self.temporal_encoder_q1.output_dim
        combined_dim = encoder_out + action_dim
        
        self.q1_fc1 = nn.Linear(combined_dim, hidden_dim)
        self.q1_ln1 = nn.LayerNorm(hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_ln2 = nn.LayerNorm(hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        self.q2_fc1 = nn.Linear(combined_dim, hidden_dim)
        self.q2_ln1 = nn.LayerNorm(hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_ln2 = nn.LayerNorm(hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)
        
        for layer in [self.q1_fc1, self.q1_fc2, self.q2_fc1, self.q2_fc2]:
            orthogonal_init(layer)
        orthogonal_init(self.q1_out, gain=1.0)
        orthogonal_init(self.q2_out, gain=1.0)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s1 = self.temporal_encoder_q1(state)
        s2 = self.temporal_encoder_q2(state)
        
        sa1 = torch.cat([s1, action], dim=-1)
        sa2 = torch.cat([s2, action], dim=-1)
        
        q1 = F.relu(self.q1_ln1(self.q1_fc1(sa1)))
        q1 = F.relu(self.q1_ln2(self.q1_fc2(q1)))
        q1 = self.q1_out(q1)
        
        q2 = F.relu(self.q2_ln1(self.q2_fc1(sa2)))
        q2 = F.relu(self.q2_ln2(self.q2_fc2(q2)))
        q2 = self.q2_out(q2)
        
        return q1, q2
    
    def q1_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        s1 = self.temporal_encoder_q1(state)
        sa1 = torch.cat([s1, action], dim=-1)
        
        q1 = F.relu(self.q1_ln1(self.q1_fc1(sa1)))
        q1 = F.relu(self.q1_ln2(self.q1_fc2(q1)))
        
        return self.q1_out(q1)