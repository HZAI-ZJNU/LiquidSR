import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as original_odeint


class TimeAttention(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()

    def forward(self, t):
        t_reshaped = t.unsqueeze(1)  # [time_steps, 1]

        scores = self.fc2(self.activation(self.fc1(t_reshaped)))  # [time_steps, 1]

        weights = F.softmax(scores.squeeze(), dim=0)  # [time_steps]
        return weights


def adaptive_odeint(func, y0, t_span, method='euler',
                    rtol=1e-7, atol=1e-9, attention=None):
    states = original_odeint(func, y0, t_span, method=method,
                             rtol=rtol, atol=atol)

    if attention is not None:
        weights = attention(t_span)
        weighted_states = states * weights.view(-1, *([1] * len(states.shape[1:])))
        return weighted_states.sum(dim=0, keepdim=True)

    return states


class DynamicNeuron(nn.Module):

    def __init__(self, channels, activation='tanh'):
        super().__init__()
        self.channels = channels

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

        self.tau = nn.Parameter(torch.Tensor(1, channels, 1, 1))

        self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU()

        nn.init.orthogonal_(self.conv.weight.squeeze(1))
        # nn.init.zeros_(self.bias)
        nn.init.uniform_(self.tau, 0.5, 2.0)

    def forward(self, t, x):
        tau = F.softplus(self.tau) + 1e-6
        dxdt = (-x + self.activation(self.conv(x))) / tau
        return dxdt

class LocalModule(nn.Sequential):
    def __init__(self, channels):
        super().__init__()
        self.add_module('pointwise_prenorm_0', nn.BatchNorm2d(channels))
        self.add_module('pointwise_conv_0', nn.Conv2d(channels, channels, kernel_size=1, bias=False))
        self.add_module('depthwise_conv',
                        nn.Conv2d(channels, channels, padding=1, kernel_size=3, groups=channels, bias=False))
        self.add_module('pointwise_prenorm_1', nn.BatchNorm2d(channels))
        self.add_module('pointwise_conv_1', nn.Conv2d(channels, channels, kernel_size=1, bias=False))

class LiquidImageLayer(nn.Module):

    def __init__(self, in_channels, time_steps=2, solver='euler', use_attention=True):
        super().__init__()
        self.in_channels = in_channels
        self.time_steps = time_steps
        self.solver = solver
        self.use_attention = use_attention

        self.dynamic_neuron = DynamicNeuron(in_channels)

        self.time_span = nn.Parameter(
            torch.linspace(0, 1.0, time_steps),
            requires_grad=False
        )

        if use_attention:
            self.time_attention = TimeAttention()
        else:
            self.time_attention = None

        self.norm = nn.LayerNorm(in_channels)
        self.ffn = FeedForward(in_channels)
        self.local = LocalModule(in_channels)

    def forward(self, x, x_size=None):

        H, W = x_size
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        local = self.local(x)
        x = local + x

        evolved_states = adaptive_odeint(
            self.dynamic_neuron,
            x,
            self.time_span.to(x.device),
            method=self.solver,

            attention=self.time_attention if self.use_attention else None
        )

        evolved_states = evolved_states[-1] if evolved_states.shape[0] > 1 else evolved_states[0]
        evolved_states = evolved_states.permute(0, 2, 3, 1)
        evolved_states = self.norm(evolved_states)
        evolved_states = evolved_states.permute(0, 3, 1, 2)
        output = evolved_states + x
        output = self.ffn(output) + output
        output = output.permute(0, 2, 3, 1).view(output.shape[0], H * W, self.in_channels)
        return output


class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super(FeedForward, self).__init__()
        hidden_features = hidden_features or in_features * 4

        self.project_in = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        # self.depthwise_conv = nn.Conv2d(hidden_features, hidden_features,
        #                                 kernel_size=3, padding=1, groups=hidden_features)
        self.act = act_layer()
        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        # x = self.depthwise_conv(x)
        x = self.act(x)
        x = self.project_out(x)
        return x
