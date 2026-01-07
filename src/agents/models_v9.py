import torch
import torch.nn as nn
# import 
import os
import sys
# Join src to the file

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from envs.pheromone_env import PheromoneEnv


class ActorNet(nn.Module):
    def __init__(self, in_channels: int, n_actions: int = 5, base_env=PheromoneEnv()):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # agent_view assumed square; two convs keep spatial dims
        self.mlp = nn.Sequential(
            nn.Linear(64 * base_env.agent_view * base_env.agent_view + 1, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, obs: torch.Tensor, carrying: torch.Tensor):
        # Support both batched (B, A, C, H, W) and unbatched (A, C, H, W)
        if obs.dim() == 5:
            b, a, c, h, w = obs.shape
            obs_f = obs.view(b * a, c, h, w)
            carry_f = carrying.view(b * a, 1)
            logits = self.mlp(torch.cat([self.conv(obs_f), carry_f], dim=-1))
            return logits.view(b, a, -1)
        elif obs.dim() == 4:
            a, c, h, w = obs.shape
            obs_f = obs
            carry_f = carrying.view(a, 1)
            logits = self.mlp(torch.cat([self.conv(obs_f), carry_f], dim=-1))
            return logits  # shape (A, actions) to match unbatched agent dimension
        else:
            raise ValueError(f"Unexpected obs shape {obs.shape}")


class CriticNet(nn.Module):
    def __init__(self, in_channels: int, base_env=PheromoneEnv()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (base_env.grid_size // 4) * (base_env.grid_size // 4), 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)