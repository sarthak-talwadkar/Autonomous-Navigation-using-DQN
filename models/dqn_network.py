"""Q-networks: CNN occupancy-grid encoder fused with a state vector.

Two variants:

* :class:`DQNNavigator` -- the README architecture: a two-layer CNN encoder
  over the egocentric occupancy window, concatenated with the 5-d state
  vector and passed through an MLP head that outputs one Q-value per action.
* :class:`DuelingDQNNavigator` -- same encoder, but the head is split into
  separate value and advantage streams combined as
  ``Q = V + A - mean(A)`` (Wang et al., 2016).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _cnn_encoder(window_size: int, out_dim: int = 256) -> nn.Sequential:
    """Conv2d(1,32,3,p1)-ReLU-Conv2d(32,64,3,p1)-ReLU-Flatten-Linear-ReLU."""
    flat = 64 * window_size * window_size
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(flat, out_dim),
        nn.ReLU(),
    )


class DQNNavigator(nn.Module):
    """CNN + MLP Q-network over (occupancy window, state vector)."""

    def __init__(
        self, window_size: int = 7, state_dim: int = 5, n_actions: int = 4
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.cnn = _cnn_encoder(window_size)
        self.mlp = nn.Sequential(
            nn.Linear(256 + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, grid: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """``grid``: (B, W, W); ``state``: (B, state_dim) -> (B, n_actions)."""
        grid_features = self.cnn(grid.unsqueeze(1).float())
        fused = torch.cat([grid_features, state], dim=-1)
        return self.mlp(fused)


class DuelingDQNNavigator(nn.Module):
    """Dueling variant: separate value/advantage streams, Q = V + A - mean(A)."""

    def __init__(
        self, window_size: int = 7, state_dim: int = 5, n_actions: int = 4
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.cnn = _cnn_encoder(window_size)
        self.trunk = nn.Sequential(
            nn.Linear(256 + state_dim, 256),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, grid: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        grid_features = self.cnn(grid.unsqueeze(1).float())
        fused = self.trunk(torch.cat([grid_features, state], dim=-1))
        value = self.value_stream(fused)  # (B, 1)
        advantage = self.advantage_stream(fused)  # (B, n_actions)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def build_network(
    window_size: int = 7,
    state_dim: int = 5,
    n_actions: int = 4,
    dueling: bool = False,
) -> nn.Module:
    """Factory used by agents / evaluate.py to construct a Q-network."""
    cls = DuelingDQNNavigator if dueling else DQNNavigator
    return cls(window_size=window_size, state_dim=state_dim, n_actions=n_actions)
