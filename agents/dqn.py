"""Vanilla DQN agent (Mnih et al., 2015) and shared agent machinery.

:class:`DQNAgent` implements the full training loop plumbing shared by all
variants: epsilon-greedy action selection with a linear schedule, a uniform
replay buffer, n-step return accumulation, target-network syncing, gradient
clipping, and checkpointing.  The vanilla TD target is

    y = r_n + gamma^n * max_a' Q_target(s', a')

Subclasses override :meth:`_next_state_values` (Double DQN) and/or the
buffer/loss (PER).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from memory.replay_buffer import ReplayBuffer
from models.dqn_network import build_network
from utils.schedules import LinearSchedule

Obs = Dict[str, np.ndarray]


class DQNAgent:
    """Vanilla DQN with uniform replay and optional n-step returns.

    Hyperparameter defaults follow the README training-pipeline table:
    buffer 50,000; batch 64; gamma 0.99; lr 1e-4; target sync every 1,000
    gradient steps; epsilon 1.0 -> 0.05 over 50k env steps; Adam; grad-clip
    norm 10.0.
    """

    name = "vanilla_dqn"

    def __init__(
        self,
        window_size: int = 7,
        state_dim: int = 5,
        n_actions: int = 4,
        dueling: bool = False,
        n_step: int = 3,
        gamma: float = 0.99,
        lr: float = 1e-4,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50_000,
        learn_start: int = 1_000,
        grad_clip: float = 10.0,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.window_size = window_size
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.dueling = bool(dueling)
        self.n_step = max(int(n_step), 1)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_start = max(learn_start, batch_size)
        self.grad_clip = grad_clip

        self.py_rng = random.Random(seed)
        self.online_net = build_network(window_size, state_dim, n_actions, dueling).to(
            self.device
        )
        self.target_net = build_network(window_size, state_dim, n_actions, dueling).to(
            self.device
        )
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        self.eps_schedule = LinearSchedule(eps_start, eps_end, eps_decay_steps)
        self.buffer = self._make_buffer(buffer_size, seed)

        # n-step accumulator: (obs, action, reward) tuples of the last n steps
        self._nstep_queue: Deque[Tuple[Obs, int, float]] = deque(maxlen=self.n_step)

        self.env_steps = 0  # environment transitions observed
        self.grad_steps = 0  # gradient updates performed

    # -------------------------------------------------------------- buffers

    def _make_buffer(self, buffer_size: int, seed: Optional[int]):
        return ReplayBuffer(buffer_size, seed=seed)

    # -------------------------------------------------------------- acting

    @property
    def epsilon(self) -> float:
        return self.eps_schedule.value(self.env_steps)

    def select_action(self, obs: Obs, greedy: bool = False) -> int:
        """Epsilon-greedy over online-network Q-values."""
        if not greedy and self.py_rng.random() < self.epsilon:
            return self.py_rng.randrange(self.n_actions)
        with torch.no_grad():
            grid = torch.as_tensor(
                obs["grid"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            vec = torch.as_tensor(
                obs["vec"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q = self.online_net(grid, vec)
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------ observing

    def observe(
        self,
        obs: Obs,
        action: int,
        reward: float,
        next_obs: Obs,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record a transition, folding it through the n-step accumulator.

        Once ``n`` rewards have accumulated, the oldest (obs, action) pair
        is inserted with the discounted n-step return, ``next_obs`` as the
        bootstrap state, and ``discount = gamma ** n``.  At episode end the
        remaining partial returns are flushed with their true horizon ``k``
        (``discount = gamma ** k``); ``done`` reflects *termination* only,
        so truncated episodes still bootstrap.
        """
        self.env_steps += 1
        self._nstep_queue.append((obs, int(action), float(reward)))

        if len(self._nstep_queue) == self.n_step:
            self._emit(next_obs, terminated, list(self._nstep_queue))
            if not (terminated or truncated):
                # sliding window: drop the oldest, keep accumulating
                self._nstep_queue.popleft()

        if terminated or truncated:
            # flush the remaining partial n-step transitions
            while self._nstep_queue:
                if len(self._nstep_queue) < self.n_step:
                    self._emit(next_obs, terminated, list(self._nstep_queue))
                self._nstep_queue.popleft()

    def _emit(self, bootstrap_obs: Obs, terminated: bool, window) -> None:
        obs0, action0, _ = window[0]
        ret = 0.0
        for k, (_, _, r) in enumerate(window):
            ret += (self.gamma ** k) * r
        self.buffer.push(
            obs0["grid"],
            obs0["vec"],
            action0,
            ret,
            bootstrap_obs["grid"],
            bootstrap_obs["vec"],
            bool(terminated),
            self.gamma ** len(window),
        )

    # ------------------------------------------------------------- learning

    def learn(self) -> Optional[float]:
        """One gradient step; returns the loss (or None before warm-up)."""
        if len(self.buffer) < self.learn_start:
            return None
        batch = self.buffer.sample(self.batch_size)
        tensors = self._to_tensors(batch)
        q_values = self._q_selected(tensors)
        targets = self._targets(tensors)
        loss = self._loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return float(loss.item())

    def _loss(self, q_values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Vanilla DQN loss: mean squared TD error (README spec)."""
        return F.mse_loss(q_values, targets)

    def _q_selected(self, t: Dict[str, torch.Tensor]) -> torch.Tensor:
        q = self.online_net(t["grids"], t["vecs"])
        return q.gather(1, t["actions"].unsqueeze(1)).squeeze(1)

    def _targets(self, t: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            next_v = self._next_state_values(t)
            return t["rewards"] + t["discounts"] * next_v * (1.0 - t["dones"])

    def _next_state_values(self, t: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Vanilla target: max_a' Q_target(s', a')."""
        return self.target_net(t["next_grids"], t["next_vecs"]).max(dim=1).values

    def _to_tensors(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}

    # --------------------------------------------------------- checkpointing

    def config(self) -> dict:
        """Everything evaluate.py needs to rebuild the network."""
        return {
            "agent": self.name,
            "window_size": self.window_size,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "dueling": self.dueling,
            "n_step": self.n_step,
        }

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {"state_dict": self.online_net.state_dict(), "config": self.config()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(payload["state_dict"])
        self.target_net.load_state_dict(payload["state_dict"])
