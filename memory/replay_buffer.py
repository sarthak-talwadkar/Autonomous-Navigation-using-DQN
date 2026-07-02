"""Uniform experience replay buffer.

Transitions store an extra ``discount`` field so that n-step returns can be
used transparently: the agent inserts the accumulated n-step reward together
with ``discount = gamma ** k`` (the factor applied to the bootstrap term),
where ``k`` is the actual number of accumulated steps (``k == 1`` recovers
standard one-step TD).
"""

from __future__ import annotations

from collections import namedtuple
from typing import Dict, Optional

import numpy as np

Transition = namedtuple(
    "Transition",
    ["grid", "vec", "action", "reward", "next_grid", "next_vec", "done", "discount"],
)


def stack_batch(transitions) -> Dict[str, np.ndarray]:
    """Stack a sequence of :class:`Transition` into a dict of arrays."""
    return {
        "grids": np.stack([t.grid for t in transitions]).astype(np.float32),
        "vecs": np.stack([t.vec for t in transitions]).astype(np.float32),
        "actions": np.asarray([t.action for t in transitions], dtype=np.int64),
        "rewards": np.asarray([t.reward for t in transitions], dtype=np.float32),
        "next_grids": np.stack([t.next_grid for t in transitions]).astype(np.float32),
        "next_vecs": np.stack([t.next_vec for t in transitions]).astype(np.float32),
        "dones": np.asarray([t.done for t in transitions], dtype=np.float32),
        "discounts": np.asarray([t.discount for t in transitions], dtype=np.float32),
    }


class ReplayBuffer:
    """Fixed-capacity ring buffer with uniform random sampling."""

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self._storage: list = []
        self._next = 0

    def push(
        self,
        grid: np.ndarray,
        vec: np.ndarray,
        action: int,
        reward: float,
        next_grid: np.ndarray,
        next_vec: np.ndarray,
        done: bool,
        discount: float,
    ) -> None:
        """Insert one transition, overwriting the oldest when full."""
        t = Transition(grid, vec, action, reward, next_grid, next_vec, done, discount)
        if len(self._storage) < self.capacity:
            self._storage.append(t)
        else:
            self._storage[self._next] = t
        self._next = (self._next + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Uniformly sample a minibatch as a dict of stacked arrays."""
        idx = self.rng.integers(len(self._storage), size=batch_size)
        return stack_batch([self._storage[i] for i in idx])

    def __len__(self) -> int:
        return len(self._storage)
