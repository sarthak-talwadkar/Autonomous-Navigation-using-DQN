"""Prioritized Experience Replay buffer (proportional variant).

Priorities are ``p_i = (|TD_i| + eps) ** alpha`` stored in a sum-tree; the
sampling probability is ``P(i) = p_i / sum_j p_j`` (stratified sampling: one
uniform draw per equal-mass segment of the total priority).  Importance
sampling weights ``w_i = (N * P(i)) ** -beta`` are normalised by the maximum
weight in the batch so that updates only ever scale gradients *down*.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from memory.replay_buffer import Transition, stack_batch
from memory.sum_tree import SumTree


class PrioritizedReplayBuffer:
    """Sum-tree backed proportional PER buffer.

    Parameters
    ----------
    capacity: maximum number of transitions.
    alpha: prioritization strength (0 = uniform, 1 = full), default 0.6.
    eps: minimum priority added to |TD| so no transition starves.
    seed: RNG seed for sampling.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        eps: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        self.tree = SumTree(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)
        self.max_priority = 1.0  # new transitions get max priority seen so far

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
        """Insert a transition with the maximum priority seen so far."""
        t = Transition(grid, vec, action, reward, next_grid, next_vec, done, discount)
        self.tree.add(self.max_priority, t)

    def sample(
        self, batch_size: int, beta: float
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """Stratified prioritized sample.

        Returns ``(tree_indices, batch, is_weights)`` where ``is_weights``
        are the max-normalised importance-sampling weights.
        """
        n = len(self.tree)
        if n < batch_size:
            raise ValueError("not enough transitions to sample")
        total = self.tree.total()
        segment = total / batch_size

        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        transitions = []
        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            for _ in range(8):  # retry guard against empty/zero leaves
                z = self.rng.uniform(lo, hi)
                idx, priority, data = self.tree.get(z)
                if isinstance(data, Transition) and priority > 0.0:
                    break
                lo, hi = 0.0, total  # fall back to sampling the whole range
            indices[i] = idx
            priorities[i] = priority
            transitions.append(data)

        probs = priorities / total
        weights = (n * probs) ** (-float(beta))
        weights = weights / weights.max()
        return indices, stack_batch(transitions), weights.astype(np.float32)

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Set leaf priorities to ``(|TD| + eps) ** alpha``."""
        td_errors = np.abs(np.asarray(td_errors, dtype=np.float64))
        for idx, td in zip(tree_indices, td_errors):
            priority = (td + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(idx), priority)

    def __len__(self) -> int:
        return len(self.tree)
