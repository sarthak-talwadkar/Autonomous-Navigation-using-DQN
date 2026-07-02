"""Sum-tree for O(log N) prioritized sampling (Schaul et al., 2016).

A complete binary tree stored in a flat array of size ``2 * capacity - 1``.
Leaves hold individual transition priorities; every internal node holds the
sum of its children, so the root is the total priority mass.  Both priority
updates and prefix-sum retrieval are O(log N).  The traversals are iterative
(rather than the README's recursive sketch) but expose the same API.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


class SumTree:
    """Fixed-capacity sum-tree with a ring-buffer write pointer."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data = np.zeros(self.capacity, dtype=object)
        self.write = 0  # next leaf slot to overwrite
        self.n_entries = 0  # number of filled leaves

    # ------------------------------------------------------------------ core

    def total(self) -> float:
        """Total priority mass (root node)."""
        return float(self.tree[0])

    def add(self, priority: float, data: Any) -> int:
        """Insert ``data`` with ``priority``; returns its tree index."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return idx

    def update(self, idx: int, priority: float) -> None:
        """Set the priority of leaf ``idx`` and propagate the change up."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:  # iterative propagation, O(log N)
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, z: float) -> Tuple[int, float, Any]:
        """Find the leaf whose cumulative-priority interval contains ``z``.

        Returns ``(tree_index, priority, data)``.
        """
        idx = 0
        while True:  # iterative retrieval, O(log N)
            left = 2 * idx + 1
            if left >= len(self.tree):
                break
            if z <= self.tree[left]:
                idx = left
            else:
                z -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

    def __len__(self) -> int:
        return self.n_entries
