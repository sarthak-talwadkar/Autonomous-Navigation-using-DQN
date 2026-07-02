"""Annealing schedules for exploration (epsilon) and PER correction (beta)."""

from __future__ import annotations


class LinearSchedule:
    """Linearly interpolate from ``start`` to ``end`` over ``duration`` steps.

    ``value(t)`` clamps to ``end`` for ``t >= duration``.  Used for both the
    epsilon-greedy schedule (e.g. 1.0 -> 0.05 over 50k steps) and PER beta
    annealing (0.4 -> 1.0 over training).
    """

    def __init__(self, start: float, end: float, duration: int) -> None:
        self.start = float(start)
        self.end = float(end)
        self.duration = max(int(duration), 1)

    def value(self, t: int) -> float:
        frac = min(max(t, 0) / self.duration, 1.0)
        return self.start + frac * (self.end - self.start)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LinearSchedule({self.start} -> {self.end} "
            f"over {self.duration} steps)"
        )
