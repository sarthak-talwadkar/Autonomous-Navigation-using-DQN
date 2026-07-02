"""Custom grid-world navigation environment.

A plain-Python environment (no gymnasium dependency) exposing the
gymnasium-style API:

    obs, info                                  = env.reset(seed=...)
    obs, reward, terminated, truncated, info   = env.step(action)

Observations are a dict with two entries:

* ``"grid"`` -- egocentric ``window_size x window_size`` occupancy window
  centred on the agent.  The window is **rotated with the agent's heading**
  so that "forward" always points up (row 0).  Cell values follow the README
  convention: ``0`` free, ``1`` obstacle, ``0.5`` unknown.  Cells outside the
  map bounds are encoded as ``1`` (obstacle) rather than ``0.5`` -- treating
  the world boundary as a wall is the safe choice because driving off the map
  is a collision.
* ``"vec"`` -- state vector ``[x_rel, y_rel, cos(theta), sin(theta),
  dist_to_goal]`` where the goal-relative coordinates are normalised to
  ``[-1, 1]`` and the distance is normalised by the map diagonal.

Actions (discrete):

    0 MOVE_FORWARD  advance one cell along the current heading
    1 TURN_LEFT     rotate heading by -45 degrees in place
    2 TURN_RIGHT    rotate heading by +45 degrees in place
    3 STOP          terminates with the goal reward only if the agent is
                    within ``goal_radius`` of the goal; otherwise it is a
                    wasted step (step penalty applies, episode continues)

Reward (README spec): ``+100`` goal, ``-50`` collision,
``+5.0 * (dist_before - dist_after)`` progress shaping, ``-0.1`` step
penalty.  Episodes terminate on goal, collision, or hit the step limit
(truncation).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

# Discrete action ids
MOVE_FORWARD = 0
TURN_LEFT = 1
TURN_RIGHT = 2
STOP = 3

ACTION_NAMES = {
    MOVE_FORWARD: "MOVE_FORWARD",
    TURN_LEFT: "TURN_LEFT",
    TURN_RIGHT: "TURN_RIGHT",
    STOP: "STOP",
}

# Occupancy encoding
FREE = 0.0
OBSTACLE = 1.0
UNKNOWN = 0.5  # reserved value; out-of-bounds cells are encoded as OBSTACLE

Obs = Dict[str, np.ndarray]


class GridWorldEnv:
    """Randomised grid world with an egocentric occupancy-window observation.

    Parameters
    ----------
    grid_size:
        Side length of the square map (default 20).
    window_size:
        Side length of the egocentric occupancy window (odd, default 7).
    obstacle_density:
        Probability that a cell is an obstacle when a layout is sampled.
    max_steps:
        Episode step limit (truncation).  Defaults to ``4 * grid_size``.
    goal_radius:
        Euclidean distance (in cells) at which the goal counts as reached.
        The default 0.5 means the agent must occupy the goal cell.
    seed:
        Seed for the environment's private RNG.
    """

    N_ACTIONS = 4
    N_HEADINGS = 8  # 45-degree increments
    STATE_DIM = 5

    def __init__(
        self,
        grid_size: int = 20,
        window_size: int = 7,
        obstacle_density: float = 0.15,
        max_steps: Optional[int] = None,
        goal_radius: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        self.grid_size = int(grid_size)
        self.window_size = int(window_size)
        self.half_window = window_size // 2
        self.obstacle_density = float(obstacle_density)
        self.max_steps = int(max_steps) if max_steps is not None else 4 * grid_size
        self.goal_radius = float(goal_radius)
        self.rng = np.random.default_rng(seed)

        self._diag = math.sqrt(2.0) * max(self.grid_size - 1, 1)

        # Episode state (populated by reset()).
        self.grid: Optional[np.ndarray] = None  # (grid_size, grid_size) float32
        self.pos = np.zeros(2, dtype=np.int64)  # (row, col)
        self.goal = np.zeros(2, dtype=np.int64)  # (row, col)
        self.heading = 0  # index into 0..7; angle = heading * 45 deg
        self.steps = 0

    # ------------------------------------------------------------------ API

    def reset(self, seed: Optional[int] = None) -> Tuple[Obs, dict]:
        """Sample a new obstacle layout, start, goal, and heading."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        for _ in range(500):
            grid = (
                self.rng.random((self.grid_size, self.grid_size))
                < self.obstacle_density
            ).astype(np.float32)
            free = np.argwhere(grid == FREE)
            if len(free) < 2:
                continue
            idx = self.rng.choice(len(free), size=2, replace=False)
            start, goal = free[idx[0]], free[idx[1]]
            if self._path_exists(grid, tuple(start), tuple(goal)):
                break
        else:  # pragma: no cover - astronomically unlikely for sane densities
            raise RuntimeError("could not sample a solvable layout")

        self.grid = grid
        self.pos = start.astype(np.int64)
        self.goal = goal.astype(np.int64)
        self.heading = int(self.rng.integers(self.N_HEADINGS))
        self.steps = 0
        return self._get_obs(), self._get_info(success=False, collision=False)

    def step(self, action: int) -> Tuple[Obs, float, bool, bool, dict]:
        """Apply a discrete action; returns (obs, reward, term, trunc, info)."""
        if self.grid is None:
            raise RuntimeError("call reset() before step()")
        action = int(action)
        if not 0 <= action < self.N_ACTIONS:
            raise ValueError(f"invalid action {action}")

        self.steps += 1
        terminated = False
        truncated = False
        collision = False
        success = False
        reward = -0.1  # step penalty (always applied)

        dist_before = self._dist_to_goal()

        if action == MOVE_FORWARD:
            drow, dcol = self._heading_delta()
            target = self.pos + np.array([drow, dcol], dtype=np.int64)
            if self._is_blocked(target):
                collision = True
                terminated = True
                reward += -50.0
            else:
                self.pos = target
        elif action == TURN_LEFT:
            self.heading = (self.heading - 1) % self.N_HEADINGS
        elif action == TURN_RIGHT:
            self.heading = (self.heading + 1) % self.N_HEADINGS
        elif action == STOP:
            if dist_before <= self.goal_radius:
                success = True
                terminated = True
                reward += 100.0
            # otherwise: wasted step, only the step penalty applies

        dist_after = self._dist_to_goal()
        reward += 5.0 * (dist_before - dist_after)  # progress shaping

        # Moving onto the goal cell also terminates the episode.
        if not terminated and dist_after <= self.goal_radius:
            success = True
            terminated = True
            reward += 100.0

        if not terminated and self.steps >= self.max_steps:
            truncated = True

        info = self._get_info(success=success, collision=collision)
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------ observation

    def _get_obs(self) -> Obs:
        return {"grid": self.local_window(), "vec": self.state_vector()}

    def local_window(
        self,
        pos: Optional[np.ndarray] = None,
        heading: Optional[int] = None,
    ) -> np.ndarray:
        """Egocentric occupancy window rotated so forward points up.

        For each window cell we rotate its (forward, lateral) offset by the
        agent heading and sample the nearest global map cell.  For cardinal
        headings this is an exact rotation; for diagonal (45-degree) headings
        it is a nearest-neighbour resampling.  Out-of-bounds cells read as
        OBSTACLE (the map boundary behaves like a wall).
        """
        assert self.grid is not None
        pos = self.pos if pos is None else pos
        heading = self.heading if heading is None else heading
        w, h = self.window_size, self.half_window
        window = np.full((w, w), OBSTACLE, dtype=np.float32)

        angle = heading * math.pi / 4.0
        # Heading unit vector in (x=col, y=row) coordinates; row grows "down".
        ux, uy = math.cos(angle), math.sin(angle)
        # Vector pointing to the agent's right (heading rotated +90 deg).
        rx, ry = -uy, ux

        row0, col0 = float(pos[0]), float(pos[1])
        for i in range(w):
            fwd = h - i  # rows above centre are "forward"
            for j in range(w):
                lat = j - h  # columns right of centre are "to the right"
                x = col0 + fwd * ux + lat * rx
                y = row0 + fwd * uy + lat * ry
                r, c = int(round(y)), int(round(x))
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    window[i, j] = self.grid[r, c]
        window[h, h] = FREE  # the agent's own cell is free by construction
        return window

    def state_vector(
        self,
        pos: Optional[np.ndarray] = None,
        heading: Optional[int] = None,
    ) -> np.ndarray:
        """``[x_rel, y_rel, cos(theta), sin(theta), dist_to_goal]``."""
        pos = self.pos if pos is None else pos
        heading = self.heading if heading is None else heading
        denom = max(self.grid_size - 1, 1)
        x_rel = (self.goal[1] - pos[1]) / denom  # in [-1, 1]
        y_rel = (self.goal[0] - pos[0]) / denom  # in [-1, 1]
        angle = heading * math.pi / 4.0
        dist = float(np.linalg.norm((self.goal - pos).astype(np.float64)))
        return np.array(
            [x_rel, y_rel, math.cos(angle), math.sin(angle), dist / self._diag],
            dtype=np.float32,
        )

    # -------------------------------------------------------------- internals

    def _heading_delta(self) -> Tuple[int, int]:
        """(drow, dcol) for one forward step along the current heading."""
        angle = self.heading * math.pi / 4.0
        return int(round(math.sin(angle))), int(round(math.cos(angle)))

    def _is_blocked(self, cell: np.ndarray) -> bool:
        r, c = int(cell[0]), int(cell[1])
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return True
        assert self.grid is not None
        return self.grid[r, c] == OBSTACLE

    def _dist_to_goal(self) -> float:
        return float(np.linalg.norm((self.goal - self.pos).astype(np.float64)))

    def _get_info(self, success: bool, collision: bool) -> dict:
        return {
            "pos": self.pos.copy(),
            "goal": self.goal.copy(),
            "heading": self.heading,
            "steps": self.steps,
            "success": success,
            "collision": collision,
            "dist_to_goal": self._dist_to_goal(),
        }

    @staticmethod
    def _path_exists(
        grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> bool:
        """BFS over free cells with 8-connectivity (agent moves diagonally)."""
        n = grid.shape[0]
        seen = np.zeros_like(grid, dtype=bool)
        queue: deque = deque([start])
        seen[start] = True
        while queue:
            r, c = queue.popleft()
            if (r, c) == tuple(goal):
                return True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < n
                        and 0 <= nc < n
                        and not seen[nr, nc]
                        and grid[nr, nc] == FREE
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
        return False
