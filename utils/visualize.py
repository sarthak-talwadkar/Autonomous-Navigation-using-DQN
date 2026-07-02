"""Plotting and rendering utilities (headless-safe).

Uses the matplotlib ``Agg`` backend so everything works without a display;
nothing here ever calls ``plt.show()`` -- all functions save to files or
return strings.  Colors follow a CVD-validated palette: categorical series
hues are assigned in a fixed order (never cycled), magnitude (Q-values) uses
a single-hue light-to-dark blue ramp, and chart chrome (grid, axes, ink) is
kept recessive.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Fixed-order categorical palette (validated; do not cycle or reorder).
SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
# Single-hue sequential ramp (blue, light -> dark) for magnitude encodings.
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def _style_axes(ax: plt.Axes) -> None:
    """Recessive chart chrome: hairline grid, muted labels, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def rolling_mean(values: Sequence[float], window: int) -> np.ndarray:
    """Trailing rolling mean with a growing window at the start."""
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    csum = np.cumsum(np.insert(values, 0, 0.0))
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def plot_reward_curve(
    rewards: Sequence[float], path: str, window: int = 20, title: str = "Training reward"
) -> None:
    """Save episode rewards (raw, faint) with a rolling mean (bold)."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    episodes = np.arange(1, len(rewards) + 1)
    ax.plot(episodes, rewards, color=SERIES_COLORS[0], alpha=0.25, linewidth=1.0)
    ax.plot(
        episodes,
        rolling_mean(rewards, window),
        color=SERIES_COLORS[0],
        linewidth=2.0,
        label=f"rolling mean ({window} ep)",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title(title)
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_reward_curves(
    curves: Dict[str, np.ndarray],
    path: str,
    window: int = 20,
    title: str = "Learning curves",
) -> None:
    """Compare agents: mean rolling reward per agent, min/max band over trials.

    ``curves`` maps label -> array of shape (n_trials, n_episodes).
    Series colors are assigned in fixed palette order.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    for slot, (label, runs) in enumerate(curves.items()):
        runs = np.atleast_2d(np.asarray(runs, dtype=np.float64))
        smoothed = np.stack([rolling_mean(r, window) for r in runs])
        episodes = np.arange(1, smoothed.shape[1] + 1)
        color = SERIES_COLORS[slot % len(SERIES_COLORS)]
        mean = smoothed.mean(axis=0)
        ax.plot(episodes, mean, color=color, linewidth=2.0, label=label)
        if runs.shape[0] > 1:
            ax.fill_between(
                episodes,
                smoothed.min(axis=0),
                smoothed.max(axis=0),
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        # Selective direct label at the line's end.
        ax.annotate(
            label,
            (episodes[-1], mean[-1]),
            textcoords="offset points",
            xytext=(4, 0),
            fontsize=8,
            color=color,
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward (rolling mean, {window} ep)")
    ax.set_title(title)
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------- grid render

_HEADING_GLYPHS = ["→", "↘", "↓", "↙", "←", "↖", "↑", "↗"]


def ascii_render(env) -> str:
    """Plain-text rendering of the grid world (headless-friendly)."""
    if env.grid is None:
        return "<environment not reset>"
    rows = []
    for r in range(env.grid_size):
        cells = []
        for c in range(env.grid_size):
            if r == env.pos[0] and c == env.pos[1]:
                cells.append(_HEADING_GLYPHS[env.heading])
            elif r == env.goal[0] and c == env.goal[1]:
                cells.append("G")
            elif env.grid[r, c] == 1.0:
                cells.append("#")
            else:
                cells.append(".")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def render_frame(env, path: str, title: Optional[str] = None) -> None:
    """Save a matplotlib rendering of the current environment state."""
    if env.grid is None:
        raise RuntimeError("environment not reset")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    # Obstacles dark, free space near-surface (single-hue, magnitude = blockage).
    ax.imshow(env.grid, cmap=SEQ_CMAP, vmin=0.0, vmax=1.4, interpolation="nearest")
    ax.scatter(
        [env.goal[1]], [env.goal[0]], marker="*", s=220, color=SERIES_COLORS[2],
        edgecolors=INK, linewidths=0.5, label="goal", zorder=3,
    )
    angle = env.heading * math.pi / 4.0
    ax.scatter(
        [env.pos[1]], [env.pos[0]], marker="o", s=90, color=SERIES_COLORS[1],
        edgecolors=INK, linewidths=0.5, label="agent", zorder=3,
    )
    ax.annotate(
        "",
        xy=(env.pos[1] + 0.9 * math.cos(angle), env.pos[0] + 0.9 * math.sin(angle)),
        xytext=(env.pos[1], env.pos[0]),
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.5),
        zorder=4,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, color=INK, fontsize=10)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


@torch.no_grad()
def q_value_heatmap(
    network: torch.nn.Module,
    env,
    path: str,
    device: Optional[torch.device] = None,
    title: str = "Q-value landscape (max over actions)",
) -> None:
    """Heatmap of ``max_a Q(s, a)`` over every free cell of the current map.

    The agent is virtually placed at each free cell with its current heading
    and the fixed episode goal; warmer (darker blue) = higher value.
    """
    if env.grid is None:
        raise RuntimeError("environment not reset")
    device = device or next(network.parameters()).device
    free_cells = np.argwhere(env.grid == 0.0)
    grids, vecs = [], []
    for cell in free_cells:
        grids.append(env.local_window(pos=cell))
        vecs.append(env.state_vector(pos=cell))
    grid_t = torch.as_tensor(np.stack(grids), dtype=torch.float32, device=device)
    vec_t = torch.as_tensor(np.stack(vecs), dtype=torch.float32, device=device)
    q_max = network(grid_t, vec_t).max(dim=1).values.cpu().numpy()

    field = np.full_like(env.grid, np.nan, dtype=np.float64)
    for (r, c), q in zip(free_cells, q_max):
        field[r, c] = q

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    masked = np.ma.masked_invalid(field)
    cmap = SEQ_CMAP.copy()
    cmap.set_bad(color=INK)  # obstacles rendered in ink
    im = ax.imshow(masked, cmap=cmap, interpolation="nearest")
    ax.scatter(
        [env.goal[1]], [env.goal[0]], marker="*", s=200, color=SERIES_COLORS[2],
        edgecolors=INK, linewidths=0.5, zorder=3,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BASELINE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
