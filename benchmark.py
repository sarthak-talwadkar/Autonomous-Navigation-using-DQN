#!/usr/bin/env python3
"""Benchmark DQN variants against each other on the grid world.

Example (README usage):

    python benchmark.py \
        --agents vanilla_dqn double_dqn double_dqn_per \
        --episodes 2000 \
        --trials 5 \
        --plot

Each (agent, trial) pair runs a full training with seed ``seed + trial``.
Outputs (under ``--save-dir``): ``benchmark_curves.png`` (rolling-mean
learning curves, min/max band over trials), ``benchmark_summary.csv``, and a
printed summary table.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from agents import AGENT_REGISTRY
from train import run_training
from utils.visualize import plot_reward_curves, rolling_mean


def benchmark(
    agents: list,
    episodes: int = 2000,
    trials: int = 5,
    plot: bool = False,
    grid_size: int = 20,
    buffer_size: int = 50_000,
    dueling: bool = False,
    n_step: int = 3,
    seed: int = 0,
    eps_decay_steps: int = 50_000,
    learn_start: int = 1_000,
    obstacle_density: float = 0.15,
    save_dir: str = "benchmark_results",
    rolling_window: int = 20,
) -> dict:
    """Run all agent x trial combinations and aggregate results."""
    os.makedirs(save_dir, exist_ok=True)
    results = {}  # agent -> dict of stacked per-trial arrays
    for agent_name in agents:
        reward_runs, success_runs = [], []
        for trial in range(trials):
            print(f"\n### {agent_name} — trial {trial + 1}/{trials} "
                  f"(seed {seed + trial}) ###")
            hist = run_training(
                agent_name=agent_name,
                grid_size=grid_size,
                episodes=episodes,
                buffer_size=buffer_size,
                save_dir=None,  # benchmark runs keep no checkpoints
                dueling=dueling,
                n_step=n_step,
                seed=seed + trial,
                eps_decay_steps=eps_decay_steps,
                learn_start=learn_start,
                obstacle_density=obstacle_density,
                rolling_window=rolling_window,
                log_every=max(episodes // 5, 1),
            )
            reward_runs.append(hist["rewards"])
            success_runs.append(hist["successes"])
        results[agent_name] = {
            "rewards": np.asarray(reward_runs, dtype=np.float64),
            "successes": np.asarray(success_runs, dtype=np.float64),
        }

    # ------------------------------------------------------------- summary
    tail = max(episodes // 5, 1)  # statistics over the final 20% of episodes
    rows = []
    for agent_name, res in results.items():
        final_roll = np.stack(
            [rolling_mean(r, rolling_window)[-1] for r in res["rewards"]]
        )
        tail_reward = res["rewards"][:, -tail:].mean(axis=1)
        tail_success = res["successes"][:, -tail:].mean(axis=1)
        rows.append(
            {
                "agent": agent_name,
                "final_rolling_reward_mean": float(final_roll.mean()),
                "final_rolling_reward_std": float(final_roll.std()),
                "tail_reward_mean": float(tail_reward.mean()),
                "tail_success_rate": float(tail_success.mean()),
            }
        )

    csv_path = os.path.join(save_dir, "benchmark_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== Benchmark summary "
          f"({trials} trial(s), {episodes} episodes, final {tail} episodes) =====")
    header = f"{'agent':<18} {'final roll reward':>18} {'tail reward':>12} {'tail success':>13}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['agent']:<18} "
            f"{row['final_rolling_reward_mean']:>10.2f} ± {row['final_rolling_reward_std']:<5.2f} "
            f"{row['tail_reward_mean']:>12.2f} "
            f"{row['tail_success_rate']:>12.2%}"
        )
    print(f"\nsummary written to {csv_path}")

    if plot:
        plot_path = os.path.join(save_dir, "benchmark_curves.png")
        plot_reward_curves(
            {name: res["rewards"] for name, res in results.items()},
            plot_path,
            window=rolling_window,
            title=f"DQN variants on {grid_size}x{grid_size} grid "
                  f"({trials} trial(s))",
        )
        print(f"learning-curve plot written to {plot_path}")

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Compare DQN variants")
    p.add_argument(
        "--agents", nargs="+", default=sorted(AGENT_REGISTRY),
        choices=sorted(AGENT_REGISTRY),
    )
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--dueling", action="store_true")
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps-decay-steps", type=int, default=50_000)
    p.add_argument("--learn-start", type=int, default=1_000)
    p.add_argument("--obstacle-density", type=float, default=0.15)
    p.add_argument("--save-dir", default="benchmark_results")
    args = p.parse_args()
    benchmark(
        agents=args.agents,
        episodes=args.episodes,
        trials=args.trials,
        plot=args.plot,
        grid_size=args.grid_size,
        buffer_size=args.buffer_size,
        dueling=args.dueling,
        n_step=args.n_step,
        seed=args.seed,
        eps_decay_steps=args.eps_decay_steps,
        learn_start=args.learn_start,
        obstacle_density=args.obstacle_density,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
