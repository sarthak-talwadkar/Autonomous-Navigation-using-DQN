#!/usr/bin/env python3
"""Training entry point for the grid-world DQN navigation agents.

Example (README usage):

    python train.py \
        --env grid \
        --grid-size 20 \
        --agent double_dqn_per \
        --episodes 2000 \
        --buffer-size 50000 \
        --save-dir checkpoints/

Optional improvements: ``--dueling`` (Dueling DQN heads) and ``--n-step N``
(n-step returns; ``--n-step 1`` recovers standard one-step TD).

Outputs in ``--save-dir``: ``best.pt`` (highest rolling-mean reward),
``last.pt``, ``metrics.csv``, ``reward_curve.png``, ``q_heatmap.png``.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from typing import Optional

import numpy as np
import torch

from agents import AGENT_REGISTRY, make_agent
from envs.grid_world import GridWorldEnv
from utils.visualize import plot_reward_curve, q_value_heatmap, rolling_mean


def set_global_seeds(seed: int) -> None:
    """Deterministic seeding for python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_training(
    agent_name: str,
    grid_size: int = 20,
    episodes: int = 2000,
    buffer_size: int = 50_000,
    save_dir: Optional[str] = None,
    dueling: bool = False,
    n_step: int = 3,
    seed: int = 0,
    lr: float = 1e-4,
    gamma: float = 0.99,
    batch_size: int = 64,
    target_update: int = 1_000,
    eps_decay_steps: int = 50_000,
    learn_start: int = 1_000,
    max_steps: Optional[int] = None,
    obstacle_density: float = 0.15,
    window_size: int = 7,
    rolling_window: int = 20,
    log_every: int = 10,
    quiet: bool = False,
) -> dict:
    """Train one agent; returns a history dict (also used by benchmark.py)."""
    set_global_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = GridWorldEnv(
        grid_size=grid_size,
        window_size=window_size,
        obstacle_density=obstacle_density,
        max_steps=max_steps,
        seed=seed,
    )
    agent = make_agent(
        agent_name,
        window_size=window_size,
        state_dim=GridWorldEnv.STATE_DIM,
        n_actions=GridWorldEnv.N_ACTIONS,
        dueling=dueling,
        n_step=n_step,
        gamma=gamma,
        lr=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update,
        eps_decay_steps=eps_decay_steps,
        learn_start=learn_start,
        device=device,
        seed=seed,
    )

    history = {"rewards": [], "lengths": [], "successes": [], "collisions": [],
               "losses": [], "epsilons": []}
    best_rolling = -np.inf
    t0 = time.time()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward, ep_losses = 0.0, []
        success = collision = False
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.observe(obs, action, reward, next_obs, terminated, truncated)
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)
            ep_reward += reward
            obs = next_obs
            done = terminated or truncated
        success, collision = info["success"], info["collision"]

        history["rewards"].append(ep_reward)
        history["lengths"].append(info["steps"])
        history["successes"].append(int(success))
        history["collisions"].append(int(collision))
        history["losses"].append(float(np.mean(ep_losses)) if ep_losses else float("nan"))
        history["epsilons"].append(agent.epsilon)

        window = min(rolling_window, episode)
        roll = float(np.mean(history["rewards"][-window:]))
        if save_dir and episode >= rolling_window and roll > best_rolling:
            best_rolling = roll
            agent.save(
                os.path.join(save_dir, "best.pt"),
                extra={"episode": episode, "rolling_reward": roll},
            )

        if not quiet and (episode % log_every == 0 or episode == episodes):
            sr = float(np.mean(history["successes"][-window:]))
            loss_str = (
                f"{history['losses'][-1]:.4f}"
                if np.isfinite(history["losses"][-1])
                else "n/a"
            )
            print(
                f"[{agent_name}] ep {episode:4d}/{episodes} | "
                f"reward {ep_reward:8.2f} | roll({window}) {roll:8.2f} | "
                f"success {sr:.2f} | eps {agent.epsilon:.3f} | "
                f"loss {loss_str} | steps {agent.env_steps}"
            )

    elapsed = time.time() - t0
    if not quiet:
        print(f"[{agent_name}] finished {episodes} episodes in {elapsed:.1f}s")

    if save_dir:
        agent.save(
            os.path.join(save_dir, "last.pt"),
            extra={"episode": episodes, "rolling_reward": roll},
        )
        if not np.isfinite(best_rolling):  # short runs: ensure best.pt exists
            agent.save(
                os.path.join(save_dir, "best.pt"),
                extra={"episode": episodes, "rolling_reward": roll},
            )
        _write_metrics(os.path.join(save_dir, "metrics.csv"), history)
        plot_reward_curve(
            history["rewards"],
            os.path.join(save_dir, "reward_curve.png"),
            window=rolling_window,
            title=f"{agent_name} on {grid_size}x{grid_size} grid",
        )
        env.reset()  # fresh layout for the value-landscape snapshot
        q_value_heatmap(
            agent.online_net, env, os.path.join(save_dir, "q_heatmap.png"),
            device=device,
        )

    history["best_rolling"] = best_rolling
    history["elapsed_s"] = elapsed
    return history


def _write_metrics(path: str, history: dict) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "reward", "length", "success", "collision", "epsilon", "mean_loss"]
        )
        for i in range(len(history["rewards"])):
            writer.writerow(
                [
                    i + 1,
                    f"{history['rewards'][i]:.4f}",
                    history["lengths"][i],
                    history["successes"][i],
                    history["collisions"][i],
                    f"{history['epsilons'][i]:.4f}",
                    f"{history['losses'][i]:.6f}",
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a DQN navigation agent")
    p.add_argument("--env", default="grid", choices=["grid"], help="environment id")
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--agent", default="double_dqn_per", choices=sorted(AGENT_REGISTRY))
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--dueling", action="store_true", help="use Dueling DQN heads")
    p.add_argument(
        "--n-step", type=int, default=3,
        help="n-step return horizon (1 = standard one-step TD)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-update", type=int, default=1_000)
    p.add_argument("--eps-decay-steps", type=int, default=50_000)
    p.add_argument("--learn-start", type=int, default=1_000)
    p.add_argument("--max-steps", type=int, default=None, help="episode step limit")
    p.add_argument("--obstacle-density", type=float, default=0.15)
    p.add_argument("--log-every", type=int, default=10)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        agent_name=args.agent,
        grid_size=args.grid_size,
        episodes=args.episodes,
        buffer_size=args.buffer_size,
        save_dir=args.save_dir,
        dueling=args.dueling,
        n_step=args.n_step,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        target_update=args.target_update,
        eps_decay_steps=args.eps_decay_steps,
        learn_start=args.learn_start,
        max_steps=args.max_steps,
        obstacle_density=args.obstacle_density,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
