#!/usr/bin/env python3
"""Evaluate a trained navigation agent (greedy policy) with optional rendering.

Example (README usage):

    python evaluate.py \
        --weights checkpoints/best.pt \
        --env grid \
        --episodes 100 \
        --render

Rendering is fully headless: ``--render`` prints an ASCII view of the first
rendered episodes to stdout and saves matplotlib PNG frames under
``--render-dir`` (default ``renders/``).
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch

from envs.grid_world import GridWorldEnv
from models.dqn_network import build_network
from utils.visualize import ascii_render, render_frame


def load_policy(weights: str, device: torch.device):
    """Rebuild the Q-network from a checkpoint's stored config."""
    payload = torch.load(weights, map_location=device, weights_only=True)
    cfg = payload["config"]
    net = build_network(
        window_size=cfg["window_size"],
        state_dim=cfg["state_dim"],
        n_actions=cfg["n_actions"],
        dueling=cfg["dueling"],
    ).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, cfg


@torch.no_grad()
def greedy_action(net, obs, device: torch.device) -> int:
    grid = torch.as_tensor(obs["grid"], dtype=torch.float32, device=device).unsqueeze(0)
    vec = torch.as_tensor(obs["vec"], dtype=torch.float32, device=device).unsqueeze(0)
    return int(net(grid, vec).argmax(dim=1).item())


def evaluate(
    weights: str,
    grid_size: int = 20,
    episodes: int = 100,
    render: bool = False,
    render_dir: str = "renders",
    render_episodes: int = 2,
    obstacle_density: float = 0.15,
    max_steps: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Run greedy rollouts; returns summary statistics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, cfg = load_policy(weights, device)
    env = GridWorldEnv(
        grid_size=grid_size,
        window_size=cfg["window_size"],
        obstacle_density=obstacle_density,
        max_steps=max_steps,
        seed=seed,
    )
    if render:
        os.makedirs(render_dir, exist_ok=True)

    rewards, lengths, successes, collisions = [], [], [], []
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        ep_reward = 0.0
        done = False
        frame = 0
        do_render = render and ep <= render_episodes
        if do_render:
            print(f"\n=== Episode {ep} (start) ===")
            print(ascii_render(env))
            render_frame(
                env,
                os.path.join(render_dir, f"ep{ep:03d}_step{frame:03d}.png"),
                title=f"episode {ep} step {frame}",
            )
        while not done:
            action = greedy_action(net, obs, device)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            if do_render:
                frame += 1
                render_frame(
                    env,
                    os.path.join(render_dir, f"ep{ep:03d}_step{frame:03d}.png"),
                    title=f"episode {ep} step {frame}",
                )
        if do_render:
            outcome = (
                "SUCCESS" if info["success"]
                else "COLLISION" if info["collision"]
                else "TIMEOUT"
            )
            print(f"=== Episode {ep} (end: {outcome}, reward {ep_reward:.1f}) ===")
            print(ascii_render(env))
        rewards.append(ep_reward)
        lengths.append(info["steps"])
        successes.append(int(info["success"]))
        collisions.append(int(info["collision"]))

    summary = {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "mean_length": float(np.mean(lengths)),
    }
    print(
        f"\nEvaluation over {episodes} episodes "
        f"(agent: {cfg['agent']}, dueling={cfg['dueling']}, n_step={cfg['n_step']}):\n"
        f"  mean reward   : {summary['mean_reward']:8.2f} +/- {summary['std_reward']:.2f}\n"
        f"  success rate  : {summary['success_rate']:.2%}\n"
        f"  collision rate: {summary['collision_rate']:.2%}\n"
        f"  mean ep length: {summary['mean_length']:.1f} steps"
    )
    if render:
        print(f"  frames saved under: {render_dir}/")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained DQN navigator")
    p.add_argument("--weights", required=True, help="checkpoint path (.pt)")
    p.add_argument("--env", default="grid", choices=["grid"])
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--render", action="store_true", help="ASCII + PNG frame rendering")
    p.add_argument("--render-dir", default="renders")
    p.add_argument("--render-episodes", type=int, default=2)
    p.add_argument("--obstacle-density", type=float, default=0.15)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    evaluate(
        weights=args.weights,
        grid_size=args.grid_size,
        episodes=args.episodes,
        render=args.render,
        render_dir=args.render_dir,
        render_episodes=args.render_episodes,
        obstacle_density=args.obstacle_density,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
