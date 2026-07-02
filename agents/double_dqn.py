"""Double DQN agent (van Hasselt et al., 2016).

Decouples action *selection* from action *evaluation* in the TD target:

    a* = argmax_a Q_online(s', a)          (online network selects)
    y  = r_n + gamma^n * Q_target(s', a*)  (target network evaluates)

This dampens the max-operator overestimation bias of vanilla DQN.  Uses a
Huber (smooth L1) loss for robustness to the occasional large TD error.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """DQN with the Double-DQN target; uniform replay."""

    name = "double_dqn"

    def _next_state_values(self, t: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Online network selects the best next action ...
        next_actions = self.online_net(t["next_grids"], t["next_vecs"]).argmax(dim=1)
        # ... target network evaluates it (not its own argmax).
        next_q = self.target_net(t["next_grids"], t["next_vecs"])
        return next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)

    def _loss(self, q_values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(q_values, targets)
