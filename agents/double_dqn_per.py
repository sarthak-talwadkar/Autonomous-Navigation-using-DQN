"""Double DQN + Prioritized Experience Replay (the full agent).

Combines the Double-DQN target with proportional PER: minibatches are drawn
by TD-error priority from a sum-tree, the loss is the importance-weighted
smooth L1, and sampled transitions' priorities are refreshed with the new
|TD| after each update.  Beta anneals linearly 0.4 -> 1.0 over training.
"""

from __future__ import annotations

from typing import Optional

import torch

from agents.double_dqn import DoubleDQNAgent
from memory.per_buffer import PrioritizedReplayBuffer
from utils.schedules import LinearSchedule


class DoubleDQNPERAgent(DoubleDQNAgent):
    """Double DQN with prioritized replay and IS-weighted loss."""

    name = "double_dqn_per"

    def __init__(
        self,
        *args,
        per_alpha: float = 0.6,
        per_eps: float = 1e-6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100_000,
        **kwargs,
    ) -> None:
        self._per_alpha = per_alpha
        self._per_eps = per_eps
        super().__init__(*args, **kwargs)
        self.beta_schedule = LinearSchedule(beta_start, beta_end, beta_steps)

    @property
    def beta(self) -> float:
        return self.beta_schedule.value(self.env_steps)

    def _make_buffer(self, buffer_size: int, seed: Optional[int]):
        return PrioritizedReplayBuffer(
            buffer_size, alpha=self._per_alpha, eps=self._per_eps, seed=seed
        )

    def learn(self) -> Optional[float]:
        """One prioritized, importance-weighted gradient step."""
        if len(self.buffer) < self.learn_start:
            return None
        indices, batch, weights = self.buffer.sample(self.batch_size, beta=self.beta)
        t = self._to_tensors(batch)
        w = torch.as_tensor(weights, device=self.device)

        q_values = self._q_selected(t)
        targets = self._targets(t)

        td_errors = (q_values - targets).abs().detach()
        elementwise = torch.nn.functional.smooth_l1_loss(
            q_values, targets, reduction="none"
        )
        loss = (w * elementwise).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Refresh sampled transitions' priorities with the fresh |TD| errors.
        self.buffer.update_priorities(indices, td_errors.cpu().numpy())

        self.grad_steps += 1
        if self.grad_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return float(loss.item())
