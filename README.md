# Autonomous Navigation using DQN

> **Status: Active Development — Work in Progress**
> Currently transitioning from custom grid world to NVIDIA Isaac Sim. Quantitative results pending.

**Deep Reinforcement Learning for autonomous robot navigation** using Double DQN with Prioritized Experience Replay. The agent learns to navigate from start to goal in obstacle-rich environments using occupancy grid observations and a state vector of position and heading — without explicit path planning or hand-crafted reward shaping. Supports both discrete and continuous action spaces, with an active migration to Isaac Sim for photorealistic sim-to-real training.

---

## Demo

![DQN Navigation Demo](assets/demo.gif)

> Agent navigating a custom grid world. Color map shows Q-value landscape learned by the network — warmer colors indicate higher value regions toward the goal.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [DQN Variants](#dqn-variants)
  - [Vanilla DQN — Baseline](#vanilla-dqn--baseline)
  - [Double DQN](#double-dqn)
  - [Prioritized Experience Replay](#prioritized-experience-replay)
  - [Combined: Double DQN + PER](#combined-double-dqn--per)
- [Network Architecture](#network-architecture)
- [Training Pipeline](#training-pipeline)
- [Environments](#environments)
  - [Custom Grid World](#custom-grid-world)
  - [Isaac Sim (In Progress)](#isaac-sim-in-progress)
- [Current Status](#current-status)
- [Future Work](#future-work)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

Classical navigation pipelines (A\*, RRT\*, Dijkstra) require an explicit map and a known goal location in world coordinates. They are brittle to dynamic obstacles and require re-planning from scratch when the environment changes. Deep RL navigation agents learn a policy directly from environment interactions — implicitly encoding obstacle avoidance, goal-seeking, and replanning into the network weights.

This project trains a DQN-based navigation agent that:
- Observes the environment as a local occupancy grid + (x, y, θ) state vector
- Selects discrete movement actions (or continuous velocity commands)
- Learns purely from reward signals — no demonstrations, no map, no planner

The two core algorithmic improvements over vanilla DQN — **Double DQN** and **Prioritized Experience Replay** — address the two most significant failure modes of naive DQN: Q-value overestimation and sample inefficiency.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Environment                            │
│         Custom Grid World  /  Isaac Sim (in progress)        │
└───────────────────────────┬──────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │      Observation           │
              │  Local occupancy grid      │
              │  (N×N window around agent) │
              │  + state vector [x, y, θ]  │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │      Online Network        │
              │   Q_online(s, a; θ)        │
              │                            │
              │  CNN (grid features)       │
              │     ↓                      │
              │  Concat + MLP              │
              │  (grid features + state)   │
              │     ↓                      │
              │  Q-values per action       │
              └──────┬──────────┬──────────┘
                     │          │
              ε-greedy       Target Network
              action          Q_target(s',a; θ⁻)
              selection       (Double DQN)
                     │
                     ▼
              ┌──────────────┐
              │  Environment │
              │  step(action)│
              └──────┬───────┘
                     │ (s, a, r, s', done)
                     ▼
              ┌──────────────────────────┐
              │  Prioritized Replay      │
              │  Buffer (PER)            │
              │  Sum-tree priority store │
              │  TD-error based sampling │
              └──────────┬───────────────┘
                         │ weighted minibatch
                         ▼
              ┌──────────────────────────┐
              │   TD Update              │
              │   Double DQN target:     │
              │   r + γ Q_target(s',     │
              │     argmax_a Q_online)   │
              └──────────────────────────┘
```

---

## Observation Space

The agent observes the environment through two complementary inputs that are fused before the policy head:

### 1. Local Occupancy Grid

A square window of size `N×N` cells centered on the agent, extracted from the global occupancy map:

```
0 = free space
1 = obstacle
0.5 = unknown

Example (7×7 window):
[ 0   0   0   0   0   0   0 ]
[ 0   1   1   0   0   0   0 ]
[ 0   1   1   0   0   0   0 ]
[ 0   0   0  [A]  0   0   0 ]   ← agent at center
[ 0   0   0   0   1   1   0 ]
[ 0   0   0   0   1   1   0 ]
[ 0   0   0   0   0   0   0 ]
```

The window rotates with the agent's heading so that "forward" always points up in the grid — making the representation **egocentric** and heading-invariant. This dramatically simplifies what the CNN needs to learn: the same obstacle pattern relative to the agent always looks the same regardless of global orientation.

### 2. State Vector

```
s_vec = [x_rel, y_rel, cos(θ), sin(θ), dist_to_goal]

where:
  x_rel, y_rel  = goal position relative to agent (normalized to [-1, 1])
  cos(θ), sin(θ) = heading encoded as unit vector (avoids angle wraparound)
  dist_to_goal  = Euclidean distance to goal (normalized)
```

`cos(θ), sin(θ)` encoding instead of raw angle avoids the discontinuity at ±π that would confuse the network — angles 179° and -179° are nearly identical headings but numerically far apart as scalars.

### Fusion

```python
# CNN processes occupancy grid
grid_features = cnn_encoder(occupancy_grid)    # R^256

# Concat with state vector
obs_fused = torch.cat([grid_features, state_vector], dim=-1)  # R^261

# Pass to MLP policy head
q_values = mlp_head(obs_fused)                 # R^|A|
```

---

## Action Space

### Discrete Actions

```python
ACTIONS = {
    0: "MOVE_FORWARD",   # advance by step_size along current heading
    1: "TURN_LEFT",      # rotate -45° in place
    2: "TURN_RIGHT",     # rotate +45° in place
    3: "STOP"            # terminate episode (used at goal)
}
```

Discrete actions simplify Q-learning — the network outputs one Q-value per action, and argmax selection is straightforward. Suitable for grid-world navigation where motion is cell-aligned.

### Continuous Actions *(in progress for Isaac Sim)*

For Isaac Sim deployment, actions are continuous velocity commands:

```python
# Action vector: [v_linear, v_angular]
# v_linear  ∈ [-0.5, 0.5] m/s   (forward/backward)
# v_angular ∈ [-1.0, 1.0] rad/s (turn rate)
```

Continuous actions require a different architecture (actor-critic methods like SAC or DDPG are better suited), but a discretized approximation is currently used to keep the DQN framework intact — velocity space is binned into a fixed set of (v_linear, v_angular) pairs treated as discrete tokens.

---

## Reward Function

The reward function is carefully shaped to encourage goal-seeking and penalize collisions without over-specifying the solution:

```python
def compute_reward(state, next_state, done, collision):
    reward = 0.0

    # Goal reached
    if done and not collision:
        reward += 100.0

    # Collision
    elif collision:
        reward -= 50.0

    # Progress reward: positive if agent moved closer to goal
    dist_before = euclidean(state.pos, goal)
    dist_after  = euclidean(next_state.pos, goal)
    reward += 5.0 * (dist_before - dist_after)   # positive = closer

    # Step penalty: discourages unnecessary exploration
    reward -= 0.1

    return reward
```

**Design rationale:**
- **Large goal reward (+100):** Creates a strong terminal signal that backpropagates through the Q-value chain
- **Collision penalty (-50):** Asymmetric with goal reward — collision is bad but not catastrophic, allowing the agent to recover from near-misses early in training
- **Progress reward:** Dense shaping signal that guides the agent toward the goal before it has ever reached it — without this, sparse rewards lead to extremely slow initial learning in large environments
- **Step penalty (-0.1):** Prevents the agent from loitering in safe regions far from the goal

---

## DQN Variants

### Vanilla DQN — Baseline

Standard DQN (Mnih et al., 2015) with experience replay and a target network:

```
TD target:  y = r + γ × max_a' Q(s', a'; θ⁻)
TD error:   δ = y - Q(s, a; θ)
Loss:       L = E[δ²]
```

**Two failure modes addressed by the variants below:**

1. **Overestimation bias:** `max_a' Q(s', a'; θ⁻)` uses the same network to both *select* the best action and *evaluate* it. If Q is noisy (early training), the max operator systematically picks overestimated Q-values, causing the target to be too optimistic, destabilizing training.

2. **Sample inefficiency:** Uniform random sampling from the replay buffer treats all transitions equally. Rare but informative transitions (high TD error) are sampled at the same frequency as common, already-learned transitions — wasting capacity on easy examples.

---

### Double DQN

Double DQN (van Hasselt et al., 2016) decouples action *selection* from action *evaluation* by using two separate networks:

```
Action selection:  a* = argmax_a  Q_online(s', a; θ)      ← online network picks action
Action evaluation: y  = r + γ × Q_target(s', a*; θ⁻)      ← target network evaluates it
```

**Why this reduces overestimation:**

In vanilla DQN, if `Q(s', a₃)` happens to be noisy-high, `max` selects it and also evaluates it — the noise compounds. In Double DQN, the online network may still select `a₃`, but the target network evaluates it independently. If the target network doesn't share the same noise, it returns a more accurate value for `a₃`, dampening the overestimation.

```python
with torch.no_grad():
    # Online network selects best action in next state
    next_actions = self.online_net(next_states).argmax(dim=1)

    # Target network evaluates selected action (not its own argmax)
    next_q = self.target_net(next_states)
    next_q_selected = next_q.gather(1, next_actions.unsqueeze(1)).squeeze()

    targets = rewards + gamma * next_q_selected * (1 - dones)
```

---

### Prioritized Experience Replay

PER (Schaul et al., 2016) replaces uniform replay sampling with priority-weighted sampling — transitions with higher TD error are sampled more frequently because they represent experiences the network has not yet learned from well.

**Priority assignment:**

```
p_i = |δ_i| + ε

where:
  δ_i = TD error for transition i
  ε   = small constant (prevents zero priority for learned transitions)
```

**Sampling probability:**

```
P(i) = p_i^α / Σ_j p_j^α

where:
  α ∈ [0, 1] controls prioritization strength
  α = 0 → uniform sampling (vanilla DQN)
  α = 1 → full prioritization
```

**Importance sampling correction:**

PER introduces a bias — high-priority transitions are over-represented relative to their true frequency in the environment. This is corrected by weighting each transition's gradient update by its inverse sampling probability:

```
w_i = (1 / (N × P(i)))^β

where:
  N = buffer size
  β anneals from β₀ → 1 over training  (full correction at convergence)
```

```python
loss = (w_i * (Q(s,a) - target)²).mean()   # importance-weighted loss
```

**Sum-tree data structure:**

Efficiently sampling from a non-uniform priority distribution over a buffer of size N requires O(log N) per sample — naïve linear search would be O(N). A **sum-tree** (binary tree where each node stores the sum of its children's priorities) enables O(log N) priority update and O(log N) sampling:

```
         58          ← root (total priority sum)
        /    \
      29       29
     /  \     /  \
   13   16   12   17
  / \  / \  / \  / \
 3  10 12 4 1  11 10 7   ← leaf nodes (individual transition priorities)
```

**Sampling:** Draw uniform random `z ∈ [0, total_priority]`, traverse tree to find the leaf whose cumulative sum contains `z` → O(log N).

```python
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)   # internal + leaf nodes
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)             # O(log N) propagation

    def get(self, z):
        idx = self._retrieve(0, z)               # O(log N) traversal
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, z):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        return self._retrieve(left, z) if z <= self.tree[left] \
               else self._retrieve(right, z - self.tree[left])
```

---

### Combined: Double DQN + PER

The full training loop combining both improvements:

```python
def train_step(self):
    # Sample prioritized minibatch from PER buffer
    indices, transitions, weights = self.per_buffer.sample(
        batch_size=64, beta=self.beta_schedule.value()
    )
    states, actions, rewards, next_states, dones = zip(*transitions)

    # Double DQN target computation
    with torch.no_grad():
        next_actions = self.online_net(next_states).argmax(dim=1)
        next_q = self.target_net(next_states).gather(
            1, next_actions.unsqueeze(1)
        ).squeeze()
        targets = rewards + self.gamma * next_q * (1 - dones)

    # Current Q-values
    q_values = self.online_net(states).gather(
        1, actions.unsqueeze(1)
    ).squeeze()

    # Importance-weighted loss
    td_errors = (q_values - targets).abs().detach()
    loss = (weights * F.smooth_l1_loss(q_values, targets, reduction='none')).mean()

    # Update priorities in sum-tree
    self.per_buffer.update_priorities(indices, td_errors + 1e-6)

    # Gradient step
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
    self.optimizer.step()

    # Periodically sync target network
    if self.steps % self.target_update_freq == 0:
        self.target_net.load_state_dict(self.online_net.state_dict())
```

**Gradient clipping** (`max_norm=10.0`) prevents exploding gradients — particularly important when PER produces highly variable loss magnitudes due to priority weighting.

---

## Network Architecture

```python
class DQNNavigator(nn.Module):
    def __init__(self, grid_size=7, state_dim=5, n_actions=4):
        super().__init__()

        # CNN encoder for occupancy grid
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 7×7 → 7×7×32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 7×7×64
            nn.ReLU(),
            nn.Flatten(),                                  # 64×7×7 = 3136
            nn.Linear(3136, 256),
            nn.ReLU(),
        )

        # MLP head: fused grid features + state vector
        self.mlp = nn.Sequential(
            nn.Linear(256 + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, grid, state):
        grid_features = self.cnn(grid.unsqueeze(1).float())
        fused = torch.cat([grid_features, state], dim=-1)
        return self.mlp(fused)
```

---

## Training Pipeline

| Hyperparameter | Value |
|---|---|
| Replay buffer size | 50,000 |
| Minibatch size | 64 |
| Discount factor γ | 0.99 |
| Learning rate | 1×10⁻⁴ |
| Target network update | Every 1,000 steps |
| ε (exploration) | 1.0 → 0.05, decay over 50K steps |
| PER α (prioritization) | 0.6 |
| PER β₀ (IS correction) | 0.4 → 1.0 over training |
| PER ε (min priority) | 1×10⁻⁶ |
| Optimizer | Adam |
| Gradient clip norm | 10.0 |

**ε-greedy schedule:** Linear decay from 1.0 (fully random) to 0.05 (mostly greedy) over the first 50K steps. This ensures the agent explores the environment thoroughly before committing to a learned policy — especially important in navigation where sparse goal rewards require exploration to discover.

---

## Environments

### Custom Grid World

- Configurable grid size (default 20×20)
- Randomly generated obstacle layouts per episode
- Randomized start and goal positions
- Egocentric 7×7 occupancy grid observation window
- Episode terminates on goal reached, collision, or step limit

### Isaac Sim *(In Progress)*

Active migration to NVIDIA Isaac Sim for:
- Photorealistic RGB-D observations replacing symbolic grid
- Physics-accurate robot dynamics (differential drive base)
- Continuous velocity action space
- Sim-to-real transfer evaluation
- Dynamic obstacle support (moving pedestrians, doors)

---

## Current Status

| Component | Status |
|---|---|
| Custom grid world environment | ✅ Complete |
| Vanilla DQN baseline | ✅ Complete |
| Double DQN | ✅ Complete |
| Sum-tree PER buffer | ✅ Complete |
| Double DQN + PER combined | ✅ Complete |
| Discrete action space | ✅ Complete |
| Egocentric occupancy grid observation | ✅ Complete |
| Isaac Sim environment integration | 🔄 In Progress |
| Continuous action space (binned) | 🔄 In Progress |
| Quantitative benchmarking (success rate, reward curves) | ⏳ Pending |
| Sim-to-real transfer evaluation | ⏳ Pending |

---

## Future Work

- **Isaac Sim full integration** — replace symbolic grid with RGB-D observations, add physics-accurate robot dynamics
- **SAC / DDPG for continuous control** — replace discretized continuous actions with a proper actor-critic method for smoother navigation in Isaac Sim
- **Dueling DQN** — separate value and advantage streams for more stable Q-value estimation in navigation tasks with many similar-value states
- **Multi-step returns (n-step TD)** — reduce bias in sparse-reward environments by propagating rewards over n steps instead of 1
- **Curriculum learning** — start with simple maps, progressively increase obstacle density and map size as the agent improves

---

## Installation

```bash
git clone https://github.com/sarthak-talwadkar/Autonomous-Navigation-DQN.git
cd Autonomous-Navigation-DQN
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch ≥ 1.10, NumPy, Matplotlib, OpenCV, Gymnasium

---

## Usage

### Train on custom grid world

```bash
python train.py \
    --env grid \
    --grid-size 20 \
    --agent double_dqn_per \
    --episodes 2000 \
    --buffer-size 50000 \
    --save-dir checkpoints/
```

### Evaluate trained agent

```bash
python evaluate.py \
    --weights checkpoints/best.pt \
    --env grid \
    --episodes 100 \
    --render
```

### Compare DQN variants

```bash
python benchmark.py \
    --agents vanilla_dqn double_dqn double_dqn_per \
    --episodes 2000 \
    --trials 5 \
    --plot
```

---

## Project Structure

```
Autonomous-Navigation-DQN/
├── agents/
│   ├── dqn.py              # Vanilla DQN
│   ├── double_dqn.py       # Double DQN
│   └── double_dqn_per.py   # Double DQN + PER (full agent)
├── memory/
│   ├── replay_buffer.py    # Uniform replay buffer
│   ├── sum_tree.py         # Sum-tree data structure
│   └── per_buffer.py       # Prioritized Experience Replay buffer
├── models/
│   └── dqn_network.py      # CNN + MLP Q-network
├── envs/
│   ├── grid_world.py       # Custom grid world environment
│   └── isaac_sim.py        # Isaac Sim interface (in progress)
├── utils/
│   ├── schedules.py        # ε-greedy + β annealing schedules
│   └── visualize.py        # Q-value landscape, reward curves
├── train.py                # Training entry point
├── evaluate.py             # Evaluation + rendering
├── benchmark.py            # Multi-agent comparison
├── requirements.txt
└── README.md
```

---

## References

- Mnih, V. et al. *"Human-level control through deep reinforcement learning."* Nature 2015. [[Paper]](https://www.nature.com/articles/nature14236) *(DQN)*
- van Hasselt, H. et al. *"Deep Reinforcement Learning with Double Q-learning."* AAAI 2016. [[Paper]](https://arxiv.org/abs/1509.06461) *(Double DQN)*
- Schaul, T. et al. *"Prioritized Experience Replay."* ICLR 2016. [[Paper]](https://arxiv.org/abs/1511.05952) *(PER)*
- Wang, Z. et al. *"Dueling Network Architectures for Deep Reinforcement Learning."* ICML 2016. [[Paper]](https://arxiv.org/abs/1511.06581) *(future work)*

---

## Author

**Sarthak Talwadkar**
MS Robotics, Northeastern University
[LinkedIn](https://linkedin.com/in/sarthak-talwadkar) · [GitHub](https://github.com/sarthak-talwadkar)
