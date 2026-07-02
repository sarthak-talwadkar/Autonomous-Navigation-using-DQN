"""DQN agent variants and a small factory."""

from agents.dqn import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.double_dqn_per import DoubleDQNPERAgent

AGENT_REGISTRY = {
    "vanilla_dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
    "double_dqn_per": DoubleDQNPERAgent,
}


def make_agent(name: str, **kwargs):
    """Instantiate an agent by CLI name (see :data:`AGENT_REGISTRY`)."""
    try:
        cls = AGENT_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown agent '{name}'; choose from {sorted(AGENT_REGISTRY)}"
        ) from None
    return cls(**kwargs)


__all__ = [
    "DQNAgent",
    "DoubleDQNAgent",
    "DoubleDQNPERAgent",
    "AGENT_REGISTRY",
    "make_agent",
]
