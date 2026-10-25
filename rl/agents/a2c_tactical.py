"""Inference-only A2C with an explicit, rule-aware tactical action filter.

The network and its learned weights are unchanged. Tactical choices are a
separate hybrid policy and must not be logged as unmodified on-policy A2C
training actions.
"""

from __future__ import annotations

from rl.agents.a2c import A2CAgent
from rl.agents.tactics import TacticalConfig, TacticalEngine


class TacticalA2CAgent(A2CAgent):
    """Use A2C to choose among the tactical engine's legal candidates."""

    def __init__(
        self,
        device=None,
        seed=None,
        training=False,
        epsilon=0.0,
        board_size=9,
        *,
        tactical_config: TacticalConfig | None = None,
        rules=None,
    ):
        if training:
            raise ValueError(
                "TacticalA2CAgent is inference-only; tactical overrides require "
                "a separately specified behavior policy before A2C training."
            )
        super().__init__(device, seed, False, epsilon, board_size)
        self.tactics = TacticalEngine(tactical_config, rules, board_size)
        self.last_tactical_decision = None

    def select_action(self, state, legal_actions):
        if self.training:
            raise ValueError("TacticalA2CAgent cannot be used as the A2C learning policy")
        self.last_tactical_decision = None
        decision = self.tactics.analyze(state, legal_actions)
        self.last_tactical_decision = decision
        if not decision.actions:
            raise ValueError("No legal tactical candidates")
        if len(decision.actions) == 1:
            return decision.actions[0]
        return super().select_action(state, decision.actions)
