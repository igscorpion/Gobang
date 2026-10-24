"""Inference-only A2C with exact/proved tactical overrides.

This ablation deliberately removes the original engine's *heuristic* root
shortlist.  It keeps legal wins, complete one-move defence, proved double
winning endpoints and bounded VCF certificates.  It never calls ordinary
policy choices safe, and it does not modify, retrain or relabel the policy.
"""

from __future__ import annotations

from rl.agents.base_agent import BaseAgent
from rl.agents.tactics import TacticalDecision, TacticalEngine
from rl.environment.rules import relative_legal_actions, rule_mode
from rl.environment.state import unpack


PROVED_OVERRIDE_REASONS = frozenset((
    "immediate_win", "create_double_threat", "forced_win", "unavoidable_loss",
))


def uses_exact_override(decision):
    """A one-ply defence is exact but deliberately not a deeper safety proof."""
    if not decision.details.get("proof_scope"):
        return False
    if decision.reason == "block_immediate_win":
        return True
    return decision.reason in PROVED_OVERRIDE_REASONS and decision.proven


class GuardedA2CAgent(BaseAgent):
    """Compose a frozen standard/research policy with a conservative guard."""

    needs_last_move = True

    def __init__(self, policy, *, rules=None, tactical_config=None):
        if getattr(policy, "training", False):
            raise ValueError("Guarded wrapper is inference-only")
        self.policy = policy
        self.rules = rule_mode(rules if rules is not None else getattr(policy, "rules", None))
        spec = getattr(policy, "spec", None)
        board_size = spec.board_size if spec is not None else getattr(policy, "board_size", 9)
        self.tactics = TacticalEngine(tactical_config, self.rules, board_size)
        self.last_source_decision = None
        self.last_tactical_decision = None

    def select_action(self, state, legal_actions):
        board, _ = unpack(state)
        supplied = {int(action) for action in legal_actions}
        legal = tuple(action for action in relative_legal_actions(board, self.rules)
                      if action in supplied)
        if not legal:
            raise ValueError("No legal supplied action")
        source = self.tactics.analyze(state, legal)
        self.last_source_decision = source
        override = uses_exact_override(source)
        candidates = tuple(action for action in source.actions if action in legal) if override else legal
        if not candidates:
            raise ValueError("Exact tactical decision has no legal candidates")
        details = dict(source.details)
        details.update(source_reason=source.reason, source_actions=source.actions,
                       exact_override=override, policy_legal_count=len(legal),
                       selected_candidate_count=len(candidates))
        if not override:
            details.pop("proof_scope", None)
            details["selection_scope"] = "policy across all legal supplied actions; no safety proof"
        self.last_tactical_decision = TacticalDecision(
            candidates, source.reason if override else "guarded_policy_full_legal",
            source.proven if override else False, details,
        )
        return (candidates[0] if len(candidates) == 1 else
                self.policy.select_action(state, candidates))
