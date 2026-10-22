"""Focused invariants for the frozen-candidate ranking experiment."""
import unittest

from rl.agents.a2c_tactical import TacticalA2CAgent
from rl.agents.tactics import TacticalDecision
from rl.environment.gomoku import GomokuEnv
from rl.evaluation.a2c_candidate_study import CandidateAgent, rank_scores, ranked_action
from rl.training.common import choose, seed_everything


class CandidateStudyTests(unittest.TestCase):
    def test_score_ties_use_smallest_legal_action(self):
        self.assertEqual(rank_scores({2: 5, 7: 5, 4: 8}, (7, 2, 4)), [4, 2, 7])

    def test_hand_ranking_changes_only_ordinary_multicandidates(self):
        neural, hand = {2: 9, 7: 0}, {2: 0, 7: 9}
        for reason in ("quiet_candidates", "tactical_candidates", "defend_forcing_threat"):
            decision = TacticalDecision((2, 7), reason)
            self.assertEqual(ranked_action(decision, neural, hand, "a2c"), 2)
            self.assertEqual(ranked_action(decision, neural, hand, "hand"), 7)
        for reason in ("immediate_win", "block_immediate_win", "create_double_threat", "forced_win", "unavoidable_loss"):
            self.assertEqual(ranked_action(TacticalDecision((2, 7), reason), neural, hand, "hand"), 2)

    def test_never_adds_a_hand_preferred_action_outside_candidates(self):
        self.assertEqual(ranked_action(TacticalDecision((2,), "quiet_candidates"), {2: 1, 7: 0}, {2: 0, 7: 99}, "hand"), 2)

    def test_neural_branch_exactly_reproduces_existing_policy(self):
        seed_everything(22)
        original = TacticalA2CAgent(device="cpu", rules="freestyle")
        study = CandidateAgent(ranking="a2c", device="cpu", rules="freestyle")
        study.net.load_state_dict(original.net.state_dict())
        env = GomokuEnv()
        for action in (40, 30, 41, 31, 49, 32):
            env.step(action)
        self.assertEqual(choose(original, env), choose(study, env))
        self.assertEqual(original.last_tactical_decision.actions, study.last_tactical_decision.actions)


if __name__ == "__main__":
    unittest.main()
