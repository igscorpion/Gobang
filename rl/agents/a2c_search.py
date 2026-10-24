"""Inference-only A2C + tactical filter + bounded general shallow search.

This is a hybrid search policy, not a stronger trained A2C checkpoint.  The
root candidate set is exactly the existing TacticalEngine's candidate set.
Below the root, ordinary moves from *both* players are searched; immediate
wins and complete one-ply defences are never removed by the ordinary cap.

Search scores are heuristic estimates, never tactical certificates.  An
unfinished iterative-deepening iteration is discarded in its entirety to
avoid preferring the subset of root moves that happened to finish first.
The node budget limits visited search positions, not wall-clock time or the
cost of tactical preprocessing, legal-move generation, or static evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite, tanh
from time import perf_counter

import torch

from rl.agents.a2c import A2CAgent
from rl.agents.greedy_agent import GREEDY_WEIGHTS
from rl.agents.q_learning import _features
from rl.agents.tactics import TacticalConfig, TacticalEngine, winning_actions
from rl.environment.rules import legal_actions, winning_move
from rl.environment.state import state_to_tensor4, unpack


@dataclass(frozen=True)
class SearchConfig:
    """General-search budget, independent of TacticalConfig's VCF budget.

    depth counts plies including the root move.  candidate_limit applies only
    to ordinary non-root moves; a complete tied heuristic tier is retained.
    learned_value_weight=0 intentionally avoids trusting the old uncalibrated
    value head.  At nonzero weight, its clipped [-1, 1] output is blended with
    tanh(handcrafted_score / static_scale), in the player-to-move perspective.
    """

    depth: int = 2
    max_nodes: int = 2000
    candidate_limit: int = 8
    neighborhood: int = 2
    learned_value_weight: float = 0.0
    static_scale: float = 1000.0

    def __post_init__(self):
        if any(type(value) is not int for value in (
            self.depth, self.max_nodes, self.candidate_limit, self.neighborhood
        )):
            raise ValueError("depth, node budget, candidate limit and neighborhood must be integers")
        if self.depth not in (1, 2, 3):
            raise ValueError("depth must be 1, 2, or 3 plies")
        if self.max_nodes < 0:
            raise ValueError("max_nodes must be nonnegative")
        if self.candidate_limit < 1 or self.neighborhood < 1:
            raise ValueError("candidate_limit and neighborhood must be positive")
        if not 0.0 <= self.learned_value_weight <= 1.0:
            raise ValueError("learned_value_weight must be between zero and one")
        if not isfinite(self.static_scale) or self.static_scale <= 0:
            raise ValueError("static_scale must be finite and positive")


class _BudgetExceeded(Exception):
    pass


def _placed(board, action, player):
    size = len(board)
    row, col = divmod(action, size)
    changed = list(board[row])
    changed[col] = player
    return board[:row] + (tuple(changed),) + board[row + 1 :]


class SearchA2CAgent(A2CAgent):
    """Unchanged A2C weights with explicitly labelled general-search inference.

    The ``select_action``/``load`` interface matches A2CAgent.  Inspect
    ``last_tactical_decision`` and ``last_search_details`` separately: only the
    former may contain a certificate.  No training or weight updates occur.
    """

    MATE_SCORE = 100.0  # Static/learned blend is bounded to [-1, 1].

    def __init__(
        self,
        device=None,
        seed=None,
        training=False,
        epsilon=0.0,
        board_size=9,
        *,
        search_config: SearchConfig | None = None,
        tactical_config: TacticalConfig | None = None,
        rules=None,
        policy_agent=None,
    ):
        if training:
            raise ValueError("SearchA2CAgent is an inference-only hybrid policy")
        if epsilon != 0.0:
            raise ValueError("SearchA2CAgent requires epsilon=0 for reproducible search")
        super().__init__(device, seed, False, 0.0, board_size)
        self.search_config = search_config or SearchConfig()
        if not isinstance(self.search_config, SearchConfig):
            raise TypeError("search_config must be SearchConfig")
        self.tactics = TacticalEngine(tactical_config, rules, board_size)
        self.rules = self.tactics.rules
        # An injected policy owns its encoder/checkpoint (e.g. extra feature
        # planes or a residual network).  It must expose deterministic
        # policy_value(relative_state) -> (board_size**2 logits, scalar value).
        self.policy_agent = policy_agent
        if policy_agent is not None and not callable(getattr(policy_agent, "policy_value", None)):
            raise TypeError("policy_agent must provide policy_value(state)")
        self.last_tactical_decision = None
        self.last_search_details = None
        self.last_search_nodes = 0
        self.last_search_depth = 0
        self.last_search_value = None
        self.last_search_seconds = 0.0
        self._network_cache = {}
        self._legal_cache = {}
        self._move_cache = {}
        self._budget_exhausted = False

    def _legal(self, board, player):
        key = board, player
        if key not in self._legal_cache:
            self._legal_cache[key] = tuple(legal_actions(board, player, self.rules))
        return self._legal_cache[key]

    def _network(self, board, player, last_move):
        key = board, player, last_move
        if key not in self._network_cache:
            relative = tuple(tuple(cell * player for cell in row) for row in board)
            logits, value = self._policy_value((relative, last_move))
            scores = tuple(float(v) for v in torch.as_tensor(logits).reshape(-1).cpu().tolist())
            if len(scores) != self.board_size**2:
                raise ValueError("policy_value must return one logit per board cell")
            estimate = float(torch.as_tensor(value).item())
            if not all(isfinite(v) for v in scores) or not isfinite(estimate):
                raise ValueError("A2C network produced nonfinite logits or value")
            self._network_cache[key] = scores, max(-1.0, min(1.0, estimate))
        return self._network_cache[key]

    def _policy_value(self, state):
        """Override/inject this hook when an experiment uses another encoder."""
        with torch.inference_mode():
            if self.policy_agent is not None:
                return self.policy_agent.policy_value(state)
            x = state_to_tensor4(state, self.board_size).unsqueeze(0).to(self.device)
            logits, value = self.net(x)
            return logits[0], value[0]

    def _move_score(self, board, action, player):
        key = board, action, player
        if key not in self._move_cache:
            relative = tuple(tuple(cell * player for cell in row) for row in board)
            features = _features(relative, action, self.board_size, 5)
            # These are ordering/evaluation heuristics, not rule certificates.
            # A black forbidden point is neither searched as a move nor valued
            # as a threat that white needs to block at this square.
            opponent_can_play_here = action in self._legal(board, -player)
            self._move_cache[key] = sum(
                GREEDY_WEIGHTS.get(name, 0.0) * value
                for name, value in features.items()
                if opponent_can_play_here or not name.startswith("opp_")
            )
        return self._move_cache[key]

    def _evaluate(self, board, player, last_move):
        legal = self._legal(board, player)
        other_legal = self._legal(board, -player)
        own = max((self._move_score(board, a, player) for a in legal), default=0.0)
        other = max((self._move_score(board, a, -player) for a in other_legal), default=0.0)
        handcrafted = tanh((own - other) / self.search_config.static_scale)
        weight = self.search_config.learned_value_weight
        if not weight:
            return handcrafted
        _, learned = self._network(board, player, last_move)
        return (1.0 - weight) * handcrafted + weight * learned

    def _ordered_candidates(self, board, player, last_move, legal):
        """All immediate wins/complete blocks, or general nearby candidates.

        In particular this is not a VCF-only expansion: quiet counterplay from
        the opponent is evaluated too.  Ordinary pruning remains heuristic and
        may miss a distant preparation move, so the result is never a proof.
        """
        immediate = set(winning_actions(board, player, self.rules))
        own_wins = tuple(a for a in legal if a in immediate)
        if own_wins:
            return own_wins
        enemy_wins = winning_actions(board, -player, self.rules)
        if enemy_wins:
            remaining = {
                a: winning_actions(_placed(board, a, player), -player, self.rules)
                for a in legal
            }
            complete_blocks = tuple(a for a in legal if not remaining[a])
            if complete_blocks:
                return complete_blocks
            minimum = min(len(wins) for wins in remaining.values())
            return tuple(a for a in legal if len(remaining[a]) == minimum)

        stones = tuple(
            (r, c) for r, row in enumerate(board) for c, value in enumerate(row) if value
        )
        nearby = tuple(
            a for a in legal
            if any(max(abs(a // self.board_size - r), abs(a % self.board_size - c))
                   <= self.search_config.neighborhood for r, c in stones)
        )
        pool = nearby or legal

        def rank(action):
            row, col = divmod(action, self.board_size)
            neighbors = sum(max(abs(row-r), abs(col-c)) <= 2 for r, c in stones)
            center = (2 * row-self.board_size+1)**2 + (2 * col-self.board_size+1)**2
            return self._move_score(board, action, player), neighbors, -center

        ranks = {a: rank(a) for a in pool}
        ordered = sorted(pool, key=lambda a: (ranks[a], -a), reverse=True)
        if len(ordered) > self.search_config.candidate_limit:
            cutoff = ranks[ordered[self.search_config.candidate_limit-1]]
            ordered = [a for a in ordered if ranks[a] >= cutoff]
        logits, _ = self._network(board, player, last_move)
        return tuple(sorted(ordered, key=lambda a: (ranks[a], logits[a], -a), reverse=True))

    def _negamax(self, board, player, last_move, depth, alpha, beta, ply):
        if self.last_search_nodes >= self.search_config.max_nodes:
            self._budget_exhausted = True
            raise _BudgetExceeded
        self.last_search_nodes += 1
        row, col = divmod(last_move, self.board_size)
        if winning_move(board, row, col, -player, self.rules):
            return -self.MATE_SCORE + 0.01 * ply
        legal = self._legal(board, player)
        if not legal:
            full = all(cell != 0 for line in board for cell in line)
            return 0.0 if full else -self.MATE_SCORE + 0.01 * ply
        if depth == 0:
            return self._evaluate(board, player, last_move)
        best = -inf
        for action in self._ordered_candidates(board, player, last_move, legal):
            value = -self._negamax(
                _placed(board, action, player), -player, action, depth-1,
                -beta, -alpha, ply+1,
            )
            best = max(best, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best

    def select_action(self, state, legal_actions):
        if self.training:
            raise ValueError("SearchA2CAgent cannot be used as the A2C learning policy")
        started = perf_counter()
        self.last_search_nodes = 0
        self.last_search_depth = 0
        self.last_search_value = None
        self._budget_exhausted = False
        self._network_cache.clear()
        self._legal_cache.clear()
        self._move_cache.clear()
        decision = self.tactics.analyze(state, legal_actions)
        self.last_tactical_decision = decision
        candidates = decision.actions
        if not candidates:
            raise ValueError("No legal tactical candidates")
        relative, last_move = unpack(state)
        player = 1 if sum(cell != 0 for row in relative for cell in row) % 2 == 0 else -1
        board = tuple(tuple(int(cell) * player for cell in row) for row in relative)
        logits, _ = self._network(board, player, last_move)
        ordered = tuple(sorted(candidates, key=lambda a: (logits[a], -a), reverse=True))
        chosen = ordered[0]
        scores = {}
        discarded_depth = None
        reason = "single_tactical_candidate" if len(candidates) == 1 else "tactical_certificate"
        if len(candidates) > 1 and not decision.proven:
            reason = "policy_fallback_no_complete_iteration"
            for depth in range(1, self.search_config.depth+1):
                current = {}
                try:
                    for action in ordered:
                        # Independent full windows give comparable root scores;
                        # alpha-beta only cuts inside each candidate's subtree.
                        current[action] = -self._negamax(
                            _placed(board, action, player), -player, action,
                            depth-1, -inf, inf, 1,
                        )
                except _BudgetExceeded:
                    discarded_depth = depth
                    break
                scores = current
                chosen = max(ordered, key=lambda a: (scores[a], logits[a], -a))
                self.last_search_depth = depth
                self.last_search_value = scores[chosen]
                reason = "completed_general_search"
        self.last_search_seconds = perf_counter() - started
        self.last_search_details = {
            "reason": reason,
            "root_candidates": candidates,
            "root_tactical_reason": decision.reason,
            "requested_depth": self.search_config.depth,
            "completed_depth": self.last_search_depth,
            "nodes": self.last_search_nodes,
            "max_nodes": self.search_config.max_nodes,
            "budget_exhausted": self._budget_exhausted,
            "discarded_incomplete_depth": discarded_depth,
            "completed_root_scores": scores,
            "selected_score": self.last_search_value,
            "score_is_proof": False,
            "learned_value_weight": self.search_config.learned_value_weight,
            "elapsed_seconds": self.last_search_seconds,
            "budget_scope": "visited general-search positions; not a wall-clock deadline",
        }
        # Avoid retaining a whole search tree while the game/opponent runs.
        self._network_cache.clear()
        self._legal_cache.clear()
        self._move_cache.clear()
        return chosen
