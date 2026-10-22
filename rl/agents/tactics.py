"""Rule-aware, bounded tactical candidates for an inference-only hybrid policy.

Only immediate wins and continuous-four certificates are called proven wins.
Open threes, including broken threes, are useful *heuristics*, never compulsory
defences by themselves.  The policy still chooses among the returned actions.

Boards in the public helpers use absolute colours (black=1, white=-1).  Engine
observations, like A2C observations, are relative to the player to move.  Colour
is recovered from occupied-cell parity, exactly as in ``relative_legal_actions``.
No helper changes a caller's board.  All rules come from the shared rules module.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
import time
from typing import Iterable

from rl.environment.rules import DIRECTIONS, forbidden_reason, legal_actions, rule_mode

Board = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TacticalConfig:
    enabled: bool = True
    forcing_depth: int = 5
    max_nodes: int = 1500
    candidate_limit: int = 12
    cache_size: int = 4096

    def __post_init__(self):
        if self.forcing_depth < 0 or self.max_nodes < 0 or self.cache_size < 0:
            raise ValueError("depth, nodes and cache size must be nonnegative")
        if self.candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")


@dataclass(frozen=True)
class TacticalDecision:
    actions: tuple[int, ...]
    reason: str
    proven: bool = False
    details: dict = field(default_factory=dict)


def _board(value) -> Board:
    result = tuple(tuple(int(cell) for cell in row) for row in value)
    if not result or any(len(row) != len(result) for row in result):
        raise ValueError("board must be nonempty and square")
    if any(cell not in (-1, 0, 1) for row in result for cell in row):
        raise ValueError("board cells must be -1, 0 or 1")
    return result


def _put(board: Board, action: int, player: int) -> Board:
    row, col = divmod(action, len(board))
    changed = list(board[row])
    changed[col] = player
    return board[:row] + (tuple(changed),) + board[row + 1 :]


def _empty(board: Board) -> tuple[int, ...]:
    size = len(board)
    return tuple(r * size + c for r in range(size) for c in range(size) if not board[r][c])


def _win_if_place(board: Board, action: int, player: int, rules: str) -> bool:
    """Equivalent to shared winning_move after a placement, without copying.

    A black exact five takes precedence over forbidden moves in shared rules;
    therefore every endpoint returned here is also a legal winning endpoint.
    """
    size = len(board)
    row, col = divmod(action, size)
    if not (0 <= row < size and 0 <= col < size) or board[row][col]:
        return False
    lengths = []
    for dr, dc in DIRECTIONS:
        count = 1
        for sign in (-1, 1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < size and 0 <= c < size and board[r][c] == player:
                count += 1
                r, c = r + sign * dr, c + sign * dc
        lengths.append(count)
    return 5 in lengths if rules == "renju" and player == 1 else max(lengths) >= 5


@lru_cache(maxsize=8192)
def _legal(board: Board, player: int, rules: str) -> tuple[int, ...]:
    return tuple(legal_actions(board, player, rules))


@lru_cache(maxsize=16384)
def _winning(board: Board, player: int, rules: str) -> tuple[int, ...]:
    return tuple(action for action in _empty(board) if _win_if_place(board, action, player, rules))


def winning_actions(board, player=1, rules="freestyle") -> tuple[int, ...]:
    """All legal next-move wins on an absolute-colour board, without truncation."""
    if player not in (-1, 1):
        raise ValueError("player must be black=1 or white=-1")
    return _winning(_board(board), player, rule_mode(rules))


@lru_cache(maxsize=16384)
def _fours(board: Board, anchor: int, player: int, rules: str):
    """Distinct four-stone groups through anchor and their legal win endpoints."""
    size = len(board)
    row, col = divmod(anchor, size)
    groups = {}
    for dr, dc in DIRECTIONS:
        for offset in range(-4, 1):
            cells = tuple((row + (offset + k) * dr, col + (offset + k) * dc) for k in range(5))
            if not all(0 <= r < size and 0 <= c < size for r, c in cells):
                continue
            stones = tuple(sorted(r * size + c for r, c in cells if board[r][c] == player))
            empties = tuple(r * size + c for r, c in cells if board[r][c] == 0)
            if len(stones) == 4 and len(empties) == 1 and anchor in stones:
                endpoint = empties[0]
                if _win_if_place(board, endpoint, player, rules):
                    groups.setdefault(stones, set()).add(endpoint)
    return tuple(sorted((stones, tuple(sorted(endpoints))) for stones, endpoints in groups.items()))


@lru_cache(maxsize=8192)
def _threes(board: Board, anchor: int, player: int, rules: str):
    """Real open threes require a legal extension to a straight two-ended four.

    This intentionally follows the shared Renju implementation, including its
    recursively checked extension legality. Edges, blocked endpoints, fake
    threes and black forbidden extensions do not count as real open threes.
    """
    size = len(board)
    row, col = divmod(anchor, size)
    groups = {}
    for dr, dc in DIRECTIONS:
        for delta in range(-3, 4):
            if not delta:
                continue
            r, c = row + delta * dr, col + delta * dc
            if not (0 <= r < size and 0 <= c < size) or board[r][c]:
                continue
            extension = r * size + c
            if _win_if_place(board, extension, player, rules):
                continue
            if forbidden_reason(board, extension, player, rules):
                continue
            extended = _put(board, extension, player)
            for stones, endpoints in _fours(extended, anchor, player, rules):
                if extension in stones and len(endpoints) >= 2:
                    three = tuple(stone for stone in stones if stone != extension)
                    groups.setdefault(three, set()).add(extension)
    return tuple(sorted((stones, tuple(sorted(extensions))) for stones, extensions in groups.items()))


def _quick_features(board: Board, action: int, player: int, rules: str) -> dict:
    after = _put(board, action, player)
    fours = _fours(after, action, player, rules)
    endpoints = tuple(sorted({endpoint for _, ends in fours for endpoint in ends}))
    return {
        "winning": _win_if_place(board, action, player, rules),
        "winning_actions": endpoints,
        "four_threats": len(fours),
        "open_fours": sum(len(ends) >= 2 for _, ends in fours),
        "double_threat": len(endpoints) >= 2,
        "four_groups": fours,
    }


def analyze_move(board, action, player=1, rules="freestyle") -> dict:
    """Explain one move on an absolute board; illegal moves have zero features.

    Stable numeric keys include ``four_threats``, ``open_fours``,
    ``real_open_threes`` and ``broken_threes``. A real broken three is a subset
    of real open threes. ``double_threat`` means two distinct legal next-move
    winning endpoints, not just two windows containing the same four stones.
    """
    board, rules, action = _board(board), rule_mode(rules), int(action)
    if player not in (-1, 1):
        raise ValueError("player must be black=1 or white=-1")
    size = len(board)
    legal = 0 <= action < size * size and not board[action // size][action % size]
    reason = forbidden_reason(board, action, player, rules) if legal else "occupied_or_outside"
    legal = legal and reason is None
    result = {
        "legal": legal,
        "forbidden_reason": reason,
        "winning": False,
        "winning_actions": (),
        "created_winning_actions": (),
        "four_threats": 0,
        "open_fours": 0,
        "real_open_threes": 0,
        "broken_threes": 0,
        "double_threat": False,
        "four_three": False,
        "double_three": False,
        "four_groups": (),
        "open_three_groups": (),
    }
    if not legal:
        return result
    result.update(_quick_features(board, action, player, rules))
    after = _put(board, action, player)
    result["created_winning_actions"] = result["winning_actions"]
    result["winning_actions"] = _winning(after, player, rules)
    result["double_threat"] = len(result["winning_actions"]) >= 2
    threes = _threes(after, action, player, rules)
    broken = 0
    for stones, _ in threes:
        rows, cols = tuple(stone // size for stone in stones), tuple(stone % size for stone in stones)
        if max(max(rows) - min(rows), max(cols) - min(cols)) > 2:
            broken += 1
    result.update(
        real_open_threes=len(threes), broken_threes=broken,
        four_three=bool(result["four_threats"] and threes),
        double_three=len(threes) >= 2, open_three_groups=threes,
    )
    return result


@dataclass
class _Budget:
    limit: int
    nodes: int = 0
    exhausted: bool = False

    def take(self) -> bool:
        if self.nodes >= self.limit:
            self.exhausted = True
            return False
        self.nodes += 1
        return True


class TacticalEngine:
    def __init__(self, config=None, rules=None, board_size=9):
        self.config = config if config is not None else TacticalConfig()
        if not isinstance(self.config, TacticalConfig):
            raise TypeError("config must be TacticalConfig")
        self.rules = rule_mode(rules)
        self.board_size = int(board_size)
        if self.board_size < 1:
            raise ValueError("board_size must be positive")
        self._cache = OrderedDict()
        self.cache_hits = 0

    @staticmethod
    def _geometry(board: Board, action: int) -> tuple[int, int]:
        size = len(board)
        row, col = divmod(action, size)
        neighbours = sum(
            bool(board[r][c])
            for r in range(max(0, row - 2), min(size, row + 3))
            for c in range(max(0, col - 2), min(size, col + 3))
        )
        centre = (2 * row - size + 1) ** 2 + (2 * col - size + 1) ** 2
        return neighbours, -centre

    def _rank(self, board, legal, ours, theirs):
        def score(action):
            own, enemy = ours.get(action, {}), theirs.get(action, {})
            # Defence and attack define broad priorities, rather than imposing
            # a hand-written final action. A2C resolves choices within a tier.
            return (
                int(enemy.get("double_threat", False)) * 8
                + int(own.get("four_three", False)) * 6
                + int(enemy.get("four_three", False)) * 5,
                int(own.get("four_threats", 0)) * 4
                + int(enemy.get("double_three", False)) * 3
                + int(own.get("double_three", False)) * 3,
                int(own.get("real_open_threes", 0)) * 2
                + int(enemy.get("real_open_threes", 0)),
                *self._geometry(board, action),
            )
        return sorted(legal, key=lambda action: (score(action), -action), reverse=True), score

    def _shortlist(self, ordered, score):
        if len(ordered) <= self.config.candidate_limit:
            return list(ordered)
        cutoff = score(ordered[self.config.candidate_limit - 1])
        # Keep a complete tied tier, preserving reflection/rotation symmetry.
        return [action for action in ordered if score(action) >= cutoff]

    def _vcf(self, board: Board, attacker: int, depth: int, budget: _Budget):
        """Return one continuous-four proof line, or None (never a loss proof).

        Opponent immediate wins are checked before accepting an attack. For a
        single winning endpoint *every* non-blocking reply loses next turn;
        its sole legal blocking reply is recursively covered. Two endpoints
        cover all legal non-winning replies. Attack truncation only loses
        completeness; it cannot turn an unproved line into a certificate.
        """
        if depth < 1 or not budget.take():
            return None
        immediate = _winning(board, attacker, self.rules)
        if immediate:
            return (immediate[0],)
        if depth < 3:
            return None
        attacks = []
        for action in _legal(board, attacker, self.rules):
            if not budget.take():
                return None
            features = _quick_features(board, action, attacker, self.rules)
            if features["winning_actions"]:
                attacks.append((len(features["winning_actions"]), action))
        for _, action in sorted(attacks, reverse=True)[: self.config.candidate_limit]:
            line = self._after_forcing_attack(board, action, attacker, depth, budget)
            if line is not None:
                return line
        return None

    def _after_forcing_attack(self, board, action, attacker, depth, budget):
        if depth < 3 or not budget.take():
            return None
        after = _put(board, action, attacker)
        if _winning(after, -attacker, self.rules):
            return None  # A defender wins immediately instead of blocking.
        endpoints = _winning(after, attacker, self.rules)
        if len(endpoints) >= 2:
            return (action, -1, endpoints[0])  # -1 denotes all defender replies.
        if not endpoints:
            return None
        defenders = _legal(after, -attacker, self.rules)
        if not defenders:
            # A full board is a draw; an opponent without a legal Renju move
            # loses according to GomokuEnv.step.
            return (action,) if _empty(after) else None
        block = endpoints[0]
        if block not in defenders:
            return (action, -1, block)
        defended = _put(after, block, -attacker)
        tail = self._vcf(defended, attacker, depth - 2, budget)
        return (action, block) + tail if tail is not None else None

    def _safe_from_double_reply(self, board, action, player, budget):
        """True/False/None: no reply that wins by a legal double endpoint.

        Single-four counterattacks are checked using their forced blocking
        reply; an opponent cannot ignore a direct winning threat to build a
        slower threat. None means the budget prevented a complete check.
        This is deliberately not a proof of safety against deeper tactics.
        """
        after = _put(board, action, player)
        own_wins = _winning(after, player, self.rules)
        replies = _legal(after, -player, self.rules)
        if not replies:
            return True
        if own_wins:
            replies = tuple(reply for reply in replies if all(reply == end for end in own_wins))
        for reply in replies:
            if not budget.take():
                return None
            if _win_if_place(after, reply, -player, self.rules):
                return False
            features = _quick_features(after, reply, -player, self.rules)
            if features["double_threat"]:
                replied = _put(after, reply, -player)
                if not _winning(replied, player, self.rules):
                    return False
        return True

    def analyze(self, state, legal_actions: Iterable[int]) -> TacticalDecision:
        started = time.perf_counter()
        # Unpack the same (relative board, last_action) shape as A2C, without
        # importing torch merely to use the state module's unpack function.
        if len(state) == 2 and (state[1] is None or isinstance(state[1], int)):
            state = state[0]
        relative = _board(state)
        if len(relative) != self.board_size:
            raise ValueError("observation has wrong board shape")
        occupied = sum(cell != 0 for row in relative for cell in row)
        player = 1 if occupied % 2 == 0 else -1
        board = tuple(tuple(cell * player for cell in row) for row in relative)
        supplied = {int(action) for action in legal_actions}
        legal = tuple(action for action in _legal(board, player, self.rules) if action in supplied)
        key = self.rules, self.config, board, player, legal
        if key in self._cache:
            cached = self._cache.pop(key)
            self._cache[key] = cached
            self.cache_hits += 1
            return cached
        budget = _Budget(self.config.max_nodes)
        info = {"rules": self.rules, "absolute_player": player, "legal_count": len(legal), "supplied_legal_count": len(supplied)}

        def finish(actions, reason, proven=False, **details):
            decision = TacticalDecision(tuple(sorted(set(actions))), reason, proven, {
                **info, **details, "nodes": budget.nodes,
                "budget_exhausted": budget.exhausted,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            })
            if self.config.cache_size:
                self._cache[key] = decision
                while len(self._cache) > self.config.cache_size:
                    self._cache.popitem(last=False)
            return decision

        if not legal:
            return finish((), "no_legal_actions")
        if not self.config.enabled:
            return finish(legal, "disabled")
        own_immediate = tuple(action for action in _winning(board, player, self.rules) if action in supplied)
        if own_immediate:
            return finish(own_immediate, "immediate_win", True, proof_scope="legal win on this move")
        enemy_immediate = _winning(board, -player, self.rules)
        if enemy_immediate:
            # This exact, all-root-move check is outside every search budget
            # and candidate cap. Multiple threats are never called defended
            # unless the actual post-move board has no enemy winning endpoint.
            remaining = {
                action: _winning(_put(board, action, player), -player, self.rules)
                for action in legal
            }
            defences = tuple(action for action, wins in remaining.items() if not wins)
            if defences:
                return finish(defences, "block_immediate_win", enemy_winning_actions=enemy_immediate,
                              proof_scope="removes every legal next-move win; no deeper safety claim")
            minimum = min(map(len, remaining.values()))
            return finish(tuple(action for action, wins in remaining.items() if len(wins) == minimum),
                          "unavoidable_loss", True, enemy_winning_actions=enemy_immediate,
                          remaining_winning_actions={str(a): w for a, w in remaining.items()},
                          proof_scope="every allowed root move permits an opponent win next move")

        ours = {action: _quick_features(board, action, player, self.rules) for action in legal}
        double = []
        for action in legal:
            if ours[action]["double_threat"]:
                after = _put(board, action, player)
                if not _winning(after, -player, self.rules):
                    double.append(action)
        if double:
            return finish(double, "create_double_threat", True,
                          winning_endpoints={str(a): ours[a]["winning_actions"] for a in double},
                          proof_scope="two legal winning endpoints; every opponent reply loses, unless already terminal")

        enemy_legal = _legal(board, -player, self.rules)
        theirs = {action: _quick_features(board, action, -player, self.rules) for action in enemy_legal}
        enemy_double = tuple(action for action in enemy_legal if theirs[action]["double_threat"])
        ordered, score = self._rank(board, legal, ours, theirs)
        # Expensive true-three legality is confined to geometrically relevant
        # positions plus all four-threat moves. This is heuristic coverage,
        # independent of the exact win/block checks above.
        local = set(self._shortlist(ordered, score))
        local.update(action for action in legal if ours[action]["four_threats"])
        local.update(action for action in legal if theirs.get(action, {}).get("four_threats"))
        for action in local:
            ours[action] = analyze_move(board, action, player, self.rules)
            if action in theirs:
                theirs[action] = analyze_move(board, action, -player, self.rules)
        ordered, score = self._rank(board, legal, ours, theirs)
        shortlist = self._shortlist(ordered, score)

        # Prove continuous-four attacks first. A sound forcing counterattack
        # takes precedence over defensive heuristics and open-three blocking.
        proofs = {}
        if self.config.forcing_depth >= 3:
            forcing = [action for action in ordered if ours[action]["winning_actions"]]
            for action in forcing[: self.config.candidate_limit]:
                line = self._after_forcing_attack(board, action, player, self.config.forcing_depth, budget)
                if line is not None:
                    proofs[str(action)] = line
            if proofs:
                return finish(tuple(int(action) for action in proofs), "forced_win", True,
                              proof_lines=proofs, proof_depth_plies=self.config.forcing_depth,
                              proof_scope="continuous legal fours; all defensive replies covered; -1 is any reply")

        safe, unknown, rejected = [], [], []
        # When a double-threat creation exists, extend the ordered search past
        # the nominal shortlist to find a defence or a sound four counterattack.
        to_check = ordered if enemy_double else shortlist
        for index, action in enumerate(to_check):
            safe_result = self._safe_from_double_reply(board, action, player, budget)
            if safe_result is True:
                safe.append(action)
            elif safe_result is False:
                rejected.append(action)
            else:
                unknown.append(action)
            if budget.exhausted:
                # Budget exhaustion does not grant the first unchecked move a
                # special preference. Keep the remaining reasonable shortlist
                # available to A2C, and disclose its unknown tactical status.
                unknown.extend(a for a in shortlist if a not in safe and a not in rejected and a not in unknown)
                break
            if (
                len(safe) >= self.config.candidate_limit
                and index + 1 < len(to_check)
                and score(to_check[index + 1]) < score(safe[self.config.candidate_limit - 1])
            ):
                break
        details = {
            "enemy_double_threat_moves": enemy_double,
            "rejected_double_threat_replies": tuple(rejected),
            "unchecked_candidates": tuple(unknown),
            "safety_scope": "opponent immediate double-endpoint replies only; deeper threats remain heuristic",
        }
        if safe:
            candidates = safe
            reason = "defend_forcing_threat" if enemy_double else "tactical_candidates"
        else:
            # Unknown is not a win, a loss, or a claimed safe move. Preserve
            # policy choice when the finite heuristic search cannot resolve it.
            candidates = unknown or shortlist
            reason = "tactical_candidates"
        has_patterns = any(
            ours.get(a, {}).get("four_threats")
            or ours.get(a, {}).get("real_open_threes")
            or theirs.get(a, {}).get("four_threats")
            or theirs.get(a, {}).get("real_open_threes")
            for a in candidates
        )
        if not enemy_double and not has_patterns:
            reason = "quiet_candidates"
        details["candidate_features"] = {
            str(action): {
                "own": {k: ours[action].get(k, 0) for k in ("four_threats", "real_open_threes", "broken_threes", "four_three", "double_three")},
                "opponent_if_played_here": {k: theirs.get(action, {}).get(k, 0) for k in ("four_threats", "real_open_threes", "broken_threes", "four_three", "double_three")},
            }
            for action in candidates
        }
        return finish(candidates, reason, **details)
