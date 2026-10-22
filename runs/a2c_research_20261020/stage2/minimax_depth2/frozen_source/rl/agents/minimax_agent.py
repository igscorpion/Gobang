"""Minimax 五子棋 Agent：Negamax 搜索 + Alpha-Beta 剪枝。

棋盘使用相对当前行动方的表示（1=己方，-1=对手），与 BaseAgent 接口一致。
候选动作采用启发式局部筛选以控制 9x9 分支因子；立即成五与阻止对手
下一手成五的动作不受候选数量上限限制。局面估值复用 Q-Learning/Greedy 的
13 维棋型特征，但通过递归搜索建模双方后续的最优应对。
"""

from __future__ import annotations

from math import inf
from rl.environment.rules import relative_legal_actions, rule_mode, winning_move
from time import perf_counter

from rl.agents.base_agent import BaseAgent
from rl.agents.greedy_agent import GREEDY_WEIGHTS
from rl.agents.q_learning import _features


class MinimaxAgent(BaseAgent):
    """Negamax + Alpha-Beta 棋力基线。

    Args:
        depth: 搜索总深度，单位为半回合（ply），至少为 1。
        candidate_limit: 普通局面每层最多搜索的候选动作；关键胜/防守点不截断。
        neighborhood: 普通候选动作距已有棋子的最大 Chebyshev 距离。
        use_alpha_beta: 是否启用 Alpha-Beta 剪枝；关闭时用于正确性对照。
    """

    MATE_SCORE = 1_000_000_000.0

    def __init__(
        self,
        board_size: int = 9,
        win_count: int = 5,
        depth: int = 2,
        candidate_limit: int = 8,
        neighborhood: int = 2,
        use_alpha_beta: bool = True,
        rules=None,
    ):
        if depth < 1:
            raise ValueError("depth 必须至少为 1")
        if candidate_limit < 1:
            raise ValueError("candidate_limit 必须至少为 1")
        if neighborhood < 1:
            raise ValueError("neighborhood 必须至少为 1")
        self.rules = rule_mode(rules)
        self.board_size = board_size
        self.win_count = win_count
        self.depth = depth
        self.candidate_limit = candidate_limit
        self.neighborhood = neighborhood
        self.use_alpha_beta = use_alpha_beta
        self.last_search_nodes = 0
        self.last_search_seconds = 0.0
        self.last_search_value = 0.0
        self.last_search_depth = depth

    def select_action(self, state, legal_actions):
        """在传入的合法动作中搜索并返回最优动作。"""
        if not legal_actions:
            raise ValueError("没有合法动作可选择")
        board = [list(row) for row in state]
        legal = [
            int(action)
            for action in legal_actions
            if 0 <= int(action) < self.board_size * self.board_size
            and board[int(action) // self.board_size][int(action) % self.board_size]
            == 0
        ]
        if not legal:
            raise ValueError("没有与当前棋盘匹配的合法动作")

        started = perf_counter()
        self.last_search_nodes = 0
        self.last_search_depth = self.depth
        score, action = self._search(board, legal, self.depth, -inf, inf, 0)
        self.last_search_seconds = perf_counter() - started
        self.last_search_value = score
        if action is None:
            raise ValueError("搜索未能找到合法动作")
        return action

    def _search(self, board, legal_actions, depth, alpha, beta, ply):
        self.last_search_nodes += 1
        if depth <= 0 or not legal_actions:
            return self._evaluate(board), None

        candidates = self._ordered_candidates(board, legal_actions)
        if not candidates:
            return 0.0, None

        best_score = -inf
        best_action = None
        for action in candidates:
            row, col = divmod(action, self.board_size)
            board[row][col] = 1

            if self._is_winning_move(board, row, col, 1):
                score = self.MATE_SCORE - ply
            elif len(legal_actions) == 1:
                score = 0.0
            else:
                next_board = [[-cell for cell in line] for line in board]
                next_legal = relative_legal_actions(next_board, self.rules)
                child_score, _ = self._search(
                    next_board,
                    next_legal,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                )
                score = -child_score

            board[row][col] = 0
            if score > best_score:
                best_score = score
                best_action = action
            if score > alpha:
                alpha = score
            if self.use_alpha_beta and alpha >= beta:
                break

        return best_score, best_action

    def _ordered_candidates(self, board, legal_actions):
        """优先直接胜/防守，其余只搜索附近的高棋型分动作。

        直接战术点在全体合法动作中检查，故不受邻域/数量剪枝影响。
        普通动作筛选属于启发式剪枝，可能遗漏远处或组合型关键手。
        """
        legal = list(legal_actions)
        own_wins = [a for a in legal if self._wins_after(board, a, 1)]
        if own_wins:
            return own_wins

        opponent_wins = [a for a in legal if self._wins_after(board, a, -1)]
        if opponent_wins:
            # 若对方有一个或多个下一手胜点，搜索所有对应堵点，不被 branch cap 截断。
            return opponent_wins

        stones = [
            (r, c)
            for r, line in enumerate(board)
            for c, value in enumerate(line)
            if value
        ]
        if not stones:
            center = (self.board_size // 2) * self.board_size + self.board_size // 2
            return [center] if center in legal else legal[: self.candidate_limit]

        nearby = []
        for action in legal:
            row, col = divmod(action, self.board_size)
            if any(
                max(abs(row - sr), abs(col - sc)) <= self.neighborhood
                for sr, sc in stones
            ):
                nearby.append(action)
        pool = nearby or legal
        ranked = sorted(
            pool,
            key=lambda action: self._move_score(board, action),
            reverse=True,
        )
        return ranked[: self.candidate_limit]

    def _evaluate(self, board):
        """静态估值：己方最佳单步棋型潜力减去对手潜力。"""
        legal = [
            r * self.board_size + c
            for r in range(self.board_size)
            for c in range(self.board_size)
            if board[r][c] == 0
        ]
        if not legal:
            return 0.0
        own = max(self._move_score(board, action) for action in legal)
        reversed_board = [[-cell for cell in line] for line in board]
        opponent = max(self._move_score(reversed_board, action) for action in legal)
        return own - opponent

    def _move_score(self, board, action):
        features = _features(board, action, self.board_size, self.win_count)
        return sum(
            GREEDY_WEIGHTS.get(name, 0.0) * value for name, value in features.items()
        )

    def _wins_after(self, board, action, player):
        row, col = divmod(action, self.board_size)
        board[row][col] = player
        won = self._is_winning_move(board, row, col, player)
        board[row][col] = 0
        return won

    def _is_winning_move(self, board, row, col, player):
        color = 1 if sum(v != 0 for row in board for v in row) % 2 else -1
        # board already contains the hypothetical move.
        actual = color * player
        absolute = [[v * color for v in row] for row in board]
        return winning_move(absolute, row, col, actual, self.rules, self.win_count)

    def update(self, state, action, reward, next_state, done):
        """Minimax 是无训练参数的搜索策略，不执行逐步更新。"""
        return None
