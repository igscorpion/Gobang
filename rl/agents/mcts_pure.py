"""纯 MCTS 基线：UCT + 随机走子模拟，不使用任何神经网络。

与学习无关的搜索基线：不依赖任何训练产物，用于对照「有搜索 / 无搜索」。
每步做 num_playouts 次模拟。

强度上限来自随机模拟本身：9×9 上随机对局几乎走不出五连，搜索得到的状态价值近似噪声，
因此增加模拟次数收益极小——实测 1000 / 4000 / 8000 次模拟对 AlphaZero 皆为全败。
若需要真正有威胁的搜索基线，应把 _rollout 换成轻量启发式落子，届时模拟次数需相应下调。
"""

from __future__ import annotations

import math
import random

from rl.agents.alphazero import _legal_actions, _move
from rl.agents.base_agent import BaseAgent


class _Node:
    """UCT 树节点；state 相对「该节点当前玩家」。"""

    __slots__ = ("state", "N", "W", "children", "untried", "terminal")

    def __init__(self, state, size, terminal=False):
        self.state = state
        self.N = 0
        self.W = 0.0
        self.children = {}  # action -> _Node
        self.untried = _legal_actions(state, size)
        self.terminal = terminal


class PureMCTSAgent(BaseAgent):
    """纯 MCTS：UCB1 选择 + 随机模拟 + 反向传播。"""

    def __init__(
        self, board_size=9, win_count=5, num_playouts=1000, c_uct=1.4, seed=None
    ):
        self.board_size = board_size
        self.win_count = win_count
        self.num_playouts = num_playouts
        self.c_uct = c_uct
        self.rng = random.Random(seed)

    def select_action(self, state, legal_actions):
        s = tuple(tuple(row) for row in state)
        root = _Node(s, self.board_size)

        for _ in range(self.num_playouts):
            node, path = root, []
            # 1) 选择：沿 UCB1 下探，直到遇到未完全展开的节点或终局
            while not node.terminal and not node.untried and node.children:
                a = self._select(node)
                path.append((node, a))
                node = node.children[a]

            # 2) 扩展 + 3) 模拟
            if node.terminal:
                value = -1.0  # 该节点玩家已输
            elif node.untried:
                a = node.untried.pop(self.rng.randrange(len(node.untried)))
                child_state, won = _move(node.state, a, self.board_size, self.win_count)
                child = _Node(child_state, self.board_size, terminal=won)
                node.children[a] = child
                path.append((node, a))
                node = child
                value = -1.0 if won else self._rollout(child.state)
            else:
                value = 0.0  # 无合法动作：和棋

            # 4) 反向传播：逐层取反（对手视角）
            node.N += 1
            node.W += value
            for parent, _a in reversed(path):
                value = -value
                parent.N += 1
                parent.W += value

        if not root.children:
            return self.rng.choice(legal_actions)
        return max(root.children, key=lambda a: root.children[a].N)

    def _select(self, node):
        """UCB1：平均价值 + 探索项。"""
        total = sum(c.N for c in node.children.values())
        log_total = math.log(total) if total > 0 else 0.0
        best_a, best_v = None, float("-inf")
        for a, c in node.children.items():
            exploit = -c.W / c.N if c.N else 0.0
            explore = self.c_uct * math.sqrt(log_total / c.N) if c.N else float("inf")
            v = exploit + explore
            if v > best_v:
                best_a, best_v = a, v
        return best_a

    def _rollout(self, state):
        """随机走到底，返回「该局面当前玩家」视角的结果：胜 1 / 负 -1 / 和 0。"""
        size = self.board_size
        empties = [
            r * size + c for r in range(size) for c in range(size) if state[r][c] == 0
        ]
        sign = 1
        while empties:
            empties = _legal_actions(state, size)
            if not empties:
                return float(-sign)
            i = self.rng.randrange(len(empties))
            a = empties[i]
            empties[i] = empties[-1]
            empties.pop()
            state, won = _move(state, a, size, self.win_count)
            if won:
                return float(sign)
            sign = -sign
        return 0.0
