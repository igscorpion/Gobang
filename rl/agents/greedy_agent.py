"""贪心启发式 Agent：按手工权重给候选落子打分，选最高分。

会进攻（优先成五/活四/活三）、会防守（优先堵对方成五/活四），
用作训练对手时能逼出学习者的防守行为，也作为比 RandomAgent 更强的评估基准。
"""

from __future__ import annotations

import random

from rl.agents.base_agent import BaseAgent
from rl.agents.q_learning import FEATURE_NAMES, _features

# 手工评分权重：只覆盖进攻与关键防守，其余特征权重为 0
GREEDY_WEIGHTS = {
    "win": 10000,          # 自己成五，必走
    "opp_win": 9000,       # 堵对方成五，必走
    "open4": 1000,         # 自己活四
    "opp_open4": 500,      # 堵对方活四
    "opp_half4": 450,      # 堵对方冲四
    "half4": 400,          # 自己冲四
    "open3": 100,          # 自己活三
    "opp_open3": 80,       # 堵对方活三
    "half3": 20,           # 自己眠三
    "open2": 10,           # 自己活二
}


class GreedyAgent(BaseAgent):
    def __init__(self, board_size=9, win_count=5, seed=None):
        self.board_size = board_size
        self.win_count = win_count
        self.rng = random.Random(seed)

    def select_action(self, state, legal_actions):
        if not legal_actions:
            raise ValueError("没有合法动作可选择")
        best_score = float("-inf")
        best = []
        for a in legal_actions:
            phi = _features(state, a, self.board_size, self.win_count)
            score = sum(GREEDY_WEIGHTS.get(name, 0.0) * value
                        for name, value in phi.items())
            if score > best_score:
                best_score = score
                best = [a]
            elif score == best_score:
                best.append(a)
        return self.rng.choice(best)
