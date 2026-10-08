"""基于线性函数近似的 Q-Learning Agent。

用少量手工特征（连线长度 + 开放端）描述 (state, action)，
Q(s, a) = w · φ(s, a)，按标准 Q-Learning 更新学习权重 w：

    Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]

训练时 Agent 固定执一方、与一个「固定对手」对弈（本项目先用 RandomAgent），
对手策略固定后，整个环境退化为稳定 MDP，适合作为第一个可学习 Agent 起手，
也是后续 DQN、策略梯度等方法的对比基准。

约定（state/action/reward 与 rl/environment/gomoku.py 一致）：
    state      相对 Agent 自己的棋盘，1 自己 / -1 对手 / 0 空
    action     row * board_size + col
    reward     对 Agent 而言：胜 +1，负 -1，和 / 未结束 0
    next_state 落子后的棋盘（仍相对 Agent 自己）
    done       是否终局
"""

from __future__ import annotations

import json
import random

from rl.agents.base_agent import BaseAgent
from rl.environment.gomoku import decode_action, encode_action

# 特征顺序固定，与 self.w 一一对应
FEATURE_NAMES = (
    "bias",         # 常数项 1
    "win",          # 己方成五
    "open4",        # 己方活四（两端开放）
    "half4",        # 己方冲四（一端被堵）
    "open3",        # 己方活三
    "half3",        # 己方眠三
    "open2",        # 己方活二
    "opp_win",      # 阻挡对方成五
    "opp_open4",    # 阻挡对方活四
    "opp_half4",    # 阻挡对方冲四
    "opp_open3",    # 阻挡对方活三
    "opp_half3",    # 对方眠三
    "opp_open2",    # 对方活二
)

# 四个扫描方向：横、竖、两条对角线
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _line_run(state, r, c, dr, dc, size, color):
    """把 (r,c) 视作 color 落子后，该方向上的连线长度与开放端数。

    color 取 1 表示自己，-1 表示对手；开放端指连线两端相邻格子为空。
    """
    run = 1
    open_ends = 0
    for step in (1, -1):
        rr, cc = r + dr * step, c + dc * step
        while 0 <= rr < size and 0 <= cc < size and state[rr][cc] == color:
            run += 1
            rr += dr * step
            cc += dc * step
        # 连线尽头：越界或撞到对方棋子 => 该端被堵；为空 => 该端开放
        if 0 <= rr < size and 0 <= cc < size and state[rr][cc] == 0:
            open_ends += 1
    return run, open_ends


def _accumulate(feat, run, open_ends, need, prefix):
    """把一条连线按长度 / 开放度计入 feat 字典的对应特征。"""
    if run >= need:
        feat[prefix + "win"] += 1
    elif run == need - 1:
        if open_ends == 2:
            feat[prefix + "open4"] += 1
        elif open_ends == 1:
            feat[prefix + "half4"] += 1
    elif run == need - 2:
        if open_ends == 2:
            feat[prefix + "open3"] += 1
        elif open_ends == 1:
            feat[prefix + "half3"] += 1
    elif run == need - 3 and open_ends == 2:
        feat[prefix + "open2"] += 1


def _features(state, action, size, need):
    """返回 (state, action) 的特征向量，长度与 FEATURE_NAMES 一致。"""
    r, c = decode_action(action, size)
    feat = {name: 0 for name in FEATURE_NAMES}
    feat["bias"] = 1.0
    for dr, dc in DIRECTIONS:
        run_mine, open_mine = _line_run(state, r, c, dr, dc, size, color=1)
        _accumulate(feat, run_mine, open_mine, need, "")
        run_opp, open_opp = _line_run(state, r, c, dr, dc, size, color=-1)
        _accumulate(feat, run_opp, open_opp, need, "opp_")
    return feat


class QLearningAgent(BaseAgent):
    """线性函数近似的 Q-Learning，用于与固定对手对弈。"""

    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.99, seed=None,
                 board_size=9, win_count=5):
        self.epsilon = epsilon  # ε-greedy 探索概率
        self.alpha = alpha      # 学习率
        self.gamma = gamma      # 折扣因子
        self.board_size = board_size
        self.win_count = win_count
        self.rng = random.Random(seed)
        self.w = {name: 0.0 for name in FEATURE_NAMES}

    def _q(self, state, action):
        """Q(s, a) = w · φ(s, a)。"""
        phi = _features(state, action, self.board_size, self.win_count)
        return sum(self.w[name] * value for name, value in phi.items())

    def _legal_from_state(self, state):
        """由棋盘直接推出合法动作（空位），避免依赖 Environment。"""
        size = len(state)
        return [
            encode_action(r, c, size)
            for r in range(size)
            for c in range(size)
            if state[r][c] == 0
        ]

    def _max_q(self, state, legal_actions):
        if not legal_actions:
            return 0.0
        return max(self._q(state, a) for a in legal_actions)

    def _best_action(self, state, legal_actions):
        """取 Q 最大的动作，多个并列时随机打破平局。"""
        best_q = float("-inf")
        best = []
        for a in legal_actions:
            q = self._q(state, a)
            if q > best_q:
                best_q = q
                best = [a]
            elif q == best_q:
                best.append(a)
        return self.rng.choice(best)

    def select_action(self, state, legal_actions):
        if not legal_actions:
            raise ValueError("没有合法动作可选择")
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(legal_actions))
        return self._best_action(state, legal_actions)

    def update(self, state, action, reward, next_state, done):
        """标准 Q-Learning 半梯度更新（单智能体视角）。"""
        if done:
            target = reward
        else:
            next_legal = self._legal_from_state(next_state)
            target = reward + self.gamma * self._max_q(next_state, next_legal)

        td = target - self._q(state, action)
        phi = _features(state, action, self.board_size, self.win_count)
        for name, value in phi.items():
            self.w[name] += self.alpha * td * value

    def save(self, path):
        """把权重与棋盘参数存为 JSON，供后端加载。"""
        data = {
            "w": self.w,
            "board_size": self.board_size,
            "win_count": self.win_count,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path):
        """从 JSON 读取权重；缺失的特征权重按 0 处理。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.w = {name: float(data["w"].get(name, 0.0)) for name in FEATURE_NAMES}
        self.board_size = data.get("board_size", self.board_size)
        self.win_count = data.get("win_count", self.win_count)
