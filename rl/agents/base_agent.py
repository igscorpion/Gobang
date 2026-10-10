"""所有强化学习 Agent 的统一接口。

后续实现 Q-Learning / SARSA / Monte Carlo / DQN 时，继承 BaseAgent，
并实现 select_action 与 update。不要改 Environment 或 Web。
"""

from __future__ import annotations


class BaseAgent:
    # 为 True 表示该 Agent 除棋盘外还需要「对手上一步」，
    # 此时调用方传入的 state 应为 (棋盘, 上一步动作)。
    needs_last_move = False

    def select_action(self, state, legal_actions):
        """根据当前状态从合法动作中选择一个动作。

        state: 相对当前玩家的棋盘（1 自己，-1 对手，0 空）；
               若 needs_last_move 为 True，则为 (棋盘, 对手上一步动作)
        legal_actions: 可落子位置列表，动作为 row * board_size + col
        """
        raise NotImplementedError

    def update(self, state, action, reward, next_state, done):
        """用一次转移更新参数。当前阶段占位，具体算法在子类中实现。"""
        raise NotImplementedError

    def save(self, path):
        """保存模型。当前阶段可为空实现。"""
        pass

    def load(self, path):
        """加载模型。当前阶段可为空实现。"""
        pass
