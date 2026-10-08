"""DQN Agent：用神经网络直接学习 Q 值。

输入为相对当前玩家的 3 通道棋盘（自己 / 对手 / 空位），
输出 board_size^2 个动作的 Q 值。训练见 rl/training/train_dqn.py，
采用自对弈 + 经验回放 + 目标网络（negamax 目标）。

依赖 torch；本文件只在按名称加载 "dqn" 时才被 import，
因此未安装 torch 时 Web 后端仍可正常启动，切换时报明确错误。
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from rl.agents.base_agent import BaseAgent


class QNetwork(nn.Module):
    """小型 CNN：3 通道棋盘 -> board_size^2 维 Q 值。"""

    def __init__(self, board_size: int = 9):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        flat = 64 * board_size * board_size
        self.head = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size),
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.head(x)


def state_to_tensor(state, board_size: int = 9):
    """相对棋盘 state -> (3, H, W) 张量，三通道为 自己/对手/空位。"""
    t = torch.tensor(state, dtype=torch.long)
    own = (t == 1).float()
    opp = (t == -1).float()
    empty = (t == 0).float()
    return torch.stack([own, opp, empty], dim=0)


class DQNAgent(BaseAgent):
    """DQN Agent：select_action 用 ε-greedy，训练由 train_dqn.py 批量完成。"""

    def __init__(self, board_size: int = 9, epsilon: float = 0.1,
                 device: str | None = None, seed: int | None = None):
        self.board_size = board_size
        self.epsilon = epsilon
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = QNetwork(board_size).to(self.device)
        self.rng = random.Random(seed)

    def _q_values(self, state):
        """返回 (action_size,) 的 CPU 张量，不记录梯度。"""
        x = state_to_tensor(state, self.board_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.net(x).squeeze(0).cpu()

    def select_action(self, state, legal_actions):
        if not legal_actions:
            raise ValueError("没有合法动作可选择")
        legal = list(legal_actions)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal)

        q = self._q_values(state)  # (action_size,)
        q_legal = q[torch.tensor(legal, dtype=torch.long)]
        return legal[int(torch.argmax(q_legal).item())]

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.to(self.device)

    def update(self, state, action, reward, next_state, done):
        # DQN 用经验回放批量训练，不走逐步 update，这里保持空实现
        return None
