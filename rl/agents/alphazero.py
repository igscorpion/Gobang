"""AlphaZero-lite：MCTS + 双头（策略/价值）网络。

网络输入相对当前玩家的 3 通道棋盘，输出：
    policy logits（board_size^2 维）+ value（[-1,1] 标量，当前玩家胜率）
MCTS 用 PUCT 选择，靠前瞻搜索发现「双杀」等组合威胁，从而超过反应式的 greedy。

训练见 rl/training/train_alphazero.py（自对弈 + 策略/价值联合损失）。
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from rl.agents.base_agent import BaseAgent
from rl.agents.dqn import state_to_tensor


class AlphaZeroNetwork(nn.Module):
    """小型 CNN，双头输出：策略 logits 与价值。"""

    def __init__(self, board_size: int = 9):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(2 * self.action_size, self.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(self.action_size, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x):
        h = self.conv(x)
        policy = self.policy_head(h)   # (B, action_size)
        value = self.value_head(h)     # (B, 1)
        return policy, value


def _legal_actions(state, size=9):
    return [r * size + c for r in range(size) for c in range(size)
            if state[r][c] == 0]


def _check_win(state, r, c, size=9, win=5):
    player = 1
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        cnt = 1
        for s in (1, -1):
            rr, cc = r + dr * s, c + dc * s
            while 0 <= rr < size and 0 <= cc < size and state[rr][cc] == player:
                cnt += 1
                rr += dr * s
                cc += dc * s
        if cnt >= win:
            return True
    return False


def _move(state, action, size=9, win=5):
    """在相对棋盘落子并翻转视角；返回 (新棋盘, 是否成五)。"""
    r, c = divmod(action, size)
    grid = [list(row) for row in state]
    grid[r][c] = 1
    won = _check_win(grid, r, c, size, win)
    flipped = tuple(tuple(-cell for cell in row) for row in grid)
    return flipped, won


class _Node:
    """MCTS 树节点。state 为相对「该节点当前玩家」的棋盘。"""

    __slots__ = ("state", "P", "N", "W", "children", "terminal", "value",
                 "add_noise")

    def __init__(self, state, P=1.0):
        self.state = state
        self.P = P          # 先验概率
        self.N = 0          # 访问次数
        self.W = 0.0        # 累计价值（本节点玩家视角）
        self.children = {}  # action -> _Node
        self.terminal = False
        self.value = 0.0    # 终局节点的价值
        self.add_noise = False


class MCTS:
    """纯 PUCT 蒙特卡洛树搜索（每步重建树，简单起见不做跨步复用）。"""

    def __init__(self, net, c_puct=1.4, num_sims=50, device="cpu",
                 board_size=9, win_count=5, add_noise=False,
                 noise_eps=0.25, noise_alpha=0.3):
        self.net = net
        self.c_puct = c_puct
        self.num_sims = num_sims
        self.device = device
        self.board_size = board_size
        self.win_count = win_count
        self.add_noise = add_noise          # 自对弈时在根节点加噪声
        self.noise_eps = noise_eps          # 噪声混合比例
        self.noise_alpha = noise_alpha      # Dirichlet 浓度参数

    def search(self, state):
        """返回 {action: 访问次数}；无合法动作返回 None。"""
        root = _Node(state)
        root.add_noise = self.add_noise  # 仅根节点加 Dirichlet 噪声
        if not _legal_actions(state, self.board_size):
            return None

        for _ in range(self.num_sims):
            node = root
            path = []
            while node.children and not node.terminal:
                a = self._select(node)
                path.append((node, a))
                node = node.children[a]

            if node.terminal:
                value = node.value
            else:
                value = self._expand(node)

            for parent, a in reversed(path):
                child = parent.children[a]
                child.N += 1
                child.W += value
                value = -value

        return {a: c.N for a, c in root.children.items()}

    def _select(self, node):
        total = sum(c.N for c in node.children.values())
        if total == 0:
            return max(node.children.items(), key=lambda kv: kv[1].P)[0]
        sqrt_total = total ** 0.5
        best_a, best_v = None, float("-inf")
        for a, c in node.children.items():
            # c.W 是「子节点（对手）视角」的价值，我方视角应取负
            q = -(c.W / c.N) if c.N > 0 else 0.0
            u = self.c_puct * c.P * sqrt_total / (1.0 + c.N)
            v = q + u
            if v > best_v:
                best_a, best_v = a, v
        return best_a

    def _expand(self, node):
        x = state_to_tensor(node.state, self.board_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
        logits = logits[0]
        value = value[0].item()

        legal = _legal_actions(node.state, self.board_size)
        if not legal:
            return 0.0  # 棋盘填满且无人获胜 => 和棋

        # 只对合法动作做 softmax，得到先验概率
        probs = torch.softmax(logits[legal], dim=0)
        if node.add_noise:
            # 根节点加 Dirichlet 噪声，鼓励探索不同开局
            concentration = torch.full((len(legal),), self.noise_alpha,
                                       device=logits.device)
            noise = torch.distributions.Dirichlet(concentration).sample()
            probs = (1.0 - self.noise_eps) * probs + self.noise_eps * noise

        for i, a in enumerate(legal):
            new_state, won = _move(node.state, a, self.board_size, self.win_count)
            child = _Node(new_state, P=float(probs[i]))
            if won:
                child.terminal = True
                child.value = -1.0  # 对手视角：已输
            node.children[a] = child
        return value


class AlphaZeroAgent(BaseAgent):
    """用 MCTS + 网络选择动作；作为人机对战的 AI 或评估用 Agent。"""

    def __init__(self, board_size=9, num_sims=50, c_puct=1.4,
                 device=None, seed=None):
        self.board_size = board_size
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AlphaZeroNetwork(board_size).to(self.device)
        self.rng = random.Random(seed)

    def select_action(self, state, legal_actions):
        s = tuple(tuple(row) for row in state)
        mcts = MCTS(self.net, self.c_puct, self.num_sims, self.device,
                    self.board_size)
        counts = mcts.search(s)
        if counts is None:
            raise ValueError("没有合法动作可选择")
        return max(counts, key=counts.get)

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.to(self.device)

    def update(self, state, action, reward, next_state, done):
        # AlphaZero 用自对弈 + MCTS 批量训练，不走逐步 update
        return None
