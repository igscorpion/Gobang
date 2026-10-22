"""AlphaZero-lite：MCTS + 双头（策略/价值）网络。

网络输入为相对当前玩家的 4 通道棋盘（己方、对手、对手上一步、当前执子方是否先手），
输出 policy logits（board_size^2 维）与 value（[-1,1] 标量，当前玩家胜率）。
MCTS 用 PUCT 选择，靠前瞻搜索发现「双杀」等组合威胁。

「对手上一步」由 MCTS 节点自身记录（node.action = 走到该节点的那一手），
无需把历史塞进状态里；「是否先手」由盘面棋子数的奇偶推出。

训练见 rl/training/train_alphazero.py（自对弈 + 策略/价值联合损失）。
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
from rl.environment.state import state_to_tensor4, last_move_action
from rl.environment.rules import relative_legal_actions, rule_mode, winning_move

from rl.agents.base_agent import BaseAgent


class AlphaZeroNetwork(nn.Module):
    """小型 CNN，4 通道输入，双头输出：策略 logits 与价值。"""

    def __init__(self, board_size: int = 9, channels=(64, 64, 128)):
        super().__init__()
        self.channels = tuple(channels)
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.conv = nn.Sequential(
            nn.Conv2d(4, channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels[2], 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.action_size, self.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels[2], 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.action_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.conv(x)
        policy = self.policy_head(h)  # (B, action_size)
        value = self.value_head(h)  # (B, 1)
        return policy, value


def sample_action(counts, temperature, candidate_ratio=0.0, rng=None):
    """按搜索访问次数选一手。

    temperature <= 0：取访问次数最多的一手（评估与实战的确定性口径）。
    temperature > 0 ：在「访问次数 ≥ 最高次数 × candidate_ratio」的候选中，
                      按 N^(1/temperature) 加权随机抽样。
    """
    rng = rng or random
    if temperature <= 0:
        return max(counts, key=counts.get)
    # 候选筛选：搜索已明确判定优劣时（如必须堵活三）只剩正确手，不会被抽样换掉；
    # 局面不明、多个走法势均力敌时才体现多样性
    best = max(counts.values())
    candidates = [a for a, n in counts.items() if n > 0 and n >= candidate_ratio * best]
    if len(candidates) <= 1:
        return max(counts, key=counts.get)
    weights = [counts[a] ** (1.0 / temperature) for a in candidates]
    return rng.choices(candidates, weights=weights)[0]


def _legal_actions(state, size=9, rules=None):
    return relative_legal_actions(state, rules)


def _check_win(state, r, c, size=9, win=5, rules=None):
    color = 1 if sum(v != 0 for row in state for v in row) % 2 else -1
    absolute = [[v * color for v in row] for row in state]
    return winning_move(absolute, r, c, color, rule_mode(rules), win)


def _move(state, action, size=9, win=5, rules=None):
    """在相对棋盘落子并翻转视角；返回 (新棋盘, 是否成五)。"""
    r, c = divmod(action, size)
    grid = [list(row) for row in state]
    grid[r][c] = 1
    won = _check_win(grid, r, c, size, win, rules)
    flipped = tuple(tuple(-cell for cell in row) for row in grid)
    return flipped, won


class _Node:
    """MCTS 树节点。state 为相对「该节点当前玩家」的棋盘。"""

    __slots__ = (
        "state",
        "action",
        "P",
        "P_raw",
        "N",
        "W",
        "children",
        "terminal",
        "value",
        "add_noise",
    )

    def __init__(self, state, action=None, P=1.0):
        self.state = state
        self.action = action  # 走到本节点的那一手（即对手上一步）
        self.P = P  # 先验概率（根节点的子节点会混入 Dirichlet 噪声）
        self.P_raw = P  # 未加噪的先验，供复用根时重新加噪
        self.N = 0  # 访问次数
        self.W = 0.0  # 累计价值（本节点玩家视角）
        self.children = {}  # action -> _Node
        self.terminal = False
        self.value = 0.0  # 终局节点的价值
        self.add_noise = False


class MCTS:
    """PUCT 蒙特卡洛树搜索，支持跨步复用搜索树。

    同一个实例连续调用 search 时，若上一棵树的根里已有「对手刚走那一手」对应的子节点、
    且局面一致，就直接以它为新的根（保留访问次数与价值），省下对已搜索部分的重复计算。
    """

    def __init__(
        self,
        net,
        c_puct=1.4,
        num_sims=50,
        device="cpu",
        board_size=9,
        win_count=5,
        add_noise=False,
        noise_eps=0.25,
        noise_alpha=0.3,
        rules=None,
    ):
        self.rules = rule_mode(rules)
        self.net = net
        self.c_puct = c_puct
        self.num_sims = num_sims
        self.device = device
        self.board_size = board_size
        self.win_count = win_count
        self.add_noise = add_noise  # 自对弈时在根节点加噪声
        self.noise_eps = noise_eps  # 噪声混合比例
        self.noise_alpha = noise_alpha  # Dirichlet 浓度参数
        self.root = None  # 上一次搜索的根，供跨步复用

    def search(self, state, last_move=None):
        """返回 {action: 访问次数}；无合法动作返回 None。

        state: 相对当前玩家的棋盘；last_move: 对手上一步（根节点用）。
        """
        # 跨步复用：上一棵树的根里若已有「对手刚走的那一手」对应的子节点，且局面一致，
        # 就直接拿它当新根，省下对已搜索部分的重复计算。
        root = None
        if last_move is not None and self.root is not None:
            candidate = self.root.children.get(last_move)
            if candidate is not None and candidate.state == state:
                root = candidate
        reused = root is not None
        if root is None:
            root = _Node(state, action=last_move)
        root.add_noise = self.add_noise  # 仅根节点加 Dirichlet 噪声
        if not _legal_actions(state, self.board_size, self.rules):
            return None
        if reused and root.add_noise and root.children:
            # 复用根时重新加噪：子节点先验是上一轮混过噪声的，这里从未加噪版本重新混合
            noise = torch.distributions.Dirichlet(
                torch.full((len(root.children),), self.noise_alpha)
            ).sample()
            for i, child in enumerate(root.children.values()):
                child.P = (1.0 - self.noise_eps) * child.P_raw + self.noise_eps * float(
                    noise[i]
                )
        self.root = root

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
        sqrt_total = total**0.5
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
        x = (
            state_to_tensor4((node.state, node.action), self.board_size)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            logits, value = self.net(x)
        logits = logits[0]
        value = value[0].item()

        legal = _legal_actions(node.state, self.board_size, self.rules)
        if not legal:
            node.terminal = True
            node.value = -1.0 if any(v == 0 for row in node.state for v in row) else 0.0
            return node.value

        # 只对合法动作做 softmax，得到先验概率
        raw = torch.softmax(logits[legal], dim=0)
        probs = raw
        if node.add_noise:
            # 根节点加 Dirichlet 噪声，鼓励探索
            concentration = torch.full(
                (len(legal),), self.noise_alpha, device=logits.device
            )
            noise = torch.distributions.Dirichlet(concentration).sample()
            probs = (1.0 - self.noise_eps) * raw + self.noise_eps * noise

        for i, a in enumerate(legal):
            new_state, won = _move(
                node.state, a, self.board_size, self.win_count, self.rules
            )
            child = _Node(new_state, action=a, P=float(probs[i]))
            child.P_raw = float(raw[i])  # 未加噪先验，供复用根时重新加噪
            if won:
                child.terminal = True
                child.value = -1.0  # 对手视角：已输
            node.children[a] = child
        return value


class AlphaZeroAgent(BaseAgent):
    """用 MCTS + 网络选择动作；作为人机对战的 AI 或评估用 Agent。

    temperature = 0：取访问次数最大的点，行为确定（评测用；注意 GPU 浮点非确定性仍会让
    接近的走法选择发生变化，同一检查点的重复测量并不逐局一致）；
    temperature > 0：仅在前 opening_moves 手抽样，之后回到确定性最优手。
    抽样只在「访问次数达到最高次数 candidate_ratio 倍」的走法中进行——搜索已明确
    判定的强制手（如必须堵活三）不会被换掉，局面不明时仍保留多样性。
    """

    needs_last_move = True  # 输入含「对手上一步」，调用方需传 (棋盘, 上一步动作)

    def __init__(
        self,
        board_size=9,
        num_sims=50,
        c_puct=1.4,
        device=None,
        seed=None,
        temperature=0.0,
        opening_moves=6,
        candidate_ratio=0.1,
        channels=(64, 64, 128),
        rules=None,
    ):
        self.board_size = board_size
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AlphaZeroNetwork(board_size, channels).to(self.device)
        self.rng = random.Random(seed)
        self.temperature = temperature
        self.opening_moves = opening_moves
        self.candidate_ratio = candidate_ratio
        # 常驻一个 MCTS 实例以复用搜索树；换局（局面不符）时会自动重建
        self.mcts = MCTS(
            self.net, c_puct, num_sims, self.device, board_size, rules=rules
        )

    def select_action(self, state, legal_actions):
        # 兼容两种输入：纯棋盘，或 (棋盘, 对手上一步)
        if len(state) == 2 and (state[1] is None or isinstance(state[1], int)):
            board, last_move = state
        else:
            board, last_move = state, None
        s = tuple(tuple(row) for row in board)

        # 温度调度：只在前 opening_moves 手抽样，之后取最优手
        stones = sum(1 for row in s for v in row if v != 0)
        temp = self.temperature if stones < self.opening_moves else 0.0

        counts = self.mcts.search(s, last_move)
        if counts is None:
            raise ValueError("没有合法动作可选择")
        counts = {a: n for a, n in counts.items() if a in legal_actions}
        if not counts:
            raise ValueError("search and environment legality disagree")
        return sample_action(counts, temp, self.candidate_ratio, self.rng)

    def save(self, path):
        torch.save(
            {"state_dict": self.net.state_dict(), "channels": self.net.channels}, path
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        weights = checkpoint.get("state_dict", checkpoint)
        channels = checkpoint.get(
            "channels",
            (
                weights["conv.0.weight"].shape[0],
                weights["conv.2.weight"].shape[0],
                weights["conv.4.weight"].shape[0],
            ),
        )
        self.net = AlphaZeroNetwork(self.board_size, channels).to(self.device)
        self.net.load_state_dict(weights)
        self.mcts.net = self.net
        self.mcts.root = None
        self.net.to(self.device)

    def update(self, state, action, reward, next_state, done):
        # AlphaZero 用自对弈 + MCTS 批量训练，不走逐步 update
        return None
