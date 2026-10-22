"""AlphaZero 对局、D4 数据增强与策略/价值更新；调度见 local_alphazero。"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from rl.agents.alphazero import (
    MCTS,
    last_move_action,
    sample_action,
    state_to_tensor4,
)
from rl.agents.registry import create_agent
from rl.environment.gomoku import BLACK, GomokuEnv


def _board_transforms(size):
    """正方形的 8 个对称变换（D4 群）：每个是 (r, c) -> (r', c')。"""
    n = size - 1
    return [
        lambda r, c: (r, c),
        lambda r, c: (c, n - r),
        lambda r, c: (n - r, n - c),
        lambda r, c: (n - c, r),
        lambda r, c: (r, n - c),
        lambda r, c: (n - r, c),
        lambda r, c: (c, r),
        lambda r, c: (n - c, n - r),
    ]


def _augment_record(state, pi, z, size):
    """把一个 (state, pi, z) 扩成 8 个旋转/镜像等价样本。

    state = (棋盘, 对手上一步)，棋盘与「上一步」都要做同一变换。
    """
    board, last_move = state
    out = []
    for f in _board_transforms(size):
        new_board = [[0] * size for _ in range(size)]
        new_pi = [0.0] * (size * size)
        for r in range(size):
            for c in range(size):
                nr, nc = f(r, c)
                new_board[nr][nc] = board[r][c]
                new_pi[nr * size + nc] = pi[r * size + c]
        new_last = None
        if last_move is not None:
            lr_, lc_ = divmod(last_move, size)
            nlr, nlc = f(lr_, lc_)
            new_last = nlr * size + nlc
        out.append(((tuple(tuple(row) for row in new_board), new_last), new_pi, z))
    return out


def selfplay_game(net, device, board_size, num_sims, c_puct, temp):
    """自对弈一局，返回 8 倍增强后的 [(state, pi, z), ...]。"""
    env = GomokuEnv()
    mcts = MCTS(net, c_puct, num_sims, device, board_size, add_noise=True)
    records = []
    state = env.reset()
    last_move = None
    while True:
        board = tuple(tuple(row) for row in state)
        counts = mcts.search(board, last_move)
        if counts is None:
            break

        # π ∝ N^(1/temp)
        action_size = board_size * board_size
        weights = [0.0] * action_size
        for a, n in counts.items():
            weights[a] = n ** (1.0 / temp)
        action = random.choices(range(action_size), weights=weights)[0]

        total = sum(weights)
        pi = [w / total for w in weights]
        records.append(((board, last_move), pi, env.current_player))

        state, _, done = env.step(action)
        last_move = action
        if done:
            break

    winner = env.winner
    samples = [(s, pi, float(winner * player)) for (s, pi, player) in records]
    # 8 倍数据增强（旋转 + 镜像），显著提升有限算力下的样本量
    augmented = []
    for s, pi, z in samples:
        augmented.extend(_augment_record(s, pi, z, board_size))
    return augmented, winner


def vs_opponent_game(
    net,
    device,
    board_size,
    num_sims,
    c_puct,
    temp,
    opponent,
    opening_moves=6,
    candidate_ratio=0.1,
):
    """让 net（+MCTS）与固定对手对弈一局，只记录 net 自己的落子。

    随机决定 net 执黑还是执白，使网络同时获得「进攻」与「防守」两类局面，
    这正是纯自对弈所缺失的（同一个网络执双方时会一路对攻、不设防）。
    net 的落子只在开局前 opening_moves 手内按温度抽样，之后一律走最优手。

    返回 (8 倍增强样本, 胜者)。
    """
    env = GomokuEnv()
    mcts = MCTS(net, c_puct, num_sims, device, board_size, add_noise=True)
    agent_color = random.choice((BLACK, -BLACK))
    records = []
    state = env.reset()
    last_move = None
    while True:
        if env.current_player == agent_color:
            board = tuple(tuple(row) for row in state)
            counts = mcts.search(board, last_move)
            if counts is None:
                break

            # π ∝ N^(1/temp)：作为策略目标，保留完整的访问分布
            action_size = board_size * board_size
            weights = [0.0] * action_size
            for a, n in counts.items():
                weights[a] = n ** (1.0 / temp)
            total = sum(weights)
            pi = [w / total for w in weights]

            # 落子只在开局前 opening_moves 手内抽样：否则每一步都抽样，
            # 那些「只被访问过一两次、等于没搜过」的走法也会被选中，拉低对局与样本质量
            stones = sum(1 for row in board for v in row if v != 0)
            play_temp = temp if stones < opening_moves else 0.0
            action = sample_action(counts, play_temp, candidate_ratio)
            records.append(((board, last_move), pi, agent_color))
        else:
            # 对手回合不记录样本：网络要学的是自己的策略，不是模仿对手
            action = opponent.select_action(env.get_state(), env.get_legal_actions())

        state, _, done = env.step(action)
        last_move = action
        if done:
            break

    winner = env.winner
    samples = [(s, pi, float(winner * player)) for (s, pi, player) in records]
    augmented = []
    for s, pi, z in samples:
        augmented.extend(_augment_record(s, pi, z, board_size))
    return augmented, winner, agent_color


L2_CONST = 1e-4  # L2 正则系数（与参考实现一致）


def _batch_tensor(batch, device, board_size):
    return torch.stack([state_to_tensor4(s, board_size) for s, _, _ in batch]).to(
        device
    )


def _policy_probs(net, xs):
    """批量计算策略概率，用于 KL 监控。"""
    with torch.no_grad():
        logits, _ = net(xs)
        return torch.softmax(logits, dim=1)


def train_step(net, optimizer, batch, device, board_size, lr):
    xs = _batch_tensor(batch, device, board_size)
    pi_t = torch.tensor([pi for _, pi, _ in batch], dtype=torch.float32).to(device)
    z_t = (
        torch.tensor([z for _, _, z in batch], dtype=torch.float32)
        .to(device)
        .unsqueeze(1)
    )

    logits, v = net(xs)
    log_p = torch.log_softmax(logits, dim=1)
    policy_loss = -(pi_t * log_p).sum(dim=1).mean()
    value_loss = F.mse_loss(v, z_t)
    l2 = sum(p.pow(2).sum() for p in net.parameters())
    loss = policy_loss + value_loss + L2_CONST * l2

    for group in optimizer.param_groups:  # 学习率由外部按 KL 自适应调节
        group["lr"] = lr
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return policy_loss.item(), value_loss.item()


def evaluate(agent, env, opponent_name, games, agent_color=BLACK):
    """与对手对弈 games 局，返回 (胜率, 和率, 负率)。

    和棋指棋盘填满且无人成五。五子棋先手优势明显，和棋对执黑 / 执白含义不同，
    因此与败局分开统计，避免混为一个「非胜」。
    """
    opponent = create_agent(opponent_name)
    wins = 0
    draws = 0
    for _ in range(games):
        state = env.reset()
        done = False
        while not done:
            legal = env.get_legal_actions()
            if env.current_player == agent_color:
                payload = state
                if agent.needs_last_move:
                    # AlphaZero 需要「对手上一步」，一并传入
                    payload = (state, last_move_action(env.last_move, env.board_size))
                action = agent.select_action(payload, legal)
            else:
                action = opponent.select_action(state, legal)
            state, _reward, done = env.step(action)
        if env.winner == agent_color:
            wins += 1
        elif env.winner == env.EMPTY:
            draws += 1
    return wins / games, draws / games, (games - wins - draws) / games
