"""AlphaZero-lite 自对弈训练。

用 MCTS 自对弈产生 (棋盘, 访问概率 π, 胜负 z) 样本，训练双头网络：
    loss = 交叉熵(policy, π) + MSE(value, z) + L2 正则
每次更新在同一个 mini-batch 上训练若干轮，并按策略 KL 散度早停、自适应调节学习率。
训练后保存到 rl/models/alphazero.pt，并周期性对 greedy / random 评估。

用法（项目根目录，需要装有 torch 的 Python，本机用系统 Python 3.11 + GPU）：
    py -3.11 -m rl.training.train_alphazero [games] [num_sims]
    games=0（默认）表示无限训练（Ctrl+C 停止）；给定 N 则训练 N 局后停止。

每 10 局保存一次检查点（单局耗时较长，缩短间隔以减少中断损失）、每 1000 局评估一次；
再次运行自动从检查点续训。
"""

from __future__ import annotations

import collections
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from rl.agents.alphazero import (AlphaZeroAgent, AlphaZeroNetwork, MCTS,
                                 last_move_action, state_to_tensor4)
from rl.agents.registry import create_agent
from rl.environment.gomoku import BLACK, GomokuEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "rl" / "models" / "alphazero.pt"
META_PATH = PROJECT_ROOT / "rl" / "models" / "alphazero_meta.json"


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


L2_CONST = 1e-4  # L2 正则系数（与参考实现一致）


def _batch_tensor(batch, device, board_size):
    return torch.stack([state_to_tensor4(s, board_size)
                        for s, _, _ in batch]).to(device)


def _policy_probs(net, xs):
    """批量计算策略概率，用于 KL 监控。"""
    with torch.no_grad():
        logits, _ = net(xs)
        return torch.softmax(logits, dim=1)


def train_step(net, optimizer, batch, device, board_size, lr):
    xs = _batch_tensor(batch, device, board_size)
    pi_t = torch.tensor([pi for _, pi, _ in batch],
                        dtype=torch.float32).to(device)
    z_t = torch.tensor([z for _, _, z in batch],
                       dtype=torch.float32).to(device).unsqueeze(1)

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
    opponent = create_agent(opponent_name)
    wins = 0
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
    return wins / games


def main():
    # games=0 表示无限训练（默认，Ctrl+C 停止）；给定 N 则训练 N 局后停止
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    board_size = 9
    c_puct = 1.4
    lr = 2e-3          # 基准学习率
    lr_multiplier = 1.0  # 按 KL 自适应调节的倍率
    batch_size = 512
    train_epochs = 5     # 每次更新在同一个 mini-batch 上训练几轮
    kl_targ = 0.02       # KL 目标：早停与学习率调节的依据
    buffer_capacity = 40000
    eval_every = 1000  # 每 1000 局评估一次（打印胜率）
    save_every = 10    # 每 10 局保存一次检查点（单局耗时较长，缩短间隔以减少中断损失）

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = GomokuEnv()
    net = AlphaZeroNetwork(board_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    buffer = collections.deque(maxlen=buffer_capacity)

    # 断点续训：存在检查点就直接从检查点继续
    if MODEL_PATH.exists():
        try:
            net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"从检查点继续: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
        except Exception as exc:
            print(f"检查点与当前网络结构不兼容，改为从头训练（{exc}）")

    # 历史累计局数（跨重启保持）
    total_done = 0
    if META_PATH.exists():
        try:
            total_done = int(json.loads(META_PATH.read_text(encoding="utf-8")).get("total_games", 0))
        except Exception:
            total_done = 0
    if total_done:
        print(f"历史累计已训练 {total_done} 局")

    mode = "无限训练（Ctrl+C 停止）" if games == 0 else f"{games} 局"
    print(f"AlphaZero-lite 自对弈 {mode}（device={device}, sims={num_sims}）...")
    game = 0
    loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    loss_count = 0
    draw_games = 0
    win_games = 0
    start_time = time.time()
    while games == 0 or game < games:
        game += 1
        data, winner = selfplay_game(net, device, board_size, num_sims, c_puct, 1.0)
        buffer.extend(data)
        if winner == 0:
            draw_games += 1
        else:
            win_games += 1

        kl = 0.0
        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            xs = _batch_tensor(batch, device, board_size)
            old_probs = _policy_probs(net, xs)
            for _ in range(train_epochs):
                p_loss, v_loss = train_step(net, optimizer, batch, device,
                                            board_size, lr * lr_multiplier)
                new_probs = _policy_probs(net, xs)
                kl = torch.mean(torch.sum(
                    old_probs * (torch.log(old_probs + 1e-10)
                                 - torch.log(new_probs + 1e-10)), dim=1)).item()
                if kl > kl_targ * 4:  # 偏离过大，提前停止
                    break

            # 按 KL 自适应调节学习率（与参考实现一致）
            if kl > kl_targ * 2 and lr_multiplier > 0.1:
                lr_multiplier /= 1.5
            elif kl < kl_targ / 2 and lr_multiplier < 10:
                lr_multiplier *= 1.5

            loss_sum += p_loss + v_loss
            policy_loss_sum += p_loss
            value_loss_sum += v_loss
            loss_count += 1

        saved = game % save_every == 0
        if saved:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(MODEL_PATH))
            META_PATH.write_text(
                json.dumps({"total_games": total_done + game}), encoding="utf-8")

        # 每局打印一行进度（均值 loss + 每局耗时），便于判断是否卡住
        n = max(loss_count, 1)
        rate = (time.time() - start_time) / 60.0 / game
        print(f"  game {game:6d} (累计 {total_done + game:6d}) | "
              f"loss {loss_sum / n:.4f} (policy {policy_loss_sum / n:.4f} / "
              f"value {value_loss_sum / n:.4f}) | lr {lr * lr_multiplier:.1e} | "
              f"KL {kl:.3f} | {rate:.1f} 分/局 | "
              f"{'已保存' if saved else '进行中'}", flush=True)
        loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        loss_count = 0

        if game % eval_every == 0:
            agent = AlphaZeroAgent(board_size=board_size, num_sims=num_sims,
                                   device=device)
            agent.net.load_state_dict(net.state_dict())
            wr_greedy = evaluate(agent, env, "greedy", games=5)
            wr_random = evaluate(agent, env, "random", games=5)
            total = total_done + game
            played = draw_games + win_games
            draw_rate = draw_games / played if played else 0.0
            print(f"  game {game:6d} (累计 {total:6d}) | vs greedy {wr_greedy:.2f} | "
                  f"vs random {wr_random:.2f} | 自对弈和棋率 {draw_rate:.2f}", flush=True)
            draw_games = 0
            win_games = 0


if __name__ == "__main__":
    main()
