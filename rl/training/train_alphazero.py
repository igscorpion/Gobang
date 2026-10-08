"""AlphaZero-lite 自对弈训练。

用 MCTS 自对弈产生 (棋盘, 访问概率 π, 胜负 z) 样本，训练双头网络：
    loss = 交叉熵(policy, π) + MSE(value, z)
训练后保存到 rl/models/alphazero.pt，并周期性对 greedy / random 评估。

用法（项目根目录，需要装有 torch 的 Python，本机用系统 Python 3.11 + GPU）：
    py -3.11 -m rl.training.train_alphazero [games] [num_sims]
    games=0（默认）表示无限训练（Ctrl+C 停止）；给定 N 则训练 N 局后停止。

每 100 局保存一次检查点、每 1000 局评估一次；再次运行自动从检查点续训。
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from rl.agents.alphazero import (AlphaZeroAgent, AlphaZeroNetwork, MCTS)
from rl.agents.dqn import state_to_tensor
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
    """把一个 (state, pi, z) 扩成 8 个旋转/镜像等价样本。"""
    out = []
    for f in _board_transforms(size):
        new_board = [[0] * size for _ in range(size)]
        new_pi = [0.0] * (size * size)
        for r in range(size):
            for c in range(size):
                nr, nc = f(r, c)
                new_board[nr][nc] = state[r][c]
                new_pi[nr * size + nc] = pi[r * size + c]
        out.append((tuple(tuple(row) for row in new_board), new_pi, z))
    return out


def selfplay_game(net, device, board_size, num_sims, c_puct, temp):
    """自对弈一局，返回 8 倍增强后的 [(state, pi, z), ...]。"""
    env = GomokuEnv()
    mcts = MCTS(net, c_puct, num_sims, device, board_size, add_noise=True)
    records = []
    state = env.reset()
    while True:
        s = tuple(tuple(row) for row in state)
        counts = mcts.search(s)
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
        records.append((s, pi, env.current_player))

        state, _, done = env.step(action)
        if done:
            break

    winner = env.winner
    samples = [(s, pi, float(winner * player)) for (s, pi, player) in records]
    # 8 倍数据增强（旋转 + 镜像），显著提升有限算力下的样本量
    augmented = []
    for s, pi, z in samples:
        augmented.extend(_augment_record(s, pi, z, board_size))
    return augmented


def train_step(net, optimizer, batch, device, board_size):
    states, pis, zs = zip(*batch)
    xs = torch.stack([state_to_tensor(s, board_size) for s in states]).to(device)
    pi_t = torch.tensor(pis, dtype=torch.float32).to(device)
    z_t = torch.tensor(zs, dtype=torch.float32).to(device).unsqueeze(1)

    logits, v = net(xs)
    log_p = torch.log_softmax(logits, dim=1)
    policy_loss = -(pi_t * log_p).sum(dim=1).mean()
    value_loss = F.mse_loss(v, z_t)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(agent, env, opponent_name, games, agent_color=BLACK):
    opponent = create_agent(opponent_name)
    wins = 0
    for _ in range(games):
        state = env.reset()
        done = False
        while not done:
            legal = env.get_legal_actions()
            if env.current_player == agent_color:
                action = agent.select_action(state, legal)
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
    lr = 1e-3
    batch_size = 128
    train_steps_per_game = 6
    buffer_capacity = 40000
    eval_every = 1000  # 每 1000 局评估一次（打印胜率）
    save_every = 100   # 每 100 局保存一次检查点（较频繁，防止中断丢太多进度）

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = GomokuEnv()
    net = AlphaZeroNetwork(board_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    buffer = collections.deque(maxlen=buffer_capacity)

    # 断点续训：存在检查点就直接从检查点继续
    if MODEL_PATH.exists():
        net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"从检查点继续: {MODEL_PATH.relative_to(PROJECT_ROOT)}")

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
    loss_count = 0
    while games == 0 or game < games:
        game += 1
        data = selfplay_game(net, device, board_size, num_sims, c_puct, 1.0)
        buffer.extend(data)

        for _ in range(train_steps_per_game):
            if len(buffer) < batch_size:
                break
            batch = random.sample(buffer, batch_size)
            loss_sum += train_step(net, optimizer, batch, device, board_size)
            loss_count += 1

        if game % save_every == 0:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(MODEL_PATH))
            total = total_done + game
            META_PATH.write_text(json.dumps({"total_games": total}), encoding="utf-8")
            avg_loss = loss_sum / max(loss_count, 1)
            print(f"  game {game:6d} (累计 {total:6d}) | avg_loss {avg_loss:.4f} | 已保存",
                  flush=True)
            loss_sum = 0.0
            loss_count = 0

        if game % eval_every == 0:
            agent = AlphaZeroAgent(board_size=board_size, num_sims=num_sims,
                                   device=device)
            agent.net.load_state_dict(net.state_dict())
            wr_greedy = evaluate(agent, env, "greedy", games=5)
            wr_random = evaluate(agent, env, "random", games=5)
            total = total_done + game
            print(f"  game {game:6d} (累计 {total:6d}) | vs greedy {wr_greedy:.2f} | "
                  f"vs random {wr_random:.2f}", flush=True)


if __name__ == "__main__":
    main()
