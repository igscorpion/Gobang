"""DQN 训练，支持两种模式。

模式:
    selfplay  单网络自对弈，negamax 目标（默认）
    greedy    执黑对 greedy 训练，标准 Q-learning
    random    执黑对 random 训练
    mix       课程混合：对手从 random 逐步过渡到 greedy（推荐用于打 greedy）

注意：greedy 明显强于初始 DQN 时，直接对 greedy 训练会因「拿不到胜局、
只有 -1 奖励」而塌缩（实测 vs greedy 全程 0.00、vs random 退化到随机）。
故默认用 selfplay；对 greedy 训练应在 Agent 已有一定水平后再用。

用法（在项目根目录，需要装有 torch 的 Python 环境）：
本机 GPU torch 在系统 Python 3.11 中，直接：
    py -3.11 -m rl.training.train_dqn [episodes] [mode]
（未装 torch 的环境会报错；本机 .venv 无 torch，请勿用 uv run 跑本脚本。）
"""

from __future__ import annotations

import collections
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from rl.agents.dqn import DQNAgent, QNetwork, state_to_tensor
from rl.agents.q_learning import _features
from rl.agents.registry import create_agent
from rl.environment.gomoku import BLACK, GomokuEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "rl" / "models" / "dqn.pt"


class ReplayBuffer:
    """定长经验回放缓冲，存 (s, a, r, s', done)。"""

    def __init__(self, capacity: int = 20000):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buf)


def _perspective(env, agent_color):
    """返回相对 agent_color 的棋盘（1=己方，-1=对手）。"""
    board = env.get_board()
    if agent_color == BLACK:
        return board
    return [[-cell for cell in row] for row in board]


# 奖励塑形权重：给非终局落子一个量级较小的即时信号（进攻 + 防守）
SHAPE_WEIGHTS = {
    "open4": 0.30, "half4": 0.15, "open3": 0.10, "half3": 0.04, "open2": 0.02,
    "opp_win": 0.30, "opp_open4": 0.20, "opp_half4": 0.10, "opp_open3": 0.08,
}


def shape_reward(state, action, board_size=9, win_count=5):
    """落子 (state, action) 的塑形奖励：奖励进攻（活四/活三）与防守（堵对方）。"""
    phi = _features(state, action, board_size, win_count)
    return sum(SHAPE_WEIGHTS.get(name, 0.0) * value
               for name, value in phi.items())


def _train_step(net, target_net, optimizer, replay, batch_size, gamma,
                device, negamax):
    """从回放缓冲采样一批并做一步梯度下降。"""
    states, actions, rewards, next_states, dones = replay.sample(batch_size)

    xs = torch.stack([state_to_tensor(s) for s in states]).to(device)
    next_xs = torch.stack([state_to_tensor(s) for s in next_states]).to(device)
    acts = torch.tensor(actions, dtype=torch.long).to(device)
    rs = torch.tensor(rewards, dtype=torch.float32).to(device)
    dn = torch.tensor(dones, dtype=torch.float32).to(device)

    q = net(xs).gather(1, acts.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_xs).max(dim=1).values
        if negamax:
            # 自对弈：next_state 已换成对手视角，价值取反
            target = rs + (1.0 - dn) * gamma * (-next_q)
        else:
            # 固定对手：标准 Q-learning
            target = rs + (1.0 - dn) * gamma * next_q

    loss = F.mse_loss(q, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def evaluate(agent, env, opponent_name, games, agent_color=BLACK):
    """统计 agent 执 agent_color 时，对给定对手的胜率。"""
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
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    mode = sys.argv[2] if len(sys.argv) > 2 else "selfplay"
    if mode not in ("selfplay", "greedy", "random", "mix"):
        raise SystemExit(f"未知 mode '{mode}'，可选: selfplay / greedy / random / mix")

    # 超参数，可按需调整
    batch_size = 64
    gamma = 0.99
    lr = 1e-3
    target_update = 200        # 每多少步把主网络拷贝到目标网络
    replay_capacity = 20000
    start_train = 512          # 回放缓冲攒够多少条才开始训练
    epsilon_start = 0.5
    epsilon_end = 0.05
    eval_every = max(episodes // 20, 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = GomokuEnv()
    agent = DQNAgent(board_size=env.board_size, epsilon=epsilon_start,
                     device=device, seed=0)
    net = agent.net
    target_net = QNetwork(env.board_size).to(device)
    target_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=lr)
    replay = ReplayBuffer(replay_capacity)

    negamax = mode == "selfplay"
    greedy_opp = create_agent("greedy")
    random_opp = create_agent("random")
    opponent = greedy_opp if mode == "greedy" else (
        random_opp if mode == "random" else None)

    def maybe_train(step):
        if len(replay) >= start_train and step % 4 == 0:
            _train_step(net, target_net, optimizer, replay, batch_size,
                        gamma, device, negamax)
        if step % target_update == 0:
            target_net.load_state_dict(net.state_dict())

    print(f"DQN 训练 {episodes} 局（mode={mode}, device={device}）...")
    step = 0
    for episode in range(1, episodes + 1):
        frac = (episode - 1) / max(episodes - 1, 1)
        agent.epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac

        if negamax:
            # 自对弈：单网络执黑白双方
            state = env.reset()
            done = False
            while not done:
                legal = env.get_legal_actions()
                action = agent.select_action(state, legal)
                next_state, reward, done = env.step(action)
                if not done:
                    reward += shape_reward(state, action)
                replay.push(state, action, reward, next_state, done)
                state = next_state
                step += 1
                maybe_train(step)
        else:
            # 固定/课程混合对手：agent 执黑，对手执白
            if mode == "mix":
                # 课程：随训练推进，对手从 random 逐步过渡到 greedy
                opp = greedy_opp if random.random() < (episode - 1) / max(episodes - 1, 1) else random_opp
            else:
                opp = opponent
            env.reset()
            state = _perspective(env, BLACK)
            while True:
                legal = env.get_legal_actions()
                action = agent.select_action(state, legal)
                _, reward, done = env.step(action)
                if done:
                    replay.push(state, action, reward,
                                _perspective(env, BLACK), True)
                    step += 1
                    maybe_train(step)
                    break

                shape = shape_reward(state, action)
                opp_action = opp.select_action(env.get_state(),
                                               env.get_legal_actions())
                _, _, done = env.step(opp_action)
                if done:
                    r = -1.0 if env.winner == -BLACK else 0.0
                    replay.push(state, action, r,
                                _perspective(env, BLACK), True)
                    step += 1
                    maybe_train(step)
                    break

                next_state = _perspective(env, BLACK)
                replay.push(state, action, shape, next_state, False)
                step += 1
                maybe_train(step)
                state = next_state

        if episode % eval_every == 0:
            agent.epsilon = 0.0
            wr_greedy = evaluate(agent, env, "greedy", games=40)
            wr_random = evaluate(agent, env, "random", games=40)
            agent.epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac
            print(f"  episode {episode:5d} | vs greedy {wr_greedy:.2f} | "
                  f"vs random {wr_random:.2f}", flush=True)

            # 定期保存检查点，避免长跑中断丢失进度
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(MODEL_PATH))

    # 最终评估并保存
    agent.epsilon = 0.0
    print("\n最终胜率（100 局）:")
    for base in ("random", "greedy"):
        wr = evaluate(agent, env, base, games=100)
        print(f"  vs {base:8s}: 执黑 {wr:.2f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), str(MODEL_PATH))
    print(f"\n模型已保存: {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
