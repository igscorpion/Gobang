"""起手演示：训练 Q-Learning Agent，并和多个基准对局看效果。

Agent 固定执黑学习，对手策略固定，因此是稳定的单智能体 Q-Learning。
特征相对当前玩家，学到的权重同样可用于执白评估。

用法（在项目根目录）：
    uv run python -m rl.demo_qlearning [episodes] [games] [opponent]

    episodes   训练局数，默认 1000
    games      每次评估局数，默认 100
    opponent   训练对手：random（默认）| greedy

说明：线性特征 + Q-Learning 的上限约等于贪心启发式，因此对 greedy 训练
会退化（学不过它）；默认对 random 训练用于产出可部署的模型。
"""

from __future__ import annotations

import sys
from pathlib import Path

from rl.agents.q_learning import QLearningAgent
from rl.agents.registry import create_agent
from rl.environment.gomoku import BLACK, WHITE, GomokuEnv

# 项目根目录 = 本文件上级目录的上级目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_vs(agent, env, opponent, episodes, eps_start=0.5, eps_end=0.05,
             eval_every=None, progress=None):
    """Agent 执黑，与固定对手对弈并在线学习；返回按局记录的胜率曲线。"""
    curve = []
    if eval_every is None:
        eval_every = max(episodes // 10, 1)

    for ep in range(1, episodes + 1):
        frac = (ep - 1) / max(episodes - 1, 1)
        agent.epsilon = eps_start + (eps_end - eps_start) * frac

        state = env.reset()
        while True:
            # Agent（黑）落子
            legal = env.get_legal_actions()
            action = agent.select_action(state, legal)
            _, reward, done = env.step(action)
            if done:
                agent.update(state, action, reward, None, True)
                break

            # 对手（白）落子
            white_state = env.get_state()
            white_action = opponent.select_action(white_state, env.get_legal_actions())
            _, _, done = env.step(white_action)
            if done:
                reward = -1.0 if env.winner == WHITE else 0.0
                agent.update(state, action, reward, None, True)
                break

            next_state = env.get_state()
            agent.update(state, action, 0.0, next_state, False)
            state = next_state

        if ep % eval_every == 0:
            cur_eps = agent.epsilon
            agent.epsilon = 0.0
            wr = evaluate_vs(agent, env, opponent, games=40, agent_color=BLACK)
            agent.epsilon = cur_eps
            curve.append((ep, wr))
            if progress:
                progress(ep, wr)
    return curve


def evaluate_vs(agent, env, opponent, games, agent_color):
    """统计 agent 执 agent_color 一方时，对给定对手的胜率。"""
    wins = 0
    for _ in range(games):
        state = env.reset()
        done = False
        while not done:
            legal_actions = env.get_legal_actions()
            if env.current_player == agent_color:
                action = agent.select_action(state, legal_actions)
            else:
                action = opponent.select_action(state, legal_actions)
            state, _reward, done = env.step(action)
        if env.winner == agent_color:
            wins += 1
    return wins / games


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    opponent_name = sys.argv[3] if len(sys.argv) > 3 else "random"

    env = GomokuEnv()
    agent = QLearningAgent(epsilon=0.5, alpha=0.1, gamma=0.99, seed=0)
    opponent = create_agent(opponent_name)

    print(f"Q-Learning 执黑对 {opponent_name} 训练 {episodes} 局...")
    train_vs(
        agent, env, opponent, episodes,
        progress=lambda ep, wr: print(f"  episode {ep:5d} | 对 {opponent_name} 胜率 {wr:.2f}"),
    )

    # 最终评估：关闭探索，分别执黑 / 执白，对 random 与 greedy 两个基准
    agent.epsilon = 0.0
    print(f"\n最终胜率（{games} 局）:")
    for base_name in ("random", "greedy"):
        base = create_agent(base_name)
        wr_black = evaluate_vs(agent, env, base, games, BLACK)
        wr_white = evaluate_vs(agent, env, base, games, WHITE)
        print(f"  vs {base_name:8s}: 执黑 {wr_black:.2f} / 执白 {wr_white:.2f}")

    # 打印学到的权重，便于理解 Agent 学到了什么
    print("\n学习到的特征权重（按大小排序）:")
    for name, value in sorted(agent.w.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:12s} {value:+.3f}")

    # 保存权重，供 web/backend 加载后做人机对战
    model_path = PROJECT_ROOT / "rl" / "models" / "q_learning.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(model_path))
    print(f"\n模型已保存: {model_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
