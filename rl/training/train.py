"""通用训练循环。

这里只编排 Environment 与 Agent 的交互，不实现任何具体 RL 更新公式。
Q 值 / 神经网络更新应写在 Agent.update() 中。
"""

from __future__ import annotations


def train(agent, env, episodes: int = 10, verbose: bool = True) -> list[float]:
    """Self-play 训练框架：同一个 agent 执黑白双方。"""
    returns: list[float] = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_return = 0.0

        while not done:
            legal_actions = env.get_legal_actions()
            action = agent.select_action(state, legal_actions)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            episode_return += reward
            state = next_state

        returns.append(episode_return)
        if verbose:
            print(
                f"episode={episode + 1}/{episodes} "
                f"return={episode_return:.1f} winner={env.winner}"
            )

    return returns


def main() -> None:
    from rl.agents.random_agent import RandomAgent
    from rl.environment.gomoku import GomokuEnv

    env = GomokuEnv()
    agent = RandomAgent(seed=1)
    print("Running training loop with RandomAgent (no learning).")
    train(agent, env, episodes=5)


if __name__ == "__main__":
    main()
