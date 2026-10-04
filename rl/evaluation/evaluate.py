"""评估框架：统计胜 / 负 / 和，不更新 Agent 参数。"""

from __future__ import annotations


def evaluate(agent, env, episodes: int = 20, opponent=None, verbose: bool = True) -> dict:
    """默认：agent 执黑先手，对手执白。"""
    if opponent is None:
        from rl.agents.random_agent import RandomAgent

        opponent = RandomAgent()

    wins = 0
    losses = 0
    draws = 0

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            legal_actions = env.get_legal_actions()
            if env.current_player == env.BLACK:
                action = agent.select_action(state, legal_actions)
            else:
                action = opponent.select_action(state, legal_actions)
            state, _reward, done = env.step(action)

        if env.winner == env.BLACK:
            wins += 1
            result = "win"
        elif env.winner == env.WHITE:
            losses += 1
            result = "loss"
        else:
            draws += 1
            result = "draw"

        if verbose:
            print(f"episode={episode + 1}/{episodes} result={result}")

    total = max(episodes, 1)
    summary = {
        "episodes": episodes,
        "win": wins,
        "loss": losses,
        "draw": draws,
        "win_rate": wins / total,
    }
    if verbose:
        print(
            f"win={wins} loss={losses} draw={draws} "
            f"win_rate={summary['win_rate']:.2f}"
        )
    return summary


def main() -> None:
    from rl.agents.random_agent import RandomAgent
    from rl.environment.gomoku import GomokuEnv

    env = GomokuEnv()
    agent = RandomAgent(seed=2)
    print("Evaluating RandomAgent vs RandomAgent.")
    evaluate(agent, env, episodes=5)


if __name__ == "__main__":
    main()
