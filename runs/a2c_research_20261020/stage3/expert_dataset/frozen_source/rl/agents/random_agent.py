"""随机 Agent：仅用于打通训练框架和 Web 人机对战。"""

from __future__ import annotations

import random

from rl.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_action(self, state, legal_actions):
        if not legal_actions:
            raise ValueError("没有合法动作可选择")
        return self.rng.choice(list(legal_actions))

    def update(self, state, action, reward, next_state, done):
        return None


if __name__ == "__main__":
    from rl.environment.gomoku import GomokuEnv

    env = GomokuEnv()
    agent = RandomAgent(seed=0)
    state = env.reset()
    action = agent.select_action(state, env.get_legal_actions())
    next_state, reward, done = env.step(action)
    agent.update(state, action, reward, next_state, done)
    print(f"RandomAgent played action={action}, reward={reward}, done={done}")
