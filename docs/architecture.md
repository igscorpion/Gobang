# 架构说明

本项目分成两层：强化学习核心与 Web 展示。当前阶段只提供可运行骨架，不包含具体 RL 算法。

## 分层

```text
浏览器 (Vue, :5173)
        ↓ fetch
FastAPI (:8000)
        ↓
GomokuEnv + RandomAgent
```

```text
rl/          核心：Environment / Agent / Training / Evaluation
web/         展示：FastAPI + Vue
docs/        文档
```

## 数据流（训练）

```text
Environment.reset()
      ↓
state（相对当前玩家：1 自己，-1 对手）
      ↓
Agent.select_action(state, legal_actions)
      ↓
action（row * 9 + col）
      ↓
Environment.step(action)
      ↓
reward + next_state + done
      ↓
Agent.update(...)
```

## 数据流（人机对战）

1. 玩家在网页点击交叉点。
2. `POST /game/move` 把 `(row, col)` 交给 FastAPI。
3. 后端检查合法性并调用 `env.step`（玩家固定执黑）。
4. 若未结束，`RandomAgent.select_action` 选择白棋落子。
5. 返回绝对棋盘（1 黑，-1 白）给 Vue 渲染。

Web 使用绝对颜色，是为了棋盘显示稳定；Agent 训练仍使用 `get_state()` 的相对视角。

## 当前 Agent

Web 与训练演示都使用 `RandomAgent`。把新算法放到 `rl/agents/` 后，只需在 `web/backend/main.py` 里替换默认 Agent 实例即可接入网页，无需改前端。
