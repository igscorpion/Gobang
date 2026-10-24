# 架构说明

本项目分成两层：强化学习核心与 Web 展示。核心包含五子棋环境、多种 Agent、训练和评估入口；Web 负责提供对战界面。

## 分层

```text
浏览器 (Vue, :5173)
        ↓ fetch
FastAPI (:8000)
      ↓
GomokuEnv + registry 选择的 Agent
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
4. 若未结束，后端通过 `rl/agents/registry.py` 创建的 Agent 选择白棋落子；可由 `GOMOKU_AI` 指定。
5. 返回绝对棋盘（1 黑，-1 白）给 Vue 渲染。

Web 使用绝对颜色，是为了棋盘显示稳定；Agent 训练仍使用 `get_state()` 的相对视角。

## Agent 与 Minimax

Web 对战使用注册表中的 Agent，前端可切换名称。新增算法可继承 `BaseAgent` 并注册到 `rl/agents/registry.py`。

Minimax 实现在 `rl/agents/minimax_agent.py`：采用 Negamax + Alpha-Beta，并复用 Greedy 的棋型特征作为叶节点估值。为控制分支因子，普通走法通过邻域和候选数量作启发式筛选，直接获胜与必须防守对手下一手成五的落点会全部保留；因此它不是全动作空间的完备搜索。运行、参数和对照评估命令见 `rl/README.md`。
