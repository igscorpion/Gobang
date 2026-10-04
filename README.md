# RL Gomoku

本项目是一个以强化学习为核心的五子棋课程项目。当前阶段提供 **Environment、Agent 接口、Training、Evaluation** 基础框架，以及 **Vue + FastAPI** 人机对战 Demo。

当前 **没有** 实现 Q-Learning、SARSA、Monte Carlo、DQN 等具体算法。网页对战使用 `RandomAgent`（从合法位置随机落子），用来打通前后端。

## 项目架构

```text
rl/                         强化学习核心（可脱离 Web 独立运行）
├── environment/            9x9 五子棋 Environment
├── agents/                 BaseAgent + RandomAgent（算法在此扩展）
├── training/               通用训练循环（不含更新公式）
├── evaluation/             胜率统计框架
└── models/                 预留给以后 save/load 的模型文件

web/                        展示层
├── backend/                FastAPI
└── frontend/               Vue 3 + Vite
```

- `rl/` 负责 Environment、Agent、Training、Evaluation。
- `web/` 负责 FastAPI 与 Vue。人机对战时，玩家执黑，服务端用 Agent 执白。

更细的数据流见 [docs/architecture.md](docs/architecture.md)，日常命令见 [docs/development.md](docs/development.md)。

## 如何下载

```bash
git clone <repository-url>
cd rl-gomoku
```

若本地仓库目录名不是 `rl-gomoku`，`cd` 到实际克隆下来的目录即可。后续命令都在**项目根目录**执行。

依赖：

- Python 3.10+
- Node.js 18+（用于前端）
- macOS 或 Windows（不依赖 Linux / Docker / WSL）

## 如何安装

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r web/backend/requirements.txt
cd web/frontend
npm install
cd ../..
```

也可以：

```bash
make install
```

### Windows（PowerShell）

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r web/backend/requirements.txt
cd web\frontend
npm install
cd ..\..
```

如果 `Activate.ps1` 报执行策略错误，可先对当前用户放行本地脚本（只需做一次）：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后再执行 `.venv\Scripts\Activate.ps1`。

若 `npm install` 长时间无响应（macOS / Windows 均适用），可换国内镜像后再装：

```bash
npm install --registry=https://registry.npmmirror.com
```

## 如何运行

以下 Python 命令默认已经激活虚拟环境。未激活时，把 `python` 换成 macOS 的 `.venv/bin/python`，或 Windows 的 `.venv\Scripts\python.exe`。

### 运行 Environment

```bash
python -m rl.environment.gomoku
```

成功会打印 `GomokuEnv self-check passed.`

### 运行 Training

```bash
python -m rl.training.train
```

当前没有可学习算法，只跑 `RandomAgent` 的训练循环框架。

### 运行 Evaluation

```bash
python -m rl.evaluation.evaluate
```

### 运行后端

在项目根目录：

```bash
uvicorn web.backend.main:app --reload --host 127.0.0.1 --port 8000
```

macOS：`make backend`  
接口文档：http://127.0.0.1:8000/docs

### 运行前端

```bash
cd web/frontend
npm run dev
```

macOS：`make frontend`  
浏览器：http://127.0.0.1:5173

前端用浏览器 `fetch()` 访问 `http://127.0.0.1:8000`。请先启动后端，并保持两个服务同时运行。

### 同时启动前后端

**macOS**

```bash
make install
make dev
```

`make dev` 会启动 FastAPI 与 Vite。Windows 默认没有 `make`，请不要把它当作 Windows 的启动方式。

**Windows PowerShell**

```powershell
.\start.ps1
```

或按上面「运行后端 / 运行前端」分别开两个终端。

## 人机对战验收路径

```text
打开 http://127.0.0.1:5173
      ↓
看到 9x9 棋盘
      ↓
点击交叉点（黑棋）
      ↓
FastAPI 校验并 env.step()
      ↓
RandomAgent 选择合法位置
      ↓
白棋出现在棋盘上
```

## 如何实现新的 RL 算法

以后实现 Q-Learning、SARSA、Monte Carlo、DQN 时，在 `rl/agents/` **新增文件**，例如 `rl/agents/q_learning.py`，继承 `BaseAgent`：

```python
from rl.agents.base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def select_action(self, state, legal_actions):
        ...

    def update(self, state, action, reward, next_state, done):
        ...

    def save(self, path):
        ...

    def load(self, path):
        ...
```

这些信号从哪里来：

| 名称 | 含义 | 来源 |
| --- | --- | --- |
| `state` | 相对当前玩家的 9x9 棋盘（`1` 自己，`-1` 对手，`0` 空） | `env.get_state()` |
| `legal_actions` | 可落子动作列表，动作为 `row * 9 + col` | `env.get_legal_actions()` |
| `action` | Agent 选出的落子 | `agent.select_action(...)` |
| `reward` | 刚落子一方的即时奖励：胜 `+1`，和 `0`，未结束 `0` | `env.step(action)` |
| `next_state` | 落子之后的相对棋盘（未结束时视角会换成对手） | `env.step(action)` |
| `done` | 是否终局 | `env.step(action)` |

完整 RL 数据流：

```text
Environment
      ↓
state
      ↓
Agent.select_action()
      ↓
action
      ↓
Environment.step()
      ↓
reward + next_state + done
      ↓
Agent.update()
```

**新增 RL 算法时，尽量只修改 `rl/agents/`，不要修改 Environment 和 Web。**  
训练循环已经写在 `rl/training/train.py`；接入网页时，把 `web/backend/main.py` 里的 `RandomAgent()` 换成你的 Agent 即可。

## 以后如何修改项目

| 想改什么 | 去哪里 |
| --- | --- |
| 棋盘规则（大小、五连、奖励） | `rl/environment/gomoku.py` |
| Agent / 新算法 | `rl/agents/` |
| 训练流程 | `rl/training/` |
| 评估 | `rl/evaluation/` |
| HTTP API | `web/backend/main.py` |
| 网页 | `web/frontend/src/` |

## 当前阶段明确不做

不包含：具体 RL 算法、神经网络、数据库、登录、Docker、云部署、在线多人对战。这些留给后续迭代。
