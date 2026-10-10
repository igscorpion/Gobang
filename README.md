# RL Gomoku

本项目是一个以强化学习为核心的五子棋课程项目，包含 **9×9 五子棋环境、多种 Agent、训练与评估脚本**，以及 **Vue + FastAPI** 人机对战 Demo。

已实现的具体算法：**线性 Q-Learning、DQN、AlphaZero-lite**；另含 `GreedyAgent`（手工贪心启发式）与 `RandomAgent`（随机落子）两个基准。网页对战时可通过下拉框切换所用 Agent。各算法的实现要点、实验结果与当前状态见 [rl/README.md](rl/README.md)。

## 项目架构

```text
rl/                         强化学习核心（可脱离 Web 独立运行）
├── environment/            9x9 五子棋 Environment
├── agents/                 BaseAgent + Random / Greedy / QLearning / DQN / AlphaZero + registry
├── training/               训练脚本：train.py（通用循环）、train_dqn.py、train_alphazero.py
├── evaluation/             胜率统计框架
└── models/                 训练产物：权重、检查点、累计局数元数据

web/                        展示层
├── backend/                FastAPI
└── frontend/               Vue 3 + Vite
```

- `rl/` 负责 Environment、Agent、Training、Evaluation。
- `web/` 负责 FastAPI 与 Vue。人机对战时，玩家执黑，服务端用 Agent 执白。

更细的数据流见 [docs/architecture.md](docs/architecture.md)，日常命令见 [docs/development.md](docs/development.md)。

## 快速开始（推荐）

在使用 `make` 命令之前，必须先确认前置环境已经安装好，尤其是 `uv`。这个项目的 `Makefile` 依赖 `uv`，所以不能跳过“安装 uv / 检查 uv 可用性”这一步。

### 1) 确认前置环境

```bash
git --version
python3 --version
uv --version
node -v
npm -v
```

如果 `uv --version` 报错，先安装 `uv`，再继续执行后面的步骤。

### 2) 从 Git 拉取代码后，执行最简流程

#### macOS / Linux

```bash
git pull
make install
make dev
```

#### Windows PowerShell

```powershell
git pull
.\start.ps1
```

然后在浏览器打开：

```text
http://127.0.0.1:5173
```

> 关键点：`make install` 本身会执行 `uv sync --python 3.11`，因此必须先确保 `uv` 已安装并且可运行；否则 `make` 会失败。`make dev` 会同时启动 FastAPI 与 Vite，不需要手动分别开两个终端。

## 这个项目是什么

这是一个前后端分离的强化学习应用。前端负责界面与交互，后端则是一个 Python 程序，提供 HTTP API、加载强化学习参数、执行推理和状态更新。这个项目的核心重点不是“直接把所有逻辑写进网页”，而是把“训练逻辑”和“部署逻辑”分开。

- 前端：负责绘制 9x9 棋盘、处理点击事件、展示回合状态和胜负结果。
- 后端：在 `web/backend/main.py` 里启动 FastAPI 服务，调用 `rl.environment.gomoku` 中的棋盘逻辑，并通过 `rl/agents/registry.py` 选择所用 Agent（默认 `q_learning`，可用环境变量 `GOMOKU_AI` 指定）。
- 强化学习部分：在 `rl/` 目录里定义环境、Agent、训练和评估框架。

真正的部署/演示场景里，通常不需要重头训练模型。一般做法是：先用训练脚本产出模型参数文件，再在后端启动时加载这些参数；前端只负责调用接口，展示结果。也就是说，训练和部署是两件事：

- 训练：产出模型参数，通常需要 GPU、仿真器、训练数据和较长时间。
- 部署/演示：加载已经训练好的参数，启动后端接口，前端直接请求接口并展示 JSON 结果。

整体数据流可以概括为：

```text
用户 -> 前端 -> HTTP 请求 -> 后端 Python -> 加载强化学习参数 -> 返回 JSON -> 前端展示
```

当前仓库的典型目录结构如下：

```text
.
├── README.md
├── pyproject.toml          # Python 依赖与 uv 入口
├── .python-version         # 统一 Python 版本（3.11）
├── environment.yml         # conda 环境文件
├── Makefile                # 本地快捷命令
├── start.ps1               # Windows 一键启动脚本
├── rl/                     # 强化学习核心代码
│   ├── agents/
│   ├── environment/
│   ├── evaluation/
│   ├── training/
│   └── models/
├── web/
│   ├── backend/
│   └── frontend/
└── docs/
```

启动顺序一般是：先启动后端 Python 程序，再启动前端；如果只是改页面样式或交互，不一定要重新训练。只有在更换算法、更新权重文件、改训练逻辑时，才需要重新评估“是否需要重新训练”。

### 常见问题

- 只改界面要不要训练？
  - 通常不需要。只调整前端样式、按钮文案、渲染逻辑，不会影响后端的 Python 运行环境。
- 换模型参数要不要改代码？
  - 取决于参数文件的加载方式：如果后端代码里定义了固定文件名或路径，通常需要同步更新加载逻辑；若只是替换参数文件并且接口不变，代码本身不一定需要大改。
- 前端能不能直接读参数？
  - 不能。前端只负责展示，真实的模型参数和网络权重通常放在后端服务端文件中，前端不应该直接访问本地参数文件。
- 合作者新增依赖怎么办？
  - 先拉取最新代码，再按项目要求执行 `uv add ...`（Python）或 `npm install ...`（前端）；不要直接把本地 `.venv` 提交到仓库。
- 为什么不要提交 `.venv`？
  - `.venv` 是本地开发环境，依赖平台和 Python 版本，通常不通用；仓库里应该保留声明式配置，如 `pyproject.toml`、`uv.lock`、`environment.yml`，让每个人用同一套依赖重建环境。

## 前置环境

在开始前，建议先确认以下工具已安装，并且 `uv` 可正常运行：

- Git
- Node.js / npm
- Python 3.11（推荐）
- uv（推荐，且必须先安装）
- conda（备选）

在执行 `make install` 之前，一定先运行：

```bash
uv --version
```

如果它报错，说明还没安装 `uv`，先安装 `uv`，再使用 `make install`。这个项目的 `Makefile` 依赖 `uv`，不能跳过这一步。

验证命令：

```bash
git --version
node -v
npm -v
python3 --version
uv --version
conda --version
```

如果你用的是 Windows PowerShell，命令形式基本相同；如果某个命令不存在，请先安装对应工具，再继续执行后续步骤。

如果项目后续要接入 GPU/CUDA 训练，需确认 NVIDIA 驱动和 CUDA 版本符合你的训练环境。当前仓库的本地演示和前后端交互并不强依赖 GPU。

## Python 环境：uv 为主，conda 备选

本仓库推荐使用 `uv`，因为它与 `pyproject.toml` 与 `uv.lock` 配合得比较好，可以直接在项目根目录执行。这里的“统一 Python 版本”是 3.11，使用前请确保本机 Python 版本与 `.python-version` 一致。

### 方式 A：uv（推荐）

在项目根目录执行：

```bash
cd /path/to/Gobang
uv sync --python 3.11
uv run python -m rl.environment.gomoku
```

预期结果：

- `uv sync` 会根据 `pyproject.toml` 创建/更新本地 `.venv`。
- `uv run python -m rl.environment.gomoku` 应输出 `GomokuEnv self-check passed.`

如果要启动后端：

```bash
cd /path/to/Gobang
uv run uvicorn web.backend.main:app --reload --host 127.0.0.1 --port 8000
```

后端接口文档： http://127.0.0.1:8000/docs

### 方式 B：conda（备选）

如果你更习惯 conda，可以在项目根目录执行：

```bash
cd /path/to/Gobang
conda env create -f environment.yml
conda activate gobang
python -m rl.environment.gomoku
```

预期结果：

- conda 环境创建成功；
- 运行 `python -m rl.environment.gomoku` 后，输出与上面一致。

### uv 与 conda 的区别

- `uv`：不需要手动 `activate`，直接用 `uv run` 自动使用项目内 `.venv`。
- `conda`：需要先 `conda activate 环境名`，然后再运行 `python` 或 `uvicorn`。
- `uv` 依赖 `pyproject.toml` 和 `uv.lock`；`conda` 通常依赖 `environment.yml`。
- 不要把 `conda` 激活的环境和 `uv` 创建的 `.venv` 混用，否则很容易出现“解释器不一致”的问题。

## 统一 Python 版本

为了避免“本地 Python 3.10、CI 3.11、README 写 3.12”的混乱，本项目统一使用 Python 3.11。所有和 Python 版本相关的配置都需要保持一致：

- `README.md`
- `pyproject.toml`
- `.python-version`
- `environment.yml`
- `web/backend/requirements.txt`
- `Makefile` / `start.ps1` 中的启动说明

如果 README 改成了 3.11，那么对应的配置文件也必须一起改。这样每个人都能在同一版本上复现环境。

## 从零重建虚拟环境

如果本地 `.venv` 出现异常、版本冲突，或者准备彻底重建环境，请在项目根目录执行：

### macOS / Linux

```bash
cd /path/to/Gobang
rm -rf .venv
uv sync --python 3.11
uv run python -m rl.environment.gomoku
```

### Windows PowerShell

```powershell
cd C:\path\to\Gobang
Remove-Item -Recurse -Force .venv
uv sync --python 3.11
uv run python -m rl.environment.gomoku
```

> 注意：执行删除命令前，必须确认你已经进入了项目根目录，并且 `.venv` 确实是当前项目的本地虚拟环境。这里不删本地 `.venv` 的操作由你在本地手动决定；README 只写“清理并重建”的标准步骤。

## 新增依赖 / 合作者同步依赖

### 新增 Python 依赖

```bash
git pull
uv add 包名
git add pyproject.toml uv.lock
git commit -m "chore: add 包名 dependency"
git push
```

`uv add` 会同时更新 `pyproject.toml` 和 `uv.lock`。合作者拉到最新代码后，执行：

```bash
git pull
uv sync
uv run python -m rl.environment.gomoku
```

`uv sync` 会对比远端声明和本地 `.venv`，只做增量安装、删除或者版本更新，不需要手动重新安装整个环境。

### 前端新增 npm 依赖

```bash
cd web/frontend
npm install
npm install 包名
git add package.json package-lock.json
git commit -m "chore: add npm dependency"
```

其他人拉取代码后：

```bash
cd web/frontend
npm install
```

### 为什么不要提交 `.venv`

`.venv` 是本地开发环境，不应被提交到 Git。它会因操作系统、Python 版本、平台差异而产生不同结果，甚至造成合作者环境不一致。仓库里应该保留可复现的声明式依赖文件（例如 `pyproject.toml`、`uv.lock`、`environment.yml` 和 `package.json`），然后大家一起执行 `uv sync` 或 `npm install`。

## 如何安装与运行

### 安装依赖

#### 方式 A：uv（推荐）

```bash
cd /path/to/Gobang
uv sync --python 3.11
cd web/frontend
npm install
cd ../..
```

#### 方式 B：conda

```bash
cd /path/to/Gobang
conda env create -f environment.yml
conda activate gobang
cd web/frontend
npm install
cd ../..
```

### 运行环境自检

```bash
cd /path/to/Gobang
uv run python -m rl.environment.gomoku
```

成功会打印：

```text
GomokuEnv self-check passed.
```

### 运行训练

训练脚本依赖 PyTorch；本机 GPU 版 torch 装在系统 Python 3.11 中（`.venv` 内为 CPU 版，也能运行但较慢），因此请用装有 torch 的解释器执行：

```bash
# AlphaZero-lite 自对弈（无限训练，Ctrl+C 停止，再次运行自动续训）
python -m rl.training.train_alphazero

# DQN
python -m rl.training.train_dqn
```

Windows 下也可直接双击项目根目录的 `训练AlphaZero.bat` / `训练DQN.bat`。训练产物见 `rl/models/`，各算法现状见 [rl/README.md](rl/README.md)。

### 运行评估

```bash
cd /path/to/Gobang
uv run python -m rl.evaluation.evaluate
```

### 运行后端

在项目根目录执行：

```bash
cd /path/to/Gobang
uv run uvicorn web.backend.main:app --reload --host 127.0.0.1 --port 8000
```

接口文档： http://127.0.0.1:8000/docs

### 运行前端

```bash
cd /path/to/Gobang/web/frontend
npm run dev
```

浏览器打开： http://127.0.0.1:5173

前端通过浏览器调用 `http://127.0.0.1:8000`，请先启动后端，再打开前端页面；前后端两个服务需要同时运行。

### 同时启动前后端

**macOS / Linux**

```bash
cd /path/to/Gobang
make install
make dev
```

**Windows PowerShell**

```powershell
cd C:\path\to\Gobang
.\start.ps1
```

如果你只是在本地修改前端页面，不需要重新训练；只要后端接口不变，通常只需要重启前端和后端即可。

## 人机对战验收路径

```text
打开 http://127.0.0.1:5173
      ↓
看到 9x9 棋盘
      ↓
点击交叉点（黑棋）
      ↓
FastAPI 校验并调用 env.step()
      ↓
所选 Agent 选择合法位置（下拉框可切换）
      ↓
白棋出现在棋盘上
```

如果该路径不能跑通，先检查后端是否成功启动，再确认前端的 `API_BASE` 是否指向 `http://127.0.0.1:8000`（端口被占用时，可在 `web/frontend/.env` 中用 `VITE_API_PORT` 覆盖）。该项目的前端默认访问后端，不依赖单独的 WebSocket 服务。

## 如何新增 RL 算法

在 `rl/agents/` **新增文件**，继承 `BaseAgent`：

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
在 `rl/agents/registry.py` 的 `REGISTRY` 中注册后，网页下拉框即可直接选择该 Agent；训练脚本可参照 `rl/training/` 下的已有实现。

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

不包含：数据库、登录、Docker、云部署、在线多人对战。这些留给后续迭代。
