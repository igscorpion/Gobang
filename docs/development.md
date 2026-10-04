# 开发说明

请在项目根目录执行命令，这样 `python -m rl...` 才能找到包。

## 环境自检

```bash
python -m rl.environment.gomoku
```

会跑一轮环境断言：reset、非法落子、横/竖/斜五连、棋盘填满和棋。

## 训练 / 评估框架

```bash
python -m rl.training.train
python -m rl.evaluation.evaluate
```

当前只验证循环能跑通，`RandomAgent.update` 是空操作。

## Web 本地开发

终端 1：

```bash
uvicorn web.backend.main:app --reload --host 127.0.0.1 --port 8000
```

终端 2：

```bash
cd web/frontend
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`。

macOS 可用 `make dev`；Windows 可用 `.\start.ps1`。不要把 Makefile 当成 Windows 的唯一启动方式。

## 实现新算法时改哪里

| 目标 | 文件 |
| --- | --- |
| 棋盘大小 / 五连规则 | `rl/environment/gomoku.py` |
| 新 RL 算法 | `rl/agents/` 新增文件，继承 `BaseAgent` |
| 训练循环 | `rl/training/train.py` |
| 评估指标 | `rl/evaluation/evaluate.py` |
| 对战 API | `web/backend/main.py` |
| 页面与棋盘 UI | `web/frontend/src/` |

新增算法时尽量只改 `rl/agents/`。Environment 的 `state / action / reward / done` 约定不要随意破坏，否则训练与 Web 会同时出错。
