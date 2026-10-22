# 开发与复现

所有命令在项目根目录运行。Windows 本机直接使用 `.venv/Scripts/python.exe`，不依赖系统 py 命令。

## 验证

```powershell
.\.venv\Scripts\python.exe -m rl.environment.gomoku
.\.venv\Scripts\python.exe -m pytest -q --basetemp=../../work/pytest-local-manual
cd web/frontend
node node_modules/vite/bin/vite.js build
```

若临时目录 ACL 限制，请把 basetemp 指向自己可写且专用于测试的目录。

## 独立训练

```powershell
.\.venv\Scripts\python.exe -m rl.training.local_dqn --output runs/dqn4_manual --episodes 30000 --seed 11 --device cuda --resume
.\.venv\Scripts\python.exe -m rl.training.local_alphazero --output runs/az_wide_manual --episodes 2000 --channels 64 64 128 --sims 400 --warm-start backups/upstream-2711cca/models/alphazero.pt --device cpu --resume
```

第二条从旧权重暖启动，检查点存在时优先完整续训。需要从头训练 AZ 则去掉 warm-start。容量消融用 `--channels 64 64 64`；固定课程用 `--fixed-ratio .6`，否则使用验证驱动课程。恢复时只增加总 episodes，不更改种子、结构、规则与关键训练参数。

## 运行网页

`启动网页对战.bat` 调用 start.ps1。使用现有 venv 与 node_modules，缺失时明确提示安装。启动脚本不执行 uv sync、不停止其他项目服务。

前后端手动启动：

```powershell
.\.venv\Scripts\python.exe -m uvicorn web.backend.main:app --host 127.0.0.1 --port 8000
```

另一终端在 web/frontend 使用 `npm run dev`。网页默认后端端口 8000，可设置 VITE_API_PORT；后端支持本机不同前端端口。

## 扩展位置

规则写入 environment/rules.py；状态写入 environment/state.py；Agent 遵守 select_action(state, legal_actions)。新模型注册用 agents/registry.py；训练用 local_dqn/local_alphazero，共用评估 common.evaluate。实验过程写入 docs/experiments.md 或 runs 报告，不堆叠到主 README。

本次 A2C 分支附带经 SHA 核验的选定推理权重、上游兼容备份和专家数据；完整训练状态、中间权重、venv、node_modules 与机器进程日志不提交。发布范围及来源见 [发布清单](release-publication.json)。
