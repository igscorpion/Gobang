# RL Gomoku

本项目是一个以强化学习为核心的五子棋课程与实验项目，包含 **9×9 五子棋环境、多种 Agent、训练与评估脚本**，以及 **Vue + FastAPI** 人机对战 Demo。

已实现 **线性 Q-Learning、DQN、同步 A2C、AlphaZero-lite**，另含 Random、Greedy、Minimax 和纯 MCTS 基准。网页通过下拉框切换算法，支持无禁手与黑方禁手规则。

本分支 `A2C` 基于原仓库提交 `2711cca`，汇总后续容量探索、A2C实现与增强研究。具体改动见 [本轮更新](docs/changes-20261022.md)，完整过程见 [性能增强研究总报告](runs/a2c_research_20261020/final_report.md)。上游已有算法、历史权重及历史文档继续保留，研究结果与其来源分别说明。

## 本轮更新

- **容量与状态**：AlphaZero第三层卷积由64扩至128，提供有SHA记录的旧权重部分迁移；DQN采用四通道状态、合法动作掩码和正确的零和回报视角。
- **A2C**：新增同步8环境、5-step Actor-Critic，完成三个训练种子；原模型 `a2c4` 独立保留。
- **战术增强**：`a2c4_tactical` 在原权重上增加直接胜、急防、双威胁和有限VCF，无需重训；与纯网络结果分开报告。
- **研究增强版**：`a2c_residual10_guarded` 使用AZ专家动作CE预训练、十通道特征、空间残差网络和A2C/CE联合训练。推理保留战术保护，普通局面由网络在完整合法集上选择动作。
- **规则与评估**：共用合法性接口、分色W/D/L、Wilson区间、验证选模型、冻结后新测试、来源SHA及逐手独立审计。
- **负结果**：保留强对手课程、部分浅搜/value混合、非单调微调与候选过筛等未奏效尝试，解释其适用边界。

## 最终实验结果

主方案为 `residual10 + guarded`，三个真实训练种子11/22/33，分别按开发集选中30k/15k/25k checkpoint。每个模型每规则每色对Greedy各100局；胜率不含和棋，±为训练种子之间的样本标准差，单位为百分点。

| 规则/执子 | 新guarded胜率均值 ± SD | 同checkpoint纯策略均值 | 旧战术增强同批测试 |
|---|---:|---:|---:|
| 无禁手/黑 | 92.00% ± 4.58 | 87.67% | 53% |
| 无禁手/白 | 78.67% ± 5.03 | 75.33% | 40% |
| 禁手/黑 | 53.00% ± 11.53 | 44.00% | 58% |
| 禁手/白 | 82.33% ± 7.64 | 79.67% | 59% |

无禁手两色等权平均胜率85.33%，禁手67.67%。**禁手黑方没有超过旧增强参考，且训练种子差异较大。** policy与guarded分别对Random完成1,200/1,200胜；有限样本全胜不证明任意种子必胜。

当前范围31项、6,200局已通过独立审计：164,639步合法重放，81,164个Greedy/Random动作精确复现。原计划中的AZ400禁手参考按研究发起者要求取消，partial不计入成绩；原冻结清单保留。展示权重在测试前已固定为seed11@30k，不根据最终测试换模型。

需要注意：

- guarded包含有限VCF等战术/证明计算；纯策略也使用规则特征和教师监督，提升不能全部归因于A2C更新公式。
- renju是无禁手训练权重的规则感知迁移；本项目9×9协议不包含完整赛事开局制度，成绩不能换算人类段位。
- A2C与AZ仅有各自对Greedy的参考成绩，没有直接对弈优劣结论。
- 每个训练模型每色给Wilson95，跨训练种子给样本SD；不将共享游戏种子的300局当完全独立样本合并区间。

逐种子W/D/L、区间、计算成本及失败分析见 [总报告](runs/a2c_research_20261020/final_report.md) 和 [完整技术统计](runs/a2c_research_20261020/final_report_technical_20261022.md)。

## 项目架构

```text
rl/
├── environment/     棋盘、四通道状态、共用禁手合法性
├── agents/          各算法、战术/搜索、研究网络、模型注册表
├── training/        A2C/DQN/AZ训练、专家数据和分阶段调度
├── evaluation/      分色评估、开发选择、冻结测试
└── models/          上游已有推理权重与历史元数据
web/
├── backend/         FastAPI
└── frontend/        Vue 3 + Vite
runs/                选定推理权重、研究报告与对局证据快照
docs/                算法、规则、研究口径、更新说明
tests/               规则、训练数学、注册表、评估和网页测试
tools/               独立审计及报告工具
```

前端展示棋盘和算法/规则选项，后端调用Agent；推理无需先重新训练。玩家执黑，AI执白。网页选项包括原来的 `a2c4`、`a2c4_tactical` 和独立的新方案 `a2c_residual10_guarded`。

## 快速开始

本轮验证环境为Windows、Python3.11、Node.js/npm、uv。依赖锁使用PyTorch2.9.1/CUDA12.8；CPU环境运行推理时设置 `GOMOKU_AI_DEVICE=cpu`。纯CPU或其他平台若无法安装锁定的Torch来源，需按官方安装方式调整 `pyproject.toml` 的Torch索引后重新锁定依赖。

```powershell
git clone --branch A2C https://github.com/igscorpion/Gobang.git
cd Gobang
uv sync --python 3.11 --extra test
cd web/frontend
npm ci
cd ../..
```

Windows启动：

```powershell
$env:GOMOKU_AI = 'a2c_residual10_guarded'
$env:GOMOKU_AI_DEVICE = 'cpu'
.\start.ps1
```

启动脚本选择空闲端口并打印实际网页地址，只管理自己启动的后端。也可双击 `启动网页对战.bat`，在页面选择模型。切换规则会重新开局。

手动分两个终端启动：

```bash
uv run uvicorn web.backend.main:app --host 127.0.0.1 --port 8000
```

```bash
cd web/frontend
npm run dev -- --host 127.0.0.1
```

默认网页为 `http://127.0.0.1:5173`，API为 `http://127.0.0.1:8000`。改后端端口时，在前端同步设置 `VITE_API_PORT`。研究报告中5193/8020是当时本地隔离验证端口，不是远程托管地址。

## 模型与发布范围

本分支附带网页所需的选定推理权重：上游模型、DQN4、扩容AZ、原A2C及新研究模型，并保留最终比较所需的其他选定权重和专家数据。新展示模型固定SHA256为：

```text
dea2fee0d128485ed20461f7ce5e793f4a9a193870af068ee4a03c9940c5b898
```

代码、推理权重和已发布研究记录可直接阅读或运行；`.venv`、`node_modules`、全部中间模型、优化器/RNG续训checkpoint及机器进程日志不提交。它是推理/代码/报告发布，不是完整续训归档。复制文件的来源SHA及排除范围见 [发布清单](docs/release-publication.json)。

原始冻结/审计文件保留原字节，内部可能记录当时的Windows绝对路径；这些记录不用于网页加载模型。当前模型目录使用项目相对路径。完整独立审计若需重跑，必须另备完整训练来源归档并重新映射输入路径，不能把发布快照误称为完整训练归档。

## 训练与验证

方法和参数见 [算法说明](rl/README.md)、[A2C原始对照](docs/a2c-control.md)、[改进研究](docs/a2c-research.md) 与 [开发说明](docs/development.md)。已有推理权重可直接使用，不需重复训练。

```bash
uv run python -m rl.environment.gomoku
uv run python -m rl.training.a2c_research --help
uv run python -m rl.training.local_alphazero --help
uv run pytest -q tests/test_local_changes.py tests/test_a2c_core.py tests/test_a2c_guarded.py tests/test_a2c_research_registry.py
```

新训练使用独立输出目录，保存真实种子、配置及完整状态。当前最终测试种子已用于评估；后续改进需要新版本和全新测试协议。

## 阅读路线与来源

- [本轮更新说明](docs/changes-20261022.md)：实现、实验结论、已排除方向、测试与限制。
- [完整性能增强研究回顾](runs/a2c_research_20261020/final_report.md)：瓶颈—尝试—结果—原因。
- [最终技术统计](runs/a2c_research_20261020/final_report_technical_20261022.md)：31项分色结果、Wilson95、成本。
- [规则范围](docs/rules.md)、[实验口径](docs/experiments.md)、[上游历史文档](docs/history/upstream-rl-README.md)。

项目源自 [igscorpion/Gobang](https://github.com/igscorpion/Gobang)，本分支继承并扩展其代码与权重。[junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)是AlphaZero五子棋实现参考，本次不向该仓库推送，也不把其代码混作本轮独立实验成果。
