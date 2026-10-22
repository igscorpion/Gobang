# 黑方禁手：定义与边界

默认 `freestyle`：双方五连及以上胜。`renju`：白方无禁手，五连及以上胜；黑方精确五连胜，其他落子若形成长连、双四或真实双活三则禁手。

## 判定

- 长连：同方向连续六子及以上。
- 四：下一手可成精确五连的四颗黑子；按棋子集合去重，一个活四的两个胜点仍是一个四。支持同方向两个不同的四。
- 真实活三：可通过合法延伸形成两端均可成五的直四。边缘、对手阻挡、延伸点自身为长连/双四/禁双三时，不简单算成真实活三。延伸合法性递归检查。
- 精确五连优先：同手成精确五连时判胜，即使别的方向同时形成禁手图形；依据 [RIF 规则第 9.2 条](https://www.renju.net/rifrules/)。

共用实现 `rl/environment/rules.py`；环境合法动作、MCTS/Pure MCTS/Minimax 各层均调用该规则。环境收到禁手会拒绝，并保持棋盘与回合不变；训练和 AI 从合法集合选择。

## 适用范围

这是 **9×9 的黑方禁手实验模式**，不是完整 15×15 比赛连珠。未提供比赛开局协议、交换、pass 等。为保证有限棋盘的仿真可终止，盘满无人成五判和，非满盘但当前玩家无合法动作判负。这是项目额外约定。

黑方禁手落点由 API 的 `forbidden_moves` 返回。网页规则下拉切换后重开棋局，已有权重不会因切规则而重新训练。

`a2c4_tactical` 的全部战术候选同样服从该规则，详情见 [A2C 战术增强](a2c-tactics.md)。活三识别用于禁手合法性时与“是否应该防守对方活三”的策略判断分开；不会将单个活三一律设为强制防守。

## 使用

```powershell
$env:GOMOKU_RULES='renju'
.\.venv\Scripts\python.exe -m rl.evaluation.eval_alphazero 20 400
```

新训练器/规则比较也可直接用参数：

```powershell
.\.venv\Scripts\python.exe -m rl.training.local_dqn --rules renju --output runs/dqn4_renju --episodes 30000 --device cuda --resume
.\.venv\Scripts\python.exe -m rl.evaluation.compare_rules --algorithm greedy --games 300 --output runs/rule_comparison_manual
```

独立 `--rules` 训练不要覆盖 freestyle 输出。禁手单测覆盖旋转、断三、边界伪三、不可合法延伸的伪三、同向双四、长连、精确五连优先和白方豁免。
