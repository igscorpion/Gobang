# 模型保存目录

存放各 Agent 的训练产物（权重、检查点）与训练进度元数据，由训练脚本自动读写，无需手动维护。

```text
rl/models/
├── alphazero.pt            AlphaZero 检查点（当前版本，4 通道输入）
├── alphazero_meta.json     AlphaZero 累计训练局数
├── dqn.pt                  DQN 权重
├── dqn_meta.json           DQN 累计训练局数
├── q_learning.json         线性 Q-Learning 权重
├── old_alphazero.pt        改进前的 AlphaZero 检查点（3 通道输入，留档）
├── old_alphazero_meta.json 改进前的累计训练局数（17700 局）
└── README.md               本文件
```

各算法的实现要点、实验结果与当前状态见 [../README.md](../README.md)。
