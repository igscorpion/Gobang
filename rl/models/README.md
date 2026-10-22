# 上游模型与本地训练产物

此目录保留源码包原有权重，不覆盖它们：alphazero.pt 为 64/64/64 三层网络，dqn.pt 为三通道 DQN。原始元信息和 SHA-256 见本地来源/变更清单；上游模型解释存档在 docs/history/upstream-models-README.md。

新训练产物存放于 runs/production/<实验名>/，包括 current.pt、best.pt、final.pt 与 latest.checkpoint。网页通过 runs/production/models.json 加载验证选出的新权重，在对应实验完成前不会把随机初始网络当作新模型展示。
