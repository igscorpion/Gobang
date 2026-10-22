# A2C 阶段3/4开发结果

更新时间：2026-10-21T04:23:00.950648+08:00。以下均为开发验证，不是新最终测试。

正式专家数据已完成512局训练+64局验证，AlphaZero每步400次MCTS；全局按棋盘/先后手/D4去重后训练12673状态、验证1745状态。数据SHA见JSON。

| 组别 | 选择候选 | 黑胜/和/负 | 白胜/和/负 | 平均score | 平均胜率 |
|---|---|---|---|---:|---:|
| strong_only_seed22 | episode_020000 | 26/8/6 | 21/8/11 | 0.68750 | 58.75% |
| warm_imitation_seed22 | best_expert_ce | 23/13/4 | 22/13/5 | 0.72500 | 56.25% |
| demo_a2c_seed22 | episode_000000 | 23/13/4 | 22/13/5 | 0.72500 | 56.25% |
| demo_regularized_seed22 | episode_005000 | 33/3/4 | 15/18/7 | 0.73125 | 60.00% |
| flat_base4_seed22_pretrain | best_expert_ce | 25/9/6 | 12/21/7 | 0.65000 | 46.25% |
| flat_base4_seed22 | episode_010000 | 27/8/5 | 13/18/9 | 0.66250 | 50.00% |
| flat_tactical10_seed22_pretrain | best_expert_ce | 34/6/0 | 25/12/3 | 0.85000 | 73.75% |
| flat_tactical10_seed22 | episode_000000 | 34/6/0 | 25/12/3 | 0.85000 | 73.75% |
| residual_base4_seed22_pretrain | best_expert_ce | 34/5/1 | 26/8/6 | 0.83125 | 75.00% |
| residual_base4_seed22 | episode_000000 | 34/5/1 | 26/8/6 | 0.83125 | 75.00% |
| residual_tactical10_seed22_pretrain | best_expert_ce | 31/7/2 | 16/19/5 | 0.75000 | 58.75% |
| residual_tactical10_seed22 | episode_010000 | 37/3/0 | 23/10/7 | 0.83125 | 75.00% |

每个候选每色40局，均按Greedy开发平均score选模型；纯网络内部best与完整战术包装best不同。各组共享专家数据，部分为旧模型暖启动，架构2×2为随机初始化，不能混为严格相同训练来源或等计算量。

最佳完整战术包装暂为flat/tactical10的监督预训练权重，黑85%、白62.5%；residual/tactical10的纯网络在内部验证5000局处黑92.5%、白62.5%，将用统一开发种子复查，并检验只保留可证明规则的包装能否减少启发式候选过筛。

当前12阶段队列已自然退出，原stage4的10000局对照目录只读保留。后续30000局确认使用新的confirmation目录，不覆盖公平对照。

[完整候选记录](selection_candidates.json) · [机器可读汇总](development_stage3_4.json)

独立完整审计已通过：43组3440局共139030合法步、69291个Greedy动作精确复现；576专家局14526合法标签，训练/验证跨集D4重合为0。[完整审计](audit_development_completed.json)。
