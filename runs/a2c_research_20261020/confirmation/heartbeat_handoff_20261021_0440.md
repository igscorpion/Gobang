# A2C confirmation handoff


## 2026-10-21 04:40 北京时间：正式确认训练与CPU开发选择器核验

本条覆盖前文“selector仍在实现”的旧状态。GPU supervisor PID63460/birth134370017491520920、训练 runtime66392/birth134370017548503583均通过实际OS身份检查；confirmation_suite.lock和train/run.lock持有。residual/tactical10 seed22已从独立复制的10k完整状态推进到17,000/30,000，8,512 updates、325,174 learner transitions；训练器内部pure验证best15k mean score .91875（3m/3.1m），只是开发值，不等于正式共同wrapper选择或最终测试。

CPU selector已正式运行，runtime66296/birth134370022396072322、launcher57288/birth134370022395903205，selector.lock持有。入口：.venv/Scripts/python.exe -u -m rl.evaluation.a2c_confirmation_select --confirmation runs/a2c_research_20261020/confirmation --output runs/a2c_research_20261020/confirmation/selection --reuse-directory runs/a2c_research_20261020/stage5/guarded_policy_study --workers 2。4个recipe（residual11/22/33、flat22）×7候选（0/5k..30k）×pure/guarded=56case，各黑白40局、开发3.1m/3.2m。目前8/56完成：4项同checkpointSHA+全部评估身份完全匹配的明确复用、4项新正式开发评估。独立8局/色probe在confirmation_selector_probe_20261021，不混正式40局。等待下个20k候选时可能没有eval子进程，属于正常等待，不能据此重复启动。

selector补齐完整ModelSpec、rules、共同datasetSHA、训练sourceSHA、CSV合法outcome/分色局数/WDL/score/Wilson长度与数值校验后正式冻结，文件 rl/evaluation/a2c_confirmation_select.py SHA9c4a5ee7bd8bc6c8b1871f7468d5a8613cb6530c9473e2adbc37b2967627dc9e；selector与训练queue所有冻结source均100%匹配。新增selector8tests、训练queue7tests此前通过；本次不重复无变化测试。共同wrapper由三个residual真实训练种子上的开发表现确定，不能混wrapper或把复用当训练重复。

搜索controller31588/birth134370018145486820仍真实存活、study.lock持有。depth2_value0已完成80局且退出释放锁；depth2_value05刚完成黑40进入白；depth3_value0已黑40白9；depth3_value05黑12。剩余三个writer的PID/birth与OS锁都正常。完整本次OS核验：heartbeat_checks/check_20261020T204018Z.json。此为瞬时历史观察，下一turn仍须实际复核。不要中途修改冻结搜索/训练/评估源码或降低预算。

GPU恢复（仅当supervisor及相关writer实际停止、OS锁全部释放后隐藏启动并保留新日志）：.venv/Scripts/python.exe -u -m rl.training.a2c_confirmation_suite --device cuda --research-root runs/a2c_research_20261020 --output runs/a2c_research_20261020/confirmation。supervisor自身不带--resume，它会对完整子训练checkpoint安排恢复。CPU selector同上入口，不带--resume；先检查自身和活跃eval全部PID/birth/锁。launcher有完整原command，优先复用。隐藏进程可能仍在Codex OS job中，不能声称应用/机器关闭后继续。

原phase4 10k、旧模型、旧报告、原local_suite与网页服务均未动；任何最终8m测试都未打开。自动跟进a2c已保持原30分钟、ACTIVE及同thread，更新为以上实际阶段。后续仍须完成三训练seed确认及全部开发选择、搜索对照，冻结完整策略后再开Greedy/Random free/renju分色新测试，独立重放统计，更新最终报告和独立新registry名称，全部完成才删除自动跟进。
