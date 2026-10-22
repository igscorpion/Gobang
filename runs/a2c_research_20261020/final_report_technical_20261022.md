# A2C 改进研究最终交付报告

交付时间：2026-10-21 23:58（北京时间）。

**本次交付范围为用户明确调整后的31项／6,200局。原冻结计划32项／6,400局保持原样；AlphaZero400禁手参考按用户要求暂缓，未完成的对局不计入成绩，不宣称原32项全部完成。**

新方案在无禁手和禁手白方对Greedy的表现明显改善；禁手黑方仍存在较大的训练种子差异，其平均胜率未超过原增强参考。结果只代表本项目9×9测试协议，不能换算成人类段位或完整赛事强度。

## 1. 最终范围与独立证据

- 原策略冻结清单SHA256：`4d0d0acc5a49fa604d1d773b5730ea352c08b77f935dfda592d83017f8721c7a`，仍由开发集选择。
- 用户范围文件：[delivery_scope_31_jobs.json](final_evaluation/delivery_scope_31_jobs.json)，SHA256 `e477a51c3a9a3f70e87ef7d7f6a82991ca1a53ae3631b700e220805ad655fe41`。
- 完整独立审计：[final_audit_31_jobs.json](final_evaluation/final_audit_31_jobs.json)，SHA256 `bdd441be7ded416b1daa8608e17b5d94c7f92636476b65b5adc9b8b3ed45fa05`。
- 覆盖6,200局、164,639步合法重放、81,164个Greedy/Random动作精确复现。候选动作核验合法性，不重新执行神经网络或证明搜索。
- 重试审计：[operational_retry_prefix_audit_31_jobs.json](final_evaluation/operational_retry_prefix_audit_31_jobs.json)，SHA256 `9f86ce6db102d002f06a78873447ef9d1225f1b1ec2221cde49192985c0fa1d5`；3个已完成任务的失败前缀与唯一完整accepted attempt一致，partial不混计。
- CPU评估，Greedy／Random分色；每个训练seed每格100局。free黑/白种子8,100,000/8,200,000，renju黑/白8,300,000/8,400,000。和棋仅在score中记0.5，不计胜。
- 三个residual训练seed为11/22/33；固定共同guarded包装，各自开发选择checkpoint为30k/15k/25k。展示模型在首局测试前已固定为seed11@30k guarded，未根据最终成绩换seed或wrapper。
- 相同对手种子构成配对；每seed每色给Wilson95，跨训练seed给均值与样本标准差（ddof=1），不把300局当独立样本合并置信区间。

## 2. 主方案对Greedy：三次真实训练

| 规则/执子 | seed11胜/和/负 | seed22胜/和/负 | seed33胜/和/负 | 胜率均值 ± 样本SD | 和率均值 | score均值 ± 样本SD |
|---|---|---|---|---:|---:|---:|
| 无禁手/黑 | 97/1/2 | 91/9/0 | 88/9/3 | 92.00% ± 4.58个百分点 | 6.33% | 0.95167 ± 0.02517 |
| 无禁手/白 | 78/17/5 | 74/22/4 | 84/11/5 | 78.67% ± 5.03个百分点 | 16.67% | 0.87000 ± 0.02291 |
| 禁手/黑 | 52/37/11 | 42/46/12 | 65/25/10 | 53.00% ± 11.53个百分点 | 36.00% | 0.71000 ± 0.06265 |
| 禁手/白 | 84/13/3 | 74/19/7 | 89/5/6 | 82.33% ± 7.64个百分点 | 12.33% | 0.88500 ± 0.04359 |

主方案无禁手黑白等权平均胜率85.33%，禁手67.67%。三seed、四格等权整体胜率76.50%；这一综合值不可替代分色结果。禁手黑方53.00%及较大种子波动是本次明确局限。

### 各训练seed每色Wilson95

| seed/规则/执子 | 胜率 | 胜率Wilson95 | score |
|---|---:|---|---:|
| 11/无禁手/黑 | 97.00% | 91.55%–98.97% | 0.9750 |
| 11/无禁手/白 | 78.00% | 68.93%–85.00% | 0.8650 |
| 11/禁手/黑 | 52.00% | 42.32%–61.54% | 0.7050 |
| 11/禁手/白 | 84.00% | 75.58%–89.90% | 0.9050 |
| 22/无禁手/黑 | 91.00% | 83.77%–95.19% | 0.9550 |
| 22/无禁手/白 | 74.00% | 64.63%–81.60% | 0.8500 |
| 22/禁手/黑 | 42.00% | 32.80%–51.79% | 0.6500 |
| 22/禁手/白 | 74.00% | 64.63%–81.60% | 0.8350 |
| 33/无禁手/黑 | 88.00% | 80.19%–93.00% | 0.9250 |
| 33/无禁手/白 | 84.00% | 75.58%–89.90% | 0.8950 |
| 33/禁手/黑 | 65.00% | 55.25%–73.64% | 0.7750 |
| 33/禁手/白 | 89.00% | 81.37%–93.75% | 0.9150 |

## 3. 相同checkpoint的纯策略消融与历史策略参照

| 规则/执子 | 新纯策略三seed平均胜率 | 新guarded三seed平均胜率 | 原增强同批新测试胜率 |
|---|---:|---:|---:|
| 无禁手/黑 | 87.67% | 92.00% | 53.00% |
| 无禁手/白 | 75.33% | 78.67% | 40.00% |
| 禁手/黑 | 44.00% | 53.00% | 58.00% |
| 禁手/白 | 79.67% | 82.33% | 59.00% |

这里的纯策略与guarded使用各seed同一主checkpoint，隔离推理包装变化；不同包装不算新增独立训练。开发阶段各自best（例如seed33 policy20k与guarded25k）不与该同checkpoint消融混淆。
纯网络无禁手黑87.67%／白75.33%，说明收益不限于战术接管；但输入同时包含规则派生特征，整体相对原模型的改善不能全归因A2C更新公式。guarded仍执行TacticalEngine，只对可证/急防等情况接管，普通回退面对完整合法集；其VCF/proof收益应与网络学习分开。

原400局增强历史基线继续保存在`runs/a2c_tactical_greedy_20261020`：free黑59/25/16、白54/20/26；renju黑54/27/19、白55/34/11。这批旧局已用于败局开发，不能作为新泛化证据。上表旧增强使用本次新种子重测，故数值与旧400局不同。

## 4. Random与contextual reference

主guarded三seed×两规则×两色共1,200局Random全部获胜；matched policy也共1,200局全部获胜。单格100/100的Wilson95约96.30%–100%，有限样本不证明任意随机种子必胜。

| 参考/规则 | 黑胜/和/负 | 黑score | 白胜/和/负 | 白score |
|---|---|---:|---|---:|
| 原增强A2C/无禁手 | 53/32/15 | 0.6900 | 40/31/29 | 0.5550 |
| 原增强A2C/禁手 | 58/27/15 | 0.7150 | 59/24/17 | 0.7100 |
| flat10单seed参考/无禁手 | 95/5/0 | 0.9750 | 82/9/9 | 0.8650 |
| flat10单seed参考/禁手 | 70/22/8 | 0.8100 | 86/9/5 | 0.9050 |
| AZ400/无禁手 | 94/3/3 | 0.9550 | 70/10/20 | 0.7500 |
| AZ400/禁手 | 用户暂缓 | — | 未完成，不作比较 | — |
| Minimax d2/c8/无禁手 | 7/65/28 | 0.3950 | 6/44/50 | 0.2800 |
| Minimax d2/c8/禁手 | 6/67/27 | 0.3950 | 9/48/43 | 0.3300 |

这些是各自对Greedy的contextual参照，不是A2C与AZ直接对弈，训练和推理预算也不相同。AZ400为每步MCTS400次，日志search_nodes=0属于计数未接入，不能解释为无搜索。Minimax固定depth2/candidates8，未为迎合结果临时加深。

## 5. 网络容量、训练与计算成本

| 网络 | 主干/头 | 参数 | 估算单次网络MAC |
|---|---|---:|---:|
| 原flat4 A2C | 3层Conv32/64/64 + FC256双头 | 1,405,042 | 5.92M |
| flat10 A2C | 相同flat骨干，10通道 | 1,406,770 | 6.06M |
| residual10 A2C | stem64 + 4个双卷积残差块（9层3×3主干），空间policy、GAP value | 305,538 | 24.36M |
| AZ64/64/128 | 3层卷积双头，MCTS400 | 132,055 | 单次前向不等于整步搜索成本 |

residual总参数比原flat4少约78.25%，但卷积主干301,248参数高于原flat主干56,608；原flat大部分参数在全连接层。仅网络形状估算MAC反而约4.12倍。估算忽略激活、池化、残差加、特征与战术，不能由参数减少推导训练或推理更便宜。

基础4平面为己方、敌方、上一步和当前行动方；另6平面为己/敌合法点、立即胜点和局部威胁强度。后者是规则/启发式先验，不是额外独立游戏或未来标签。

| 训练lineage | CE实际epoch/最佳载入 | 累计A2C局数 | updates | learner transitions | 日志trainer wall秒 |
|---|---:|---:|---:|---:|---:|
| residual_tactical10_seed11 | 20/16 | 30,000 | 15,247 | 580,853 | 2163.73 |
| residual_tactical10_seed22 | 20/17 | 30,000 | 14,883 | 568,379 | 2251.11 |
| residual_tactical10_seed33 | 20/16 | 30,000 | 14,767 | 559,937 | 2025.19 |
| flat_tactical10_seed22 | 20/20 | 30,000 | 13,444 | 510,844 | 1812.33 |

三residual每seed实际训练CE20 + 累计30k A2C+CE(.1)，Greedy/Random/selfplay按episode比例.6/.2/.2，8env×5step，Adam3e-4、gamma.99、value.5、entropy.01、gradclip5、TF32关闭。seed11/33各自随机初始化；seed22和flat22复制完整原10k Adam/RNG后扩到30k，复制新增0次更新，原10k只计一次。共享同一份AZ专家数据，三训练种子并非三个独立教师数据集。
四lineage累计120,000局、58,341 updates、2,220,013学习转移、8,252.36秒日志trainer wall；confirmation内实际新增100,000局。每条CE实跑20epoch、1,000updates、253,460次状态呈现；CE载入验证best早于20不减少已付训练成本。日志wall含内部验证与IO，部分预处理不在计时内；CE净时间未记录，launcher时间窗不能伪装GPU净时。
原A2C三seed各30k、合计90k，51,275updates／1,981,867学习转移／2,229.20秒活动wall；旧战术增强直接复用seed22@19k，额外训练0。原时间字段归因与新trainer不同、且跨日/并发负载不同，不作严格速度倍数结论。19k来自历史sidecar，旧payload无episode元数据仍写未知，不能解释为没训练。
共同专家数据仅生成一次：576局（512/64），14,526原始标签，保留训练12,673/验证1,745状态，跨集D4重合0。AZ400标签调用的配置模拟预算5,810,400次，不是实测网络前向数；树复用、终局会改变展开工作。生成精确wall/CPU时间未记录；既有教师训练、所有开发试验与评估成本另计，不能把上述四训练wall称为整项研究总成本。原始成本与SHA见[costs/index.json](costs/index.json)。

### 实际对局时延（展示模型）

| 规则/执子 | 均值ms | 中位数ms | P95 ms | 最大ms |
|---|---:|---:|---:|---:|
| 无禁手/黑 | 35.78 | 50.18 | 73.93 | 92.16 |
| 无禁手/白 | 34.96 | 36.15 | 83.97 | 109.51 |
| 禁手/黑 | 115.07 | 63.40 | 345.70 | 618.31 |
| 禁手/白 | 378.33 | 139.07 | 1452.20 | 3318.37 |

这些来自不同到达局面与并发对局，不是严格同局面速度实验。guarded仍有forcing depth5/max_nodes1500/candidates12/cache4096的证明/战术成本；节点预算不是硬毫秒上限。旧增强禁手白方存在约724.9秒的计时长尾，具体原因未确认，可能含进程挂起/系统调度；原数值保留，不据受长尾影响的均值宣称算法速度优劣。

## 6. 开发经过、负结果与局限

- 诊断与阶段2搜索已完成；固定普通候选直接替换、深度/候选扩展没有普遍改善。阶段3/4共3,440开发局审计，guarded筛选960局、训练后搜索320局均审计保留。
- 四种full-tactics-root搜索最高mean score .78125，未超过同cp guarded .89375；这是这些具体根候选/预算的负结果，不能外推所有搜索。旧value混合也未证明可靠，未盲信价值头。
- 原强课程only没有改善内部纯网络；专家示范、特征、结构及推理包装组合有效，不能称纯粹A2C公式或单纯扩容的胜利。原4×10k表示对照与胜者额外扩训分开。
- confirmation56项开发评估4,480记录，含6项精确复用480局；其余4,000为新完整case，复用不算额外独立训练。所有开发选优均先于最终冻结。
- renju为freestyle权重的规则感知迁移，黑方禁手改变合法空间。黑方平均胜率未提升，且三seed从42%到65%，体现迁移不稳定。知识迁移依赖教师、局面覆盖与规则特征，未专门训练完整禁手课程。
- 9×9及当前开局协议不代表完整比赛开局制度；Greedy/Random不是所有对手。共同专家来源、教师warm-start与不同推理预算限制因果归因。
- 用户暂停的是耗时的AZ禁手contextual参考；原冻结/partial保留，未来恢复需新明确授权，不能把暂停项补写为通过。本次交付不继续用已打开测试调参。

## 7. 网页与保留项

新独立模型`a2c_residual10_guarded`已接入主项目registry及隔离网页，固定seed11@30k，SHA `dea2fee0d128485ed20461f7ce5e793f4a9a193870af068ee4a03c9940c5b898`。原`a2c4_tactical`、旧模型目录、全部历史权重和报告保留。服务地址：[本地网页](http://127.0.0.1:5193)，API8020。独立部署不使用reload，不停止原服务。

15项注册表测试、24项部署门控测试通过，前端构建成功；实际浏览器验证模型列表、新模型落子、旧模型切换与禁手规则切换。原沙箱前端启动因esbuild读取父目录受限失败，保留原日志后在同一独立目录恢复前端；后端未重启。部署SHA、端口和进程启动记录见[部署证据](final_evaluation/web_integration/deployment_evidence.json)。

## 附录：全部31任务分色成绩

| 任务/执子 | W/D/L | 胜率 | Wilson95 | score |
|---|---|---:|---|---:|
| residual_seed11_guarded_freestyle_greedy/黑 | 97/1/2 | 97.00% | 91.55%–98.97% | 0.9750 |
| residual_seed11_guarded_freestyle_greedy/白 | 78/17/5 | 78.00% | 68.93%–85.00% | 0.8650 |
| residual_seed11_guarded_freestyle_random/黑 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed11_guarded_freestyle_random/白 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed11_guarded_renju_greedy/黑 | 52/37/11 | 52.00% | 42.32%–61.54% | 0.7050 |
| residual_seed11_guarded_renju_greedy/白 | 84/13/3 | 84.00% | 75.58%–89.90% | 0.9050 |
| residual_seed11_guarded_renju_random/黑 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed11_guarded_renju_random/白 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed11_policy_freestyle_greedy/黑 | 93/2/5 | 93.00% | 86.25%–96.57% | 0.9400 |
| residual_seed11_policy_freestyle_greedy/白 | 76/18/6 | 76.00% | 66.77%–83.31% | 0.8500 |
| residual_seed11_policy_freestyle_random/黑 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed11_policy_freestyle_random/白 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed11_policy_renju_greedy/黑 | 44/40/16 | 44.00% | 34.67%–53.77% | 0.6400 |
| residual_seed11_policy_renju_greedy/白 | 83/14/3 | 83.00% | 74.45%–89.11% | 0.9000 |
| residual_seed11_policy_renju_random/黑 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed11_policy_renju_random/白 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed22_guarded_freestyle_greedy/黑 | 91/9/0 | 91.00% | 83.77%–95.19% | 0.9550 |
| residual_seed22_guarded_freestyle_greedy/白 | 74/22/4 | 74.00% | 64.63%–81.60% | 0.8500 |
| residual_seed22_guarded_freestyle_random/黑 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed22_guarded_freestyle_random/白 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed22_guarded_renju_greedy/黑 | 42/46/12 | 42.00% | 32.80%–51.79% | 0.6500 |
| residual_seed22_guarded_renju_greedy/白 | 74/19/7 | 74.00% | 64.63%–81.60% | 0.8350 |
| residual_seed22_guarded_renju_random/黑 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed22_guarded_renju_random/白 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed22_policy_freestyle_greedy/黑 | 87/11/2 | 87.00% | 79.02%–92.24% | 0.9250 |
| residual_seed22_policy_freestyle_greedy/白 | 69/26/5 | 69.00% | 59.37%–77.22% | 0.8200 |
| residual_seed22_policy_freestyle_random/黑 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed22_policy_freestyle_random/白 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed22_policy_renju_greedy/黑 | 32/54/14 | 32.00% | 23.67%–41.66% | 0.5900 |
| residual_seed22_policy_renju_greedy/白 | 70/21/9 | 70.00% | 60.41%–78.11% | 0.8050 |
| residual_seed22_policy_renju_random/黑 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed22_policy_renju_random/白 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed33_guarded_freestyle_greedy/黑 | 88/9/3 | 88.00% | 80.19%–93.00% | 0.9250 |
| residual_seed33_guarded_freestyle_greedy/白 | 84/11/5 | 84.00% | 75.58%–89.90% | 0.8950 |
| residual_seed33_guarded_freestyle_random/黑 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed33_guarded_freestyle_random/白 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed33_guarded_renju_greedy/黑 | 65/25/10 | 65.00% | 55.25%–73.64% | 0.7750 |
| residual_seed33_guarded_renju_greedy/白 | 89/5/6 | 89.00% | 81.37%–93.75% | 0.9150 |
| residual_seed33_guarded_renju_random/黑 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed33_guarded_renju_random/白 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed33_policy_freestyle_greedy/黑 | 83/10/7 | 83.00% | 74.45%–89.11% | 0.8800 |
| residual_seed33_policy_freestyle_greedy/白 | 81/11/8 | 81.00% | 72.22%–87.49% | 0.8650 |
| residual_seed33_policy_freestyle_random/黑 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed33_policy_freestyle_random/白 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed33_policy_renju_greedy/黑 | 56/33/11 | 56.00% | 46.23%–65.33% | 0.7250 |
| residual_seed33_policy_renju_greedy/白 | 86/5/9 | 86.00% | 77.86%–91.47% | 0.8850 |
| residual_seed33_policy_renju_random/黑 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| residual_seed33_policy_renju_random/白 | 100/0/0 | 100.00% | 96.30%–100.00% | 1.0000 |
| flat_seed22_guarded_freestyle_greedy/黑 | 95/5/0 | 95.00% | 88.82%–97.85% | 0.9750 |
| flat_seed22_guarded_freestyle_greedy/白 | 82/9/9 | 82.00% | 73.33%–88.30% | 0.8650 |
| flat_seed22_guarded_renju_greedy/黑 | 70/22/8 | 70.00% | 60.41%–78.11% | 0.8100 |
| flat_seed22_guarded_renju_greedy/白 | 86/9/5 | 86.00% | 77.86%–91.47% | 0.9050 |
| original_a2c4_tactical_freestyle_greedy/黑 | 53/32/15 | 53.00% | 43.29%–62.49% | 0.6900 |
| original_a2c4_tactical_freestyle_greedy/白 | 40/31/29 | 40.00% | 30.94%–49.80% | 0.5550 |
| original_a2c4_tactical_renju_greedy/黑 | 58/27/15 | 58.00% | 48.21%–67.20% | 0.7150 |
| original_a2c4_tactical_renju_greedy/白 | 59/24/17 | 59.00% | 49.20%–68.13% | 0.7100 |
| alphazero_width128_alphazero_freestyle_greedy/黑 | 94/3/3 | 94.00% | 87.52%–97.22% | 0.9550 |
| alphazero_width128_alphazero_freestyle_greedy/白 | 70/10/20 | 70.00% | 60.41%–78.11% | 0.7500 |
| minimax_depth2_minimax_freestyle_greedy/黑 | 7/65/28 | 7.00% | 3.43%–13.75% | 0.3950 |
| minimax_depth2_minimax_freestyle_greedy/白 | 6/44/50 | 6.00% | 2.78%–12.48% | 0.2800 |
| minimax_depth2_minimax_renju_greedy/黑 | 6/67/27 | 6.00% | 2.78%–12.48% | 0.3950 |
| minimax_depth2_minimax_renju_greedy/白 | 9/48/43 | 9.00% | 4.81%–16.23% | 0.3300 |
