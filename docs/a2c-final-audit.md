# A2C 最终评估独立审计工具

`audit_final_a2c_research.py` 只审计显式指定的评估目录，不搜索实验目录、不运行候选策略、战术搜索、训练或测试。工具目前只用非 8m 开发局及合成数据自检，没有打开正式最终测试结果。

## CLI

推荐通过单独的输入 manifest 指定已完成任务：

```powershell
.\.venv\Scripts\python.exe -u tools/audit_final_a2c_research.py --project-root <PROJECT_ROOT> --manifest <AUDIT_INPUTS.json> --output <NEW_AUDIT.json>
```

也可重复 `--directory <EXACT_EVALUATION_DIR>`。默认只接受 `split=test`、每色至少 100 局的完整冻结测试。`--allow-development` 仅用于明确的开发集自检，不改变最终测试的冻结校验。`--random-self-test` 附加两个非保留种子的微型合成 Random 回放与合法动作篡改拒绝检查。

`--output` 必须是尚不存在的文件，并位于全部输入评估目录之外。当前任务的脚本、自检及输出均放在 visualization 目录，没有修改项目。

## 单独的审计输入 manifest

```json
{
  "format": "a2c_independent_audit_inputs_v1",
  "evaluations": [
    {
      "directory": "<completed job directory>",
      "checkpoint_sha256": "<64 character checkpoint SHA256>",
      "training_seed": 11,
      "training_run": "<actual training directory>",
      "training_protocol_sha256": "<protocol.json SHA256>",
      "recipe": "residual_tactical10_ce20_a2c_ce01",
      "role": "candidate",
      "frozen_selection": "<frozen experiment selection.json>"
    }
  ]
}
```

只有 `directory` 是通用必填字段，其他字段提供额外断言；最终测试会从冻结 `models` 取得并核验配方、训练种子及训练来源。输入 manifest 的相对路径相对于 manifest 本身。不能把重复评估同一权重声明为多个训练种子。

原增强 A2C、flat 单种子参考和 AlphaZero 使用 `role="contextual_reference"` 与各自不同的 `recipe`。原 A2C/AlphaZero 的 `config.json` 可作为旧实验的训练种子来源，工具会准确记录缺少现代训练 protocol 的限制。Minimax 必须显式标为 contextual reference、提供独立 recipe；它不需要 checkpoint，也不会被编造训练种子。

## 最终实验冻结 manifest 的真实约束

冻结文件由研究主流程在开发全部结束后创建。本工具没有创建正式冻结文件。

必须有 `status="frozen"`、`selected_using`、`models`、`evaluations`。`models` 用 **checkpoint 文件 SHA256** 作键，每个值至少包括：

```json
{
  "role": "candidate",
  "recipe": "residual_tactical10_ce20_a2c_ce01",
  "training_seed": 11,
  "training_run": "<project-relative or absolute training directory>",
  "training_protocol_sha256": "<training protocol SHA256>",
  "model_spec": {"architecture": "residual", "features": "tactical10"},
  "selected_episode": 5000
}
```

这里 `model_spec` 是可选字段；若提供，必须是检查点中的**完整** spec，不能只提供上面展示的两个字段。`selected_episode`/`selected_epoch` 可选，研究检查点存在相应元数据时会精确核对。旧参考使用 `config.json` 的 SHA 填写 `training_protocol_sha256`。Minimax 没有 checkpoint SHA，不进入 `models`；其 role/recipe 由审计输入 manifest 明确提供。

`evaluations` 中每项只允许原评估器的精确 identity 键值；不要把 recipe、training_seed 或 output 添加到 identity 字典中。键为：

```text
network_kind, mode, device, rules, opponent, games,
seed_black, seed_white, depth, nodes, search_candidates,
tactical_candidates, value_weight, sims,
checkpoint_sha256, source_sha256
```

recipe/seed/job 上下文放在 SHA 索引的 `models` 和单独审计输入 manifest 中。不同 wrapper 可以引用同一个 checkpoint。冻结文件中的训练路径相对于 `--project-root`。评估器记录的冻结文件 SHA、identity 和 selected_using 都必须一致。

最终测试只接受 freestyle 黑/白 8,100,000 / 8,200,000，renju 黑/白 8,300,000 / 8,400,000；非 test 若与 8,000,000–8,499,999 相交，在打开任何 games/moves/summary 前拒绝。

## 核验与统计口径

- 通过只读文件句柄取得实际 OS lock；eval.lock 持有、状态未完成、文件/来源变化均拒绝。最终候选训练也必须完成且训练锁空；没有训练 lock 制度的旧 contextual reference 会明确保留限制。
- 全部游戏逐手核对规则、落子合法性、轮次、终局、局长；每局重建 Greedy/Random 的独立 RNG，精确核对全部对手动作。候选策略落子只核验合法性，不重新运行神经网络或搜索。
- 核对逐局和汇总 W/D/L、胜率、score、Wilson95（原项目 z=1.96）、棋长、延迟、原因计数与搜索节点。
- 冻结完整 identity、源码和权重 SHA；研究候选还核对完整 companion checkpoint 的 net 与训练 seed。不会以新规则静默重放旧源。
- 训练 seed 取实际训练 protocol/config，不能只由目录名或 manifest 声明。同组重复 seed、跨 seed 同 state_dict 拒绝伪独立汇总。
- 分 wrapper、规则、对手、结构、训练配方、设备和搜索预算汇总。每训练 seed、每色保留 W/D/L 和 Wilson；跨 seed 只报均值及样本标准差（ddof=1），n=1 标准差为 null。
- 相同游戏种子跨模型、wrapper 或训练 seed 是配对设计，不给合并 300 局的独立样本置信区间。和棋只在 score 中计 0.5，绝不计为胜。
- 0 局候选追溯初始 CE 权重并验证 state_dict 完全相同，明确标为 demonstration_pretraining_only，A2C 更新标志为 false。不能据此声称 A2C 训练提升。

## 已完成的非最终测试自检

`test_audit_final_a2c_research.py` 包含 11 项完整性检查：Random RNG 正向/篡改、OS 锁、未完成拒绝、保留种子拒绝、纯字典冻结 identity 夹具、三 seed 样本标准差、重复 seed/相同权重拒绝、同权重多 wrapper、旧标准 A2C/AZ 来源、Minimax 无训练 seed、0 局 CE 来源。

实际开发回放自检使用 residual10_005k pure/guarded，加一个 0 局 tactical 开发任务（共 240 局）；另通过 `audit_final_inputs_development_smoke.json` 检查输入 manifest 分支。原始 400 局基线没有作为新的泛化测试。

限制：最终 test 分支只用无结果的合成 identity 字典做了匹配测试，没有运行或读取真实最终测试。完整 companion checkpoint 含 Python/NumPy RNG，使用可信本地 torch pickle；不能把未知来源文件交给这一分支。


项目交付更新（2026-10-21）：冻结工具已逐字节复制到 `tools/audit_final_a2c_research.py`，测试在 `tools/test_audit_final_a2c_research.py`；11项自检、实际开发回放与独立审查记录保存在 `runs/a2c_research_20261020/audit_tool_validation/`。工具SHA为 `a0a5268cd09f301a03a8786d381fc6fcee900a2d4398e81781e3422b2e6a73ce`。复制未改变源码，没有运行或读取正式最终测试。
