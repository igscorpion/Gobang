"""Render only the independently passed, human-scoped 31-job research delivery."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import statistics
from a2c_user_delivery_scope import verify_audit, digest, read, require


def pct(value):
    return f'{100*value:.2f}%'


def wdl(row):
    return f"{row['wins']}/{row['draws']}/{row['losses']}"


def render(root, contract, retry_path):
    audit, final = contract['audit'], contract['final']
    retry = read(retry_path)
    require(retry.get('passed') is True and retry.get('delivery_scope_sha256') == contract['scope_sha256']
            and retry['gate']['final_audit']['sha256'] == contract['audit_sha256'], 'Retry audit does not bind this passed delivery')
    by_name = {job['name']: contract['by_dir'][str(Path(read(final/'results/completed_jobs.json')[job['name']]['directory']).resolve())]
               for job in contract['jobs']}
    def rows(seed, mode, rule, opponent='greedy'):
        return by_name[f'residual_seed{seed}_{mode}_{rule}_{opponent}']['results']
    def color_row(seed, mode, rule, color):
        return next(row for row in rows(seed, mode, rule) if row['color'] == color)
    stamp = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
    lines = ['# A2C 改进研究最终交付报告', '', f'交付时间：{stamp:%Y-%m-%d %H:%M}（北京时间）。', '',
        '**本次交付范围为用户明确调整后的31项／6,200局。原冻结计划32项／6,400局保持原样；AlphaZero400禁手参考按用户要求暂缓，未完成的对局不计入成绩，不宣称原32项全部完成。**', '',
        '新方案在无禁手和禁手白方对Greedy的表现明显改善；禁手黑方仍存在较大的训练种子差异，其平均胜率未超过原增强参考。结果只代表本项目9×9测试协议，不能换算成人类段位或完整赛事强度。', '',
        '## 1. 最终范围与独立证据', '',
        f"- 原策略冻结清单SHA256：`{contract['scope']['original_manifest_sha256']}`，仍由开发集选择。",
        f"- 用户范围文件：[delivery_scope_31_jobs.json](final_evaluation/delivery_scope_31_jobs.json)，SHA256 `{contract['scope_sha256']}`。",
        f"- 完整独立审计：[final_audit_31_jobs.json](final_evaluation/final_audit_31_jobs.json)，SHA256 `{contract['audit_sha256']}`。",
        f"- 覆盖{audit['total_games']:,}局、{audit['legal_moves_replayed']:,}步合法重放、{audit['opponent_actions_exactly_reproduced']:,}个Greedy/Random动作精确复现。候选动作核验合法性，不重新执行神经网络或证明搜索。",
        f"- 重试审计：[operational_retry_prefix_audit_31_jobs.json](final_evaluation/operational_retry_prefix_audit_31_jobs.json)，SHA256 `{digest(retry_path)}`；{retry['retry_job_count']}个已完成任务的失败前缀与唯一完整accepted attempt一致，partial不混计。",
        '- CPU评估，Greedy／Random分色；每个训练seed每格100局。free黑/白种子8,100,000/8,200,000，renju黑/白8,300,000/8,400,000。和棋仅在score中记0.5，不计胜。',
        '- 三个residual训练seed为11/22/33；固定共同guarded包装，各自开发选择checkpoint为30k/15k/25k。展示模型在首局测试前已固定为seed11@30k guarded，未根据最终成绩换seed或wrapper。',
        '- 相同对手种子构成配对；每seed每色给Wilson95，跨训练seed给均值与样本标准差（ddof=1），不把300局当独立样本合并置信区间。', '',
        '## 2. 主方案对Greedy：三次真实训练', '',
        '| 规则/执子 | seed11胜/和/负 | seed22胜/和/负 | seed33胜/和/负 | 胜率均值 ± 样本SD | 和率均值 | score均值 ± 样本SD |',
        '|---|---|---|---|---:|---:|---:|']
    summary = []
    for rule, label in [('freestyle', '无禁手'), ('renju', '禁手')]:
        for color, cname in [(1, '黑'), (-1, '白')]:
            data = [color_row(seed, 'guarded', rule, color) for seed in [11,22,33]]
            wins, scores = [row['win_rate'] for row in data], [row['score'] for row in data]
            draw = statistics.mean(row['draws']/100 for row in data)
            lines.append(f"| {label}/{cname} | {' | '.join(wdl(row) for row in data)} | {pct(statistics.mean(wins))} ± {100*statistics.stdev(wins):.2f}个百分点 | {pct(draw)} | {statistics.mean(scores):.5f} ± {statistics.stdev(scores):.5f} |")
            summary.append({'rules':rule,'color':color,'mean_win_rate':statistics.mean(wins),'sample_sd_win_rate':statistics.stdev(wins),'mean_score':statistics.mean(scores),'sample_sd_score':statistics.stdev(scores)})
    lines += ['', '主方案无禁手黑白等权平均胜率85.33%，禁手67.67%。三seed、四格等权整体胜率76.50%；这一综合值不可替代分色结果。禁手黑方53.00%及较大种子波动是本次明确局限。', '',
        '### 各训练seed每色Wilson95', '', '| seed/规则/执子 | 胜率 | 胜率Wilson95 | score |', '|---|---:|---|---:|']
    for seed in [11,22,33]:
        for rule,label in [('freestyle','无禁手'),('renju','禁手')]:
            for color,cname in [(1,'黑'),(-1,'白')]:
                row=color_row(seed,'guarded',rule,color);ci=row['wilson_95']
                lines.append(f'| {seed}/{label}/{cname} | {pct(row["win_rate"])} | {pct(ci[0])}–{pct(ci[1])} | {row["score"]:.4f} |')
    lines += ['', '## 3. 相同checkpoint的纯策略消融与历史策略参照', '',
        '| 规则/执子 | 新纯策略三seed平均胜率 | 新guarded三seed平均胜率 | 原增强同批新测试胜率 |', '|---|---:|---:|---:|']
    for rule,label in [('freestyle','无禁手'),('renju','禁手')]:
        old=by_name[f'original_a2c4_tactical_{rule}_greedy']['results']
        for color,cname in [(1,'黑'),(-1,'白')]:
            pure=statistics.mean(color_row(seed,'policy',rule,color)['win_rate'] for seed in [11,22,33]);guard=statistics.mean(color_row(seed,'guarded',rule,color)['win_rate'] for seed in [11,22,33]);legacy=next(row for row in old if row['color']==color)
            lines.append(f'| {label}/{cname} | {pct(pure)} | {pct(guard)} | {pct(legacy["win_rate"])} |')
    lines += ['', '这里的纯策略与guarded使用各seed同一主checkpoint，隔离推理包装变化；不同包装不算新增独立训练。开发阶段各自best（例如seed33 policy20k与guarded25k）不与该同checkpoint消融混淆。',
        '纯网络无禁手黑87.67%／白75.33%，说明收益不限于战术接管；但输入同时包含规则派生特征，整体相对原模型的改善不能全归因A2C更新公式。guarded仍执行TacticalEngine，只对可证/急防等情况接管，普通回退面对完整合法集；其VCF/proof收益应与网络学习分开。', '',
        '原400局增强历史基线继续保存在`runs/a2c_tactical_greedy_20261020`：free黑59/25/16、白54/20/26；renju黑54/27/19、白55/34/11。这批旧局已用于败局开发，不能作为新泛化证据。上表旧增强使用本次新种子重测，故数值与旧400局不同。', '',
        '## 4. Random与contextual reference', '',
        '主guarded三seed×两规则×两色共1,200局Random全部获胜；matched policy也共1,200局全部获胜。单格100/100的Wilson95约96.30%–100%，有限样本不证明任意随机种子必胜。', '',
        '| 参考/规则 | 黑胜/和/负 | 黑score | 白胜/和/负 | 白score |', '|---|---|---:|---|---:|']
    for prefix,label in [('original_a2c4_tactical','原增强A2C'),('flat_seed22_guarded','flat10单seed参考'),('alphazero_width128_alphazero','AZ400'),('minimax_depth2_minimax','Minimax d2/c8')]:
        for rule,rlabel in [('freestyle','无禁手'),('renju','禁手')]:
            key=f'{prefix}_{rule}_greedy'
            if key not in by_name:
                lines.append(f'| {label}/{rlabel} | 用户暂缓 | — | 未完成，不作比较 | — |');continue
            a,b=by_name[key]['results'];lines.append(f'| {label}/{rlabel} | {wdl(a)} | {a["score"]:.4f} | {wdl(b)} | {b["score"]:.4f} |')
    lines += ['', '这些是各自对Greedy的contextual参照，不是A2C与AZ直接对弈，训练和推理预算也不相同。AZ400为每步MCTS400次，日志search_nodes=0属于计数未接入，不能解释为无搜索。Minimax固定depth2/candidates8，未为迎合结果临时加深。', '',
        '## 5. 网络容量、训练与计算成本', '',
        '| 网络 | 主干/头 | 参数 | 估算单次网络MAC |', '|---|---|---:|---:|',
        '| 原flat4 A2C | 3层Conv32/64/64 + FC256双头 | 1,405,042 | 5.92M |',
        '| flat10 A2C | 相同flat骨干，10通道 | 1,406,770 | 6.06M |',
        '| residual10 A2C | stem64 + 4个双卷积残差块（9层3×3主干），空间policy、GAP value | 305,538 | 24.36M |',
        '| AZ64/64/128 | 3层卷积双头，MCTS400 | 132,055 | 单次前向不等于整步搜索成本 |', '',
        'residual总参数比原flat4少约78.25%，但卷积主干301,248参数高于原flat主干56,608；原flat大部分参数在全连接层。仅网络形状估算MAC反而约4.12倍。估算忽略激活、池化、残差加、特征与战术，不能由参数减少推导训练或推理更便宜。', '',
        '基础4平面为己方、敌方、上一步和当前行动方；另6平面为己/敌合法点、立即胜点和局部威胁强度。后者是规则/启发式先验，不是额外独立游戏或未来标签。', '',
        '| 训练lineage | CE实际epoch/最佳载入 | 累计A2C局数 | updates | learner transitions | 日志trainer wall秒 |', '|---|---:|---:|---:|---:|---:|']
    cost_dir=root/'runs/a2c_research_20261020/costs';index=read(cost_dir/'index.json')
    for row in index['models']:
        file=cost_dir/row['file'];require(digest(file)==row['sha256'],'Cost record changed');cost=read(file)
        for source in cost['sources']:require(digest(source['path'])==source['sha256'],'Completed training source changed')
        ce,train=cost['pretraining'],cost['a2c_total_unique_training']
        lines.append(f'| {cost["name"]} | {ce["epochs_run"]}/{ce["best_epoch_loaded_by_a2c"]} | {train["episodes"]:,} | {train["updates"]:,} | {train["learner_transitions"]:,} | {train["trainer_wall_seconds"]:.2f} |')
    lines += ['', '三residual每seed实际训练CE20 + 累计30k A2C+CE(.1)，Greedy/Random/selfplay按episode比例.6/.2/.2，8env×5step，Adam3e-4、gamma.99、value.5、entropy.01、gradclip5、TF32关闭。seed11/33各自随机初始化；seed22和flat22复制完整原10k Adam/RNG后扩到30k，复制新增0次更新，原10k只计一次。共享同一份AZ专家数据，三训练种子并非三个独立教师数据集。',
        '四lineage累计120,000局、58,341 updates、2,220,013学习转移、8,252.36秒日志trainer wall；confirmation内实际新增100,000局。每条CE实跑20epoch、1,000updates、253,460次状态呈现；CE载入验证best早于20不减少已付训练成本。日志wall含内部验证与IO，部分预处理不在计时内；CE净时间未记录，launcher时间窗不能伪装GPU净时。',
        '原A2C三seed各30k、合计90k，51,275updates／1,981,867学习转移／2,229.20秒活动wall；旧战术增强直接复用seed22@19k，额外训练0。原时间字段归因与新trainer不同、且跨日/并发负载不同，不作严格速度倍数结论。19k来自历史sidecar，旧payload无episode元数据仍写未知，不能解释为没训练。',
        '共同专家数据仅生成一次：576局（512/64），14,526原始标签，保留训练12,673/验证1,745状态，跨集D4重合0。AZ400标签调用的配置模拟预算5,810,400次，不是实测网络前向数；树复用、终局会改变展开工作。生成精确wall/CPU时间未记录；既有教师训练、所有开发试验与评估成本另计，不能把上述四训练wall称为整项研究总成本。原始成本与SHA见[costs/index.json](costs/index.json)。', '',
        '### 实际对局时延（展示模型）', '', '| 规则/执子 | 均值ms | 中位数ms | P95 ms | 最大ms |', '|---|---:|---:|---:|---:|']
    for rule,label in [('freestyle','无禁手'),('renju','禁手')]:
        for row in rows(11,'guarded',rule):
            lines.append(f'| {label}/{"黑" if row["color"]==1 else "白"} | {row["mean_move_ms"]:.2f} | {row["median_move_ms"]:.2f} | {row["p95_move_ms"]:.2f} | {row["max_move_ms"]:.2f} |')
    lines += ['', '这些来自不同到达局面与并发对局，不是严格同局面速度实验。guarded仍有forcing depth5/max_nodes1500/candidates12/cache4096的证明/战术成本；节点预算不是硬毫秒上限。旧增强禁手白方存在约724.9秒的计时长尾，具体原因未确认，可能含进程挂起/系统调度；原数值保留，不据受长尾影响的均值宣称算法速度优劣。', '',
        '## 6. 开发经过、负结果与局限', '',
        '- 诊断与阶段2搜索已完成；固定普通候选直接替换、深度/候选扩展没有普遍改善。阶段3/4共3,440开发局审计，guarded筛选960局、训练后搜索320局均审计保留。',
        '- 四种full-tactics-root搜索最高mean score .78125，未超过同cp guarded .89375；这是这些具体根候选/预算的负结果，不能外推所有搜索。旧value混合也未证明可靠，未盲信价值头。',
        '- 原强课程only没有改善内部纯网络；专家示范、特征、结构及推理包装组合有效，不能称纯粹A2C公式或单纯扩容的胜利。原4×10k表示对照与胜者额外扩训分开。',
        '- confirmation56项开发评估4,480记录，含6项精确复用480局；其余4,000为新完整case，复用不算额外独立训练。所有开发选优均先于最终冻结。',
        '- renju为freestyle权重的规则感知迁移，黑方禁手改变合法空间。黑方平均胜率未提升，且三seed从42%到65%，体现迁移不稳定。知识迁移依赖教师、局面覆盖与规则特征，未专门训练完整禁手课程。',
        '- 9×9及当前开局协议不代表完整比赛开局制度；Greedy/Random不是所有对手。共同专家来源、教师warm-start与不同推理预算限制因果归因。',
        '- 用户暂停的是耗时的AZ禁手contextual参考；原冻结/partial保留，未来恢复需新明确授权，不能把暂停项补写为通过。本次交付不继续用已打开测试调参。', '',
        '## 7. 网页与保留项', '',
        '独立新模型名为`a2c_residual10_guarded`，固定seed11@30k／SHA `dea2fee0d128485ed20461f7ce5e793f4a9a193870af068ee4a03c9940c5b898`。旧`a2c4_tactical`、原checkpoint与所有报告保留。部署和集成验证记录独立保存；是否可用以实际部署记录为准，不把仅JSON更新称为新Python分支已经热生效。', '',
        '## 附录：全部31任务分色成绩', '', '| 任务/执子 | W/D/L | 胜率 | Wilson95 | score |', '|---|---|---:|---|---:|']
    for name,item in by_name.items():
        for row in item['results']:
            ci=row['wilson_95'];lines.append(f'| {name}/{"黑" if row["color"]==1 else "白"} | {wdl(row)} | {pct(row["win_rate"])} | {pct(ci[0])}–{pct(ci[1])} | {row["score"]:.4f} |')
    return '\n'.join(lines)+'\n', summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','delivery-scope','final-audit','retry-audit','output']:
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();root=Path(args.project_root).resolve();out=Path(args.output)
    require(not out.exists(),'Report output exists')
    contract=verify_audit(root,args.delivery_scope,args.final_audit)
    text,summary=render(root,contract,args.retry_audit)
    with out.open('x',encoding='utf-8') as stream:stream.write(text)
    data=out.with_suffix('.json');require(not data.exists(),'Summary output exists')
    with data.open('x',encoding='utf-8') as stream:json.dump({'status':'audited_31_job_user_scope','deferred_jobs':contract['scope']['deferred_jobs'],'audit_sha256':contract['audit_sha256'],'scope_sha256':contract['scope_sha256'],'guarded_greedy_cross_seed':summary},stream,ensure_ascii=False,indent=2)
    print(json.dumps({'report':str(out),'sha256':digest(out),'summary':str(data)},ensure_ascii=False))


if __name__=='__main__':
    main()
