"""Publish the independently audited 31-job delivery without changing frozen policy."""
from pathlib import Path
import datetime, hashlib, json, shutil
P=Path(__file__).resolve().parents[1]
R=P/'runs/a2c_research_20261020';F=R/'final_evaluation'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(F/'final_audit_31_jobs.json')['passed'] is True
assert read(F/'operational_retry_prefix_audit_31_jobs.json')['passed'] is True
backup=F/'documentation_before_delivery';backup.mkdir(exist_ok=True)
for rel in ['README.md','rl/README.md','docs/a2c-research.md','experiments/a2c_research.json']:
    dest=backup/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copy2(P/rel,dest)
refs={name:{'path':str((F/name).relative_to(P)).replace('\\','/'),'sha256':sha(F/name)} for name in ['delivery_scope_31_jobs.json','final_audit_31_jobs.json','operational_retry_prefix_audit_31_jobs.json','web_integration/deployment_evidence.json']}
stats=[{'rules':'freestyle','color':'black','win_mean':.92,'win_sample_sd':.0458257569495584,'original_enhanced_win':.53},
       {'rules':'freestyle','color':'white','win_mean':.7866666666666666,'win_sample_sd':.0503322295684717,'original_enhanced_win':.40},
       {'rules':'renju','color':'black','win_mean':.53,'win_sample_sd':.11532562594670796,'original_enhanced_win':.58},
       {'rules':'renju','color':'white','win_mean':.8233333333333334,'win_sample_sd':.07637626158259736,'original_enhanced_win':.59}]
exp=read(P/'experiments/a2c_research.json');ev=exp['final_evaluation']
ev['status']='complete_audited_user_scoped_31_jobs_az_renju_deferred'
ev['completed_at_utc']=now;ev['delivery_evidence']=refs
ev['completed_jobs']=31;ev['audited_games']=6200
ev['original_frozen_jobs']=32;ev['original_frozen_games']=6400
ev['original_queue_status_is_historical']=True
ev['final_report']='runs/a2c_research_20261020/final_report.md'
ev['guarded_greedy_cross_training_seed']=stats
ev['web']={'registry_name':'a2c_residual10_guarded','training_seed':11,'episode':30000,'selected_using':'development only','url':'http://127.0.0.1:5193','api':'http://127.0.0.1:8020','isolated_snapshot':True,'registry_tests_passed':15,'deployment_gate_tests_passed':24,'frontend_build_passed':True,'browser_validation_passed':True,'old_models_preserved':True,'old_services_stopped':False}
ev['delivery_preparation']['status']='applied_audited_and_deployed_with_explicit_31_job_scope'
ev['delivery_preparation']['registry_integration_tests_executed']=True
ev['delivery_preparation']['real_final_audit_executed']=True
ev['resume_deferred_only_on_explicit_user_request']=True
write(P/'experiments/a2c_research.json',exp)
status=read(F/'delivery_status_31_jobs.json');status.update(state='complete_audited_documented_and_web_verified',updated_at_utc=now,evidence=refs,web_url='http://127.0.0.1:5193',report='../final_report.md');write(F/'delivery_status_31_jobs.json',status)
table='''| 规则/执子 | 新方案胜率均值 ± 样本SD | 原增强同次新测试胜率 |
|---|---:|---:|
| 无禁手/黑 | 92.00% ± 4.58个百分点 | 53.00% |
| 无禁手/白 | 78.67% ± 5.03个百分点 | 40.00% |
| 禁手/黑 | 53.00% ± 11.53个百分点 | 58.00% |
| 禁手/白 | 82.33% ± 7.64个百分点 | 59.00% |'''
section='''## A2C 研究已交付（2026-10-22）

按用户调整后的范围完成31项、6,200局最终测试及独立审计：重放164,639步合法落子，精确复现81,164个Greedy/Random动作；3项操作性重试前缀核验通过，未混用partial局。原冻结计划32项/6,400局保留；耗时的AZ400禁手参考按用户要求暂缓，尚未完成，不自动恢复。训练、开发选择与最终测试严格分开。

TABLE

表中为真实训练seed11/22/33的跨种子统计，每seed每色100局，和棋不计胜。无禁手黑白等权平均胜率85.33%，禁手67.67%；禁手黑方未改善。新guarded及同checkpoint纯策略分别对Random完成1,200/1,200胜。逐seed Wilson95、和率、score、纯策略消融及计算成本见最终报告。

网页新增独立名称`a2c_residual10_guarded`，展示模型在测试前已按开发集固定为seed11@30k；原`a2c4_tactical`、权重与历史报告保留。独立页面：[本地对弈](http://127.0.0.1:5193)，API端口8020；源码主项目的注册表也已接入。服务采用独立快照，不重启原网页。15项注册表测试、24项部署门控测试、前端构建及实际页面落子/规则切换验证通过。

新模型305,538参数，原增强网络1,405,042参数；单次纯网络MAC约24.36M对5.92M，参数少并不代表计算更省。guarded还包含VCF证明/战术计算。共享AZ400专家数据、CE暖启动与A2C联合训练，不能把全部收益归因于纯A2C；禁手是无禁手权重的规则感知迁移，9×9 Greedy/Random成绩不能换算人类段位，也没有A2C直接战胜AZ的对弈证据。

'''.replace('TABLE',table)
for rel,prefix in [('README.md',''),('rl/README.md','../')]:
    path=P/rel;text=path.read_text(encoding='utf-8-sig')
    block=section+f'[最终研究报告]({prefix}runs/a2c_research_20261020/final_report.md) · [完整研究记录]({prefix}docs/a2c-research.md)\n\n'
    if '## A2C 研究已交付（2026-10-22）' not in text:
        first,rest=text.split('\n',1);path.write_text(first+'\n\n'+block+rest,encoding='utf-8')
path=P/'docs/a2c-research.md';text=path.read_text(encoding='utf-8-sig')
head=text.find('## 阶段7')
assert head>=0
historical=text[head:];(backup/'stage7_historical.md').write_text(historical,encoding='utf-8')
path.write_text(text[:head]+section+'[最终报告](../runs/a2c_research_20261020/final_report.md) · [31项独立审计](../runs/a2c_research_20261020/final_evaluation/final_audit_31_jobs.json) · [重试核验](../runs/a2c_research_20261020/final_evaluation/operational_retry_prefix_audit_31_jobs.json)\n\n原队列suite_status为停止前历史快照，当前以delivery_status_31_jobs.json及用户范围文件为准。原冻结SHA、8m测试种子、checkpoint和推理预算均未改动。恢复/异常退出原始证据及暂停AZ partial全部保留，暂停项未计入本报告。\n',encoding='utf-8')
path=R/'final_report.md';text=path.read_text(encoding='utf-8');start=text.index('## 7.');end=text.index('## 附录',start)
text=text[:start]+'''## 7. 网页与保留项

新独立模型`a2c_residual10_guarded`已接入主项目registry及隔离网页，固定seed11@30k，SHA `dea2fee0d128485ed20461f7ce5e793f4a9a193870af068ee4a03c9940c5b898`。原`a2c4_tactical`、旧模型目录、全部历史权重和报告保留。服务地址：[本地网页](http://127.0.0.1:5193)，API8020。独立部署不使用reload，不停止原服务。

15项注册表测试、24项部署门控测试通过，前端构建成功；实际浏览器验证模型列表、新模型落子、旧模型切换与禁手规则切换。原沙箱前端启动因esbuild读取父目录受限失败，保留原日志后在同一独立目录恢复前端；后端未重启。部署SHA、端口和进程启动记录见[部署证据](final_evaluation/web_integration/deployment_evidence.json)。

'''+text[end:];path.write_text(text,encoding='utf-8')
write(F/'delivery_artifact_manifest.json',{'updated_at_utc':now,'artifacts':{str(p.relative_to(P)).replace('\\','/'):sha(p) for p in [R/'final_report.md',R/'final_report.json',P/'README.md',P/'rl/README.md',P/'docs/a2c-research.md',P/'experiments/a2c_research.json',P/'rl/agents/registry.py',P/'runs/production/a2c_research_models.json']},'frozen_selection_unchanged':sha(F/'final_selection.json')=='4d0d0acc5a49fa604d1d773b5730ea352c08b77f935dfda592d83017f8721c7a'})
memory=P.parents[1]/'work/A2C_RESEARCH_MEMORY.md'
with memory.open('a',encoding='utf-8') as stream:
    stream.write('\n\n# 当前状态覆盖更新：2026-10-22，用户范围31项交付完成\n\n31/6200最终测试通过原a0a独立审计，164639合法步、81164对手动作精确复现；3项retry前缀审计通过。final_audit_31_jobs.json SHA '+sha(F/'final_audit_31_jobs.json')+'。原32冻结清单及全部策略/权重未改。AZrenju按用户要求deferred，禁止旧heartbeat自动恢复；原suite_status仅历史。\n\n最终报告runs/a2c_research_20261020/final_report.md及experiments/docs/README/rl README已更新。新registry独立名称a2c_residual10_guarded，预选11@30k。15注册表tests/24门控tests/前端build/真实网页落子及规则和旧模型切换通过。隔离快照在visualization/Gobang-a2c-web-verified，网页5193/API8020；无旧服务被停止。部署初始frontend sandbox受限，保留日志后同目录前端单独恢复，backend未重启。\n\nGreedy三seed胜率无禁手黑92±4.58/白78.67±5.03%，禁手黑53±11.53/白82.33±7.64%。禁手黑未超过旧增强58%，不可掩盖。纯策略/guarded分开报告，和棋不计胜。所有实际交付完成后删除a2c自动跟进；暂缓参考无需等待，不恢复。\n')
print(json.dumps({'status':'complete','report_sha256':sha(R/'final_report.md')}))
