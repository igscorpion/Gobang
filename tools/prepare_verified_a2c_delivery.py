"""Apply audited fixed showcase and create the isolated local web snapshot."""
from pathlib import Path
import hashlib, json, shutil, datetime, importlib.util

P=Path(__file__).resolve().parents[1]
V=Path('C:/Users/29024/.codex/visualizations/2026/10/08/01a11b5a-bef7-7131-b4d7-d8c302a6db25')
S=V/'prepared_web_integration_20261021'
D=V/'Gobang-a2c-web-verified'
F=P/'runs/a2c_research_20261020/final_evaluation'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

assert read(F/'final_audit_31_jobs.json')['passed'] is True
assert read(F/'operational_retry_prefix_audit_31_jobs.json')['passed'] is True
registry=P/'rl/agents/registry.py'
backup=F/'web_integration/registry.before.py'
backup.parent.mkdir(parents=True,exist_ok=True)
if not backup.exists():
    assert sha(registry)=='7b3aa514df639afdab24698b187b425b95df4fbc2a572f8cb4784ee5cf8069ec'
    shutil.copy2(registry,backup)
assert sha(S/'registry.prepared.py')=='77126a9a8fd3194cb42f2fb944a1dd1325512d9430d4aa39eba6c3569297eb2a'
shutil.copy2(S/'registry.prepared.py',registry)
catalog=read(S/'a2c_research_models.pending.json')
item=catalog['a2c_residual10_guarded'];item['enabled']=True
item['deployment_gate']='31 user-scoped jobs/6200 games independently audited; AZ renju reference explicitly deferred.'
item['provenance']={name:{'path':str(F/name),'sha256':sha(F/name)} for name in ['final_selection.json','delivery_scope_31_jobs.json','final_audit_31_jobs.json','operational_retry_prefix_audit_31_jobs.json']}
write(P/'runs/production/a2c_research_models.json',catalog)
D.mkdir(exist_ok=True)
for folder in ['rl','web','tools']:
    shutil.copytree(P/folder,D/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','node_modules','dist','.git'))
for rel in ['runs/production/models.json','runs/production/a2c_models.json','runs/production/a2c_research_models.json']:
    dest=D/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(P/rel,dest)
for c in ['models.json','a2c_models.json','a2c_research_models.json']:
    for model in read(P/'runs/production'/c).values():
        rel=model['checkpoint'];dest=D/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(P/rel,dest)
deps=D/'web/frontend/node_modules'
if not deps.exists():
    shutil.copytree(P/'web/frontend/node_modules',deps,ignore=shutil.ignore_patterns('.vite'))
spec=importlib.util.spec_from_file_location('webgate',S/'verify_a2c_web_deployment_v3.py')
import sys
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
gate=read(S/'deployment-gate.v2.template.json')
gate.update(format='a2c_verified_web_deployment_gate_v3_user_scope',tests_complete=True,final_audit_passed=True)
for key,name in [('delivery_scope','delivery_scope_31_jobs.json'),('audit_inputs','audit_inputs_31_jobs.json'),('final_audit','final_audit_31_jobs.json')]:
    gate[key+'_path']=str(F/name);gate[key+'_sha256']=sha(F/name)
gate['old_services_checked_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
gate['deployment_sources_sha256']=m.source_inventory(D)
write(D/'deployment-gate.json',gate)
print(json.dumps({'snapshot':str(D),'registry_sha256':sha(registry),'gate_sha256':sha(D/'deployment-gate.json')}))
