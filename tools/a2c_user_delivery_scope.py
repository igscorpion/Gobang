"""Validate the explicit human-requested 31-job delivery, preserving the freeze."""
import hashlib
import json
from pathlib import Path

FROZEN_SHA = '4d0d0acc5a49fa604d1d773b5730ea352c08b77f935dfda592d83017f8721c7a'
AUDITOR_SHA = 'a0a5268cd09f301a03a8786d381fc6fcee900a2d4398e81781e3422b2e6a73ce'
DEFERRED = 'alphazero_width128_alphazero_renju_greedy'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_scope(project_root, scope_path):
    root, path = Path(project_root).resolve(), Path(scope_path).resolve()
    final = root / 'runs/a2c_research_20261020/final_evaluation'
    require(path == final / 'delivery_scope_31_jobs.json', 'Unexpected delivery scope location')
    scope = read(path)
    frozen_path = final / 'final_selection.json'
    require(digest(frozen_path) == FROZEN_SHA, 'Original frozen manifest changed')
    require(scope.get('format') == 'a2c_explicit_user_delivery_scope_v1'
            and scope.get('status') == 'user_authorized_deferred_reference', 'Unapproved scope format')
    require(Path(scope['original_frozen_manifest']).resolve() == frozen_path
            and scope.get('original_manifest_sha256') == FROZEN_SHA, 'Scope names another freeze')
    frozen = read(frozen_path)
    require(frozen.get('status') == 'frozen' and frozen.get('selected_using') == 'development only'
            and len(frozen['jobs']) == len(frozen['evaluations']) == 32
            and frozen.get('total_games') == 6400, 'Original experiment contract differs')
    request_path = Path(scope['user_request']).resolve()
    require(request_path.is_relative_to(final / 'recoveries')
            and digest(request_path) == scope.get('user_request_sha256'), 'User request evidence changed')
    request = read(request_path)
    require(request.get('format') == 'a2c_user_deferred_reference_v1'
            and request.get('user_instruction') == '这最后一项测试暂时先不做，先把其他未完成的完成了'
            and request.get('deferred_job') == DEFERRED
            and request.get('original_manifest_sha256') == FROZEN_SHA, 'Explicit human deferral is missing')
    jobs = [job for job in frozen['jobs'] if job['name'] != DEFERRED]
    require(len(jobs) == 31 and scope.get('included_jobs') == [job['name'] for job in jobs]
            and scope.get('deferred_jobs') == [DEFERRED], 'Scope must omit exactly the user-deferred reference')
    require(scope.get('included_jobs_count') == 31 and scope.get('included_games') == 6200
            and scope.get('original_jobs_count') == 32 and scope.get('original_games') == 6400
            and scope.get('policy_or_checkpoint_selection_changed') is False
            and scope.get('remaining_job_identities_and_budgets_unchanged') is True,
            'Delivery budget or model selection differs')
    for job in jobs:
        identity = frozen['evaluations'][job['identity_index']]
        require(identity['games'] == 100 and identity['device'] == 'cpu', 'Remaining job budget differs')
    return {'scope': scope, 'scope_path': path, 'scope_sha256': digest(path),
            'frozen': frozen, 'frozen_path': frozen_path, 'jobs': jobs,
            'user_request': request, 'user_request_path': request_path, 'final': final}


def verify_audit(project_root, scope_path, audit_path):
    contract = load_scope(project_root, scope_path)
    final, frozen = contract['final'], contract['frozen']
    inputs_path = final / 'audit_inputs_31_jobs.json'
    inputs = read(inputs_path)
    require(inputs.get('format') == 'a2c_independent_audit_inputs_v1'
            and len(inputs.get('evaluations', [])) == 31, 'Delivery inputs are incomplete')
    audit = read(audit_path)
    require(audit.get('format') == 'a2c_independent_final_audit_v1'
            and audit.get('passed') is True and audit.get('errors') == []
            and audit.get('allow_development') is False and audit.get('tool_sha256') == AUDITOR_SHA
            and audit.get('input_manifest_sha256') == digest(inputs_path)
            and audit.get('total_games') == 6200 and len(audit.get('evaluations', [])) == 31,
            'Independent audit is not a passed complete 31-job delivery audit')
    require(Path(audit['project_root']).resolve() == Path(project_root).resolve(), 'Audit root differs')
    completed = read(final / 'results/completed_jobs.json')
    require(set(completed) == {job['name'] for job in contract['jobs']}, 'Completed set differs from human scope')
    by_dir = {str(Path(item['directory']).resolve()): item for item in audit['evaluations']}
    require(len(by_dir) == 31 and set(by_dir) == {str(Path(row['directory']).resolve()) for row in inputs['evaluations']},
            'Audit directories are duplicated or differ from inputs')
    for job in contract['jobs']:
        accepted = completed[job['name']]
        item = by_dir[str(Path(accepted['directory']).resolve())]
        require(item.get('passed') is True and item.get('games') == 200 and item.get('split') == 'test'
                and item.get('identity') == frozen['evaluations'][job['identity_index']]
                and item.get('artifact_sha256') == accepted.get('artifacts')
                and item.get('frozen_selection', {}).get('verified') is True
                and item['frozen_selection'].get('sha256') == FROZEN_SHA, 'Audited frozen job identity differs')
        require({row['color'] for row in item['results']} == {1, -1}
                and all(row['games'] == 100 for row in item['results']), 'Audited color budget differs')
    return {**contract, 'audit': audit, 'audit_path': Path(audit_path).resolve(),
            'audit_sha256': digest(audit_path), 'inputs_path': inputs_path, 'by_dir': by_dir}
