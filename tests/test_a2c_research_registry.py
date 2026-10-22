"""Prepared integration tests; execute only after final test/audit deployment gate.

The registry source is imported through a private temporary module. No live
backend is imported, reloaded, contacted or reset. No final results are read.
"""
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import pytest
import torch

from rl.agents.a2c import A2CAgent
from rl.agents.a2c_guarded import GuardedA2CAgent
from rl.agents.a2c_research import ModelSpec, ResearchA2CAgent
from rl.agents.a2c_tactical import TacticalA2CAgent
from rl.agents.tactics import TacticalConfig
from rl.environment.rules import relative_legal_actions

NAME = 'a2c_residual10_guarded'
PROJECT = Path(os.environ.get('GOBANG_PROJECT_ROOT',
    str(Path(__file__).resolve().parents[1])))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv('GOMOKU_RULES', 'freestyle')
    monkeypatch.setenv('GOMOKU_AI_DEVICE', 'cpu')
    prepared = Path(__file__).with_name('registry.prepared.py')
    source = prepared if prepared.is_file() else PROJECT/'rl/agents/registry.py'
    isolated = tmp_path/'isolated_registry.py'
    isolated.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
    spec = importlib.util.spec_from_file_location('_independent_registry_integration', isolated)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'PROJECT_ROOT', tmp_path)
    policy = ResearchA2CAgent(ModelSpec(architecture='residual', features='tactical10',
                                     residual_width=4, residual_blocks=1), device='cpu', seed=11)
    checkpoint = tmp_path/'models/research.pt'
    checkpoint.parent.mkdir()
    policy.save(checkpoint, stage='A2C+imitation', episode=30000)
    old_checkpoint = tmp_path/'models/original.pt'
    A2CAgent(device='cpu', seed=22).save(old_checkpoint)
    write_json(tmp_path/'runs/production/a2c_models.json',
               {'a2c4': {'algorithm':'a2c', 'checkpoint':'models/original.pt'}})
    entry = {'enabled':True, 'algorithm':'a2c_research', 'wrapper':'guarded', 'device':'cpu',
             'checkpoint':'models/research.pt', 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             'model_spec':asdict(policy.spec), 'tactical_config':asdict(TacticalConfig())}
    path = tmp_path/'runs/production/a2c_research_models.json'
    write_json(path, {NAME:entry})
    return module, path, entry, policy, checkpoint, old_checkpoint


def test_old_names_remain_and_new_policy_weights_are_identical(catalog):
    registry, _, _, policy, checkpoint, old_checkpoint = catalog
    before = (checkpoint.read_bytes(), old_checkpoint.read_bytes())
    assert {'a2c4', 'a2c4_tactical', NAME} <= set(registry.available_agents())
    agent = registry.create_agent(NAME)
    assert isinstance(agent, GuardedA2CAgent)
    assert agent.policy.spec == policy.spec and agent.policy.device.type == 'cpu'
    assert not agent.policy.training
    assert isinstance(registry.create_agent('a2c4'), A2CAgent)
    assert isinstance(registry.create_agent('a2c4_tactical'), TacticalA2CAgent)
    for key, value in policy.net.state_dict().items():
        assert torch.equal(value, agent.policy.net.state_dict()[key])
    assert before == (checkpoint.read_bytes(), old_checkpoint.read_bytes())


def test_catalog_refresh_does_not_need_python_reload_after_factory_exists(catalog):
    registry, path, entry, *_ = catalog
    assert NAME in registry.available_agents()
    write_json(path, {NAME:{**entry, 'enabled':False}})
    assert NAME not in registry.available_agents()
    with pytest.raises(ValueError, match='未知 AI'):
        registry.create_agent(NAME)
    write_json(path, {NAME:entry})
    assert NAME in registry.available_agents()


def test_bad_sha_is_rejected_without_rewriting_either_model(catalog):
    registry, path, entry, _, checkpoint, old_checkpoint = catalog
    before = (checkpoint.read_bytes(), old_checkpoint.read_bytes())
    write_json(path, {NAME:{**entry, 'checkpoint_sha256':'0'*64}})
    with pytest.raises(ValueError, match='SHA256'):
        registry.create_agent(NAME)
    assert before == (checkpoint.read_bytes(), old_checkpoint.read_bytes())


@pytest.mark.parametrize('color', (1, -1))
def test_renju_transfer_updates_feature_encoder_and_guard(catalog, monkeypatch, color):
    registry, *_ = catalog
    monkeypatch.setenv('GOMOKU_RULES', 'renju')
    agent = registry.create_agent(NAME)
    assert agent.rules == agent.policy.rules == 'renju'
    absolute = [[0]*9 for _ in range(9)]
    for action in (39, 41, 31, 49): absolute[action//9][action%9] = 1
    for action in ((0,2,6,8) if color == 1 else (0,2,6)):
        absolute[action//9][action%9] = -1
    relative = [[value*color for value in row] for row in absolute]
    state = relative, 8 if color == 1 else 49
    encoded = agent.policy.encode(state)
    legal = relative_legal_actions(relative, 'renju')
    assert bool(encoded[4,4,4]) == (40 in legal) == (color == -1)
    supplied = [action for action in range(81) if not relative[action//9][action%9]]
    assert agent.select_action(state, supplied) in legal


def test_published_cpu_and_wrapper_budget_ignore_legacy_environment_overrides(catalog, monkeypatch):
    registry, *_ = catalog
    monkeypatch.setenv('GOMOKU_AI_DEVICE', 'cuda')
    monkeypatch.setenv('GOMOKU_A2C_TACTIC_DEPTH', '1')
    monkeypatch.setenv('GOMOKU_A2C_TACTIC_NODES', '0')
    agent = registry.create_agent(NAME)
    assert agent.policy.device.type == 'cpu'
    assert agent.tactics.config == TacticalConfig()


@pytest.mark.parametrize('change, message', (
    ({'device':'cuda'}, 'CPU'),
    ({'wrapper':'tactical'}, 'guarded'),
    ({'tactical_config':{}}, 'tactical configuration'),
    ({'model_spec':{'architecture':'flat','features':'base4'}}, 'specification'),
))
def test_published_recipe_mismatch_is_rejected(catalog, change, message):
    registry, path, entry, *_ = catalog
    write_json(path, {NAME:{**entry, **change}})
    with pytest.raises(ValueError, match=message):
        registry.create_agent(NAME)


def test_checkpoint_must_remain_inside_project(catalog):
    registry, path, entry, *_ = catalog
    write_json(path, {NAME:{**entry, 'checkpoint':'../outside.pt'}})
    with pytest.raises(ValueError, match='outside project'):
        registry.create_agent(NAME)


@pytest.mark.parametrize('name', ('a2c4', 'a2c4_tactical', 'greedy'))
def test_independent_catalog_cannot_override_an_old_name(catalog, name):
    registry, path, entry, *_ = catalog
    write_json(path, {name:entry})
    with pytest.raises(ValueError, match='collides'):
        registry.available_agents()


def test_fixed_real_showcase_checkpoint_constructs_without_reading_results(catalog, monkeypatch):
    registry, _, _, *_ = catalog
    metadata = Path(__file__).with_name('a2c_research_models.pending.json')
    if not metadata.is_file():
        metadata = PROJECT/'runs/production/a2c_research_models.json'
    pending = json.loads(metadata.read_text())
    entry = {**pending[NAME], 'enabled':True}
    monkeypatch.setattr(registry, 'PROJECT_ROOT', PROJECT)
    monkeypatch.setattr(registry, '_local_models', lambda:{NAME:entry})
    agent = registry.create_agent(NAME)
    assert agent.policy.spec == ModelSpec(architecture='residual', features='tactical10')
    assert agent.policy.device.type == 'cpu' and agent.tactics.config == TacticalConfig()
    assert sum(value.numel() for value in agent.policy.net.parameters()) == 305538
