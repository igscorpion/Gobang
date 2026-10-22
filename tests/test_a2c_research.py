"""Research invariants: actual policy likelihoods, data isolation and exact resume."""

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.agents.a2c import ActorCriticNetwork
from rl.agents.random_agent import RandomAgent
from rl.agents.a2c_research import (ModelSpec, ResearchA2CAgent, ResearchNetwork,
    ResearchTacticalAgent, encode_state)
from rl.environment.gomoku import GomokuEnv
from rl.environment.state import observation
from rl.training.a2c_math import masked_distribution
from rl.training.a2c_research import (assert_on_policy, canonical_position_id,
    generate, load_demonstrations, masked_expert_loss, opponent_kind, parser,
    pretrain, reject_seed_overlap, split_position_id, train, transform_sample, transform_spatial)


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for item, other in zip(left, right):
            same(item, other)
    else:
        assert left == right


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def dataset_args(directory):
    return parser().parse_args(['generate', '--output', str(directory),
        '--train-games', '2', '--validation-games', '2', '--expert-depth', '1',
        '--expert-candidates', '3', '--save-every', '1'])


@pytest.fixture
def demonstrations(tmp_path):
    args = dataset_args(tmp_path / 'data')
    generate(args)
    return tmp_path / 'data' / 'demonstrations.pt'


def test_original_architecture_is_exactly_compatible_and_roundtrips(tmp_path):
    torch.manual_seed(12)
    original = ActorCriticNetwork()
    network = ResearchNetwork()
    network.load_state_dict(original.state_dict(), strict=True)
    assert sum(p.numel() for p in network.parameters()) == 1_405_042
    x = torch.randn(3, 4, 9, 9)
    for actual, expected in zip(network(x), original(x)):
        assert torch.equal(actual, expected)
    agent = ResearchA2CAgent()
    agent.net.load_state_dict(original.state_dict())
    agent.save(tmp_path / 'model.pt')
    restored = ResearchA2CAgent()
    restored.load(tmp_path / 'model.pt')
    same(agent.net.state_dict(), restored.net.state_dict())
    logits, value = restored.policy_value(observation(GomokuEnv()))
    assert logits.shape == (81,) and isinstance(value, float)
    payload = torch.load(tmp_path / 'model.pt', weights_only=True)
    payload['input_channels'] = 10
    torch.save(payload, tmp_path / 'tampered.pt')
    with pytest.raises(ValueError, match='metadata'):
        restored.load(tmp_path / 'tampered.pt')


def test_d4_transforms_keep_all_channels_masks_targets_aligned():
    base = torch.zeros(4, 9, 9)
    base[0, 1, 2], base[1, 6, 8], base[2, 6, 8], base[3] = 1, 1, 1, 1
    mask = torch.ones(81, dtype=torch.bool)
    mask[11], mask[62] = False, False
    action = 32
    identity = canonical_position_id(base)
    for symmetry in range(8):
        x, legal, transformed_action = transform_sample(base, mask, action, symmetry)
        assert bool(legal[transformed_action])
        marker = torch.zeros(9, 9)
        marker.flatten()[action] = 1
        assert transform_spatial(marker, symmetry).flatten()[transformed_action] == 1
        assert canonical_position_id(x) == identity
        assert torch.equal(x[1], x[2])


def test_expert_ce_only_changes_policy_path_and_masks_illegal_logits():
    logits = torch.tensor([[1., 2., 100.]], requires_grad=True)
    legal = torch.tensor([[True, True, False]])
    loss = masked_expert_loss(logits, legal, torch.tensor([1]))
    expected = -torch.log_softmax(logits[:, :2], -1)[0, 1]
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad[0, 2] == 0
    network = ResearchNetwork()
    output, _ = network(torch.zeros(2, 4, 9, 9))
    masked_expert_loss(output, torch.ones(2, 81, dtype=torch.bool),
                       torch.tensor([2, 4])).backward()
    assert network.policy_head.weight.grad is not None
    assert network.value_head.weight.grad is None
    with pytest.raises(ValueError, match='legal'):
        masked_expert_loss(logits, legal, torch.tensor([2]))


def test_policy_guard_rejects_rule_override_or_stale_likelihood():
    logits = torch.tensor([[2., 0., -1.]])
    masks = torch.ones_like(logits, dtype=torch.bool)
    action = torch.tensor([0])
    likelihood = masked_distribution(logits, masks).log_prob(action)
    assert_on_policy(logits, masks, action, likelihood)
    with pytest.raises(RuntimeError, match='behavior differs'):
        assert_on_policy(logits, masks, torch.tensor([1]), likelihood)
    # Equal-probability actions cannot be distinguished by likelihood alone.
    uniform = torch.zeros(1, 3)
    same_probability = masked_distribution(uniform, masks).log_prob(torch.tensor([0]))
    with pytest.raises(RuntimeError, match='action override'):
        assert_on_policy(uniform, masks, torch.tensor([1]), same_probability,
                         sampled_actions=torch.tensor([0]))
    policy = ResearchA2CAgent(training=True)
    with pytest.raises(ValueError, match='inference-only'):
        ResearchTacticalAgent(policy)


def test_data_generated_from_disjoint_games_and_d4_leakage_rejected(demonstrations, tmp_path):
    data = load_demonstrations(demonstrations, 'freestyle')
    assert not set(data['train']['position_ids']) & set(data['validation']['position_ids'])
    assert not set(data['train']['game_seeds'].tolist()) & set(data['validation']['game_seeds'].tolist())
    assert data['protocol']['external_trajectories_allowed'] is False
    assert not list(demonstrations.parent.glob('*final*'))
    damaged = copy.deepcopy(data)
    damaged['validation']['base'][0] = transform_spatial(data['train']['base'][0], 5)
    damaged['validation']['position_ids'][0] = canonical_position_id(damaged['validation']['base'][0])
    damaged['validation']['split_ids'][0] = split_position_id(damaged['validation']['base'][0])
    # Make the target and mask match too: the remaining rejection is leakage.
    _, mask, action = transform_sample(data['train']['base'][0], data['train']['masks'][0],
                                       data['train']['actions'][0], 5)
    damaged['validation']['masks'][0] = mask
    damaged['validation']['actions'][0] = action
    torch.save(damaged, tmp_path / 'leak.pt')
    with pytest.raises(ValueError, match='leaked'):
        load_demonstrations(tmp_path / 'leak.pt', 'freestyle')
    with pytest.raises(ValueError, match='overlap'):
        reject_seed_overlap([5_000_001], [5_000_001])
    with pytest.raises(ValueError, match='reserved'):
        reject_seed_overlap([2_000_001], [5_100_001])
    with pytest.raises(ValueError, match='reserved'):
        reject_seed_overlap([3_400_001], [5_100_001])

    # A teacher seeing the same board still leaks labels if last-move differs.
    other_last = copy.deepcopy(data)
    index = next(i for i, base in enumerate(data['train']['base']) if base[1].sum() >= 2)
    source = data['train']['base'][index].clone()
    alternatives = source[1].flatten().nonzero().flatten().tolist()
    new_last = next(action for action in alternatives if source[2].flatten()[action] == 0)
    source[2].zero_()
    source[2].flatten()[new_last] = 1
    assert canonical_position_id(source) != data['train']['position_ids'][index]
    assert split_position_id(source) == data['train']['split_ids'][index]
    other_last['validation']['base'][0] = source
    other_last['validation']['masks'][0] = data['train']['masks'][index]
    other_last['validation']['actions'][0] = data['train']['actions'][index]
    other_last['validation']['position_ids'][0] = canonical_position_id(source)
    other_last['validation']['split_ids'][0] = split_position_id(source)
    torch.save(other_last, tmp_path / 'other_last.pt')
    with pytest.raises(ValueError, match='leaked'):
        load_demonstrations(tmp_path / 'other_last.pt', 'freestyle')


def test_feature_and_architecture_ablation_interfaces_preserve_board_symmetry():
    env = GomokuEnv()
    for action in (30, 0, 31, 1, 32, 2, 33, 9):
        env.step(action)
    state = observation(env)
    base = encode_state(state)
    rich = encode_state(state, ModelSpec(features='tactical10'))
    assert rich.shape == (10, 9, 9) and torch.equal(rich[:4], base)
    # Both ends of this open four are immediate legal wins for the acting side.
    assert rich[6].flatten()[29] == 1 and rich[6].flatten()[34] == 1
    for architecture in ('flat', 'residual'):
        for features in ('base4', 'tactical10'):
            spec = ModelSpec(architecture=architecture, features=features)
            logits, value = ResearchNetwork(spec)(torch.zeros(2, spec.input_channels, 9, 9))
            assert logits.shape == (2, 81) and value.shape == (2,)
    assert [opponent_kind(x) for x in (.0, .5999, .6, .7999, .8, .999)] == [
        'greedy', 'greedy', 'random', 'random', 'selfplay', 'selfplay']


def train_args(directory, episodes, resume=False, dataset=None):
    flags = ['train', '--output', str(directory), '--episodes', str(episodes),
        '--device', 'cpu', '--seed', '17', '--num-envs', '2', '--rollout-steps', '3',
        '--save-every', '4', '--eval-every', '4', '--candidate-every', '4',
        '--validation-games', '1']
    if dataset:
        flags += ['--dataset', str(dataset), '--imitation-coef', '.1',
                  '--imitation-batch-size', '4']
    if resume:
        flags.append('--resume')
    return parser().parse_args(flags)


def test_a2c_plus_imitation_resume_is_bit_exact_and_never_tests(demonstrations, tmp_path):
    full, split = tmp_path / 'full', tmp_path / 'split'
    train(train_args(full, 8, dataset=demonstrations))
    train(train_args(split, 4, dataset=demonstrations))
    train(train_args(split, 8, True, demonstrations))
    one = torch.load(full / 'latest.checkpoint', weights_only=False)
    two = torch.load(split / 'latest.checkpoint', weights_only=False)
    for key in ('net', 'optimizer', 'episode', 'learner_transitions', 'environment_plies',
                'updates', 'best_pure_score', 'best_pure_episode', 'best_pure_weights',
                'opponent_counts', 'transition_counts', 'validation', 'worker_rngs',
                'imitation_rng', 'agent_rng', 'rng'):
        same(one[key], two[key])
    for filename in ('training.csv', 'updates.csv', 'validation.csv'):
        assert read_rows(full / filename) == read_rows(split / filename)
    assert not (split / 'final_games.csv').exists()
    assert not (split / 'final_evaluation.json').exists()
    assert sorted(p.name for p in (split / 'candidates').glob('*.pt')) == [
        'episode_000000.pt', 'episode_000004.pt', 'episode_000008.pt']
    assert two['protocol']['algorithm'] == 'A2C+imitation'


def pretrain_args(directory, dataset, epochs, resume=False):
    flags = ['pretrain', '--output', str(directory), '--dataset', str(dataset),
             '--device', 'cpu', '--seed', '17', '--epochs', str(epochs), '--batch-size', '16']
    if resume:
        flags.append('--resume')
    return parser().parse_args(flags)


def test_imitation_epoch_resume_preserves_optimizer_sampling_and_best(demonstrations, tmp_path):
    full, split = tmp_path / 'pre_full', tmp_path / 'pre_split'
    pretrain(pretrain_args(full, demonstrations, 2))
    pretrain(pretrain_args(split, demonstrations, 1))
    pretrain(pretrain_args(split, demonstrations, 2, True))
    one = torch.load(full / 'latest.checkpoint', weights_only=False)
    two = torch.load(split / 'latest.checkpoint', weights_only=False)
    for key in ('net', 'optimizer', 'epoch', 'updates', 'best_ce', 'best_epoch',
                'best_weights', 'sampler_rng', 'rng'):
        same(one[key], two[key])
    assert read_rows(full / 'pretraining.csv') == read_rows(split / 'pretraining.csv')


def test_generator_restart_replays_same_games_and_binds_teacher_provenance(tmp_path, monkeypatch):
    import rl.training.a2c_research as research
    full, split = tmp_path / 'data_full', tmp_path / 'data_split'
    factory = lambda seed, rules: RandomAgent(seed=seed)
    provenance = {'kind': 'random_fixture', 'checkpoint_sha256': 'fixture-only'}
    generate(dataset_args(full), factory, provenance)
    real_save = research.atomic_torch
    def stop_after_first_commit(path, payload):
        real_save(path, payload)
        if path.name == 'latest.checkpoint' and payload.get('completed_games') == 1:
            raise KeyboardInterrupt()
    with monkeypatch.context() as patch:
        patch.setattr(research, 'atomic_torch', stop_after_first_commit)
        with pytest.raises(KeyboardInterrupt):
            generate(dataset_args(split), factory, provenance)
    resumed_args = dataset_args(split)
    resumed_args.resume = True
    generate(resumed_args, factory, provenance)
    one = load_demonstrations(full / 'demonstrations.pt', 'freestyle')
    two = load_demonstrations(split / 'demonstrations.pt', 'freestyle')
    same(one['train'], two['train'])
    same(one['validation'], two['validation'])
    with pytest.raises(ValueError, match='expert'):
        generate(resumed_args, factory, {'kind': 'different_teacher'})


def test_uncommitted_optimizer_updates_and_log_rows_are_rolled_back_exactly(tmp_path, monkeypatch):
    import rl.training.a2c_research as research
    full, split = tmp_path / 'control', tmp_path / 'crashed'
    train(train_args(full, 8))
    first = torch.load(full / 'candidates' / 'episode_000004.checkpoint', weights_only=False)
    real_append = research.append_csv
    def fail_after_uncommitted_update(path, row):
        real_append(path, row)
        if path.name == 'updates.csv' and row['update'] > first['updates']:
            raise KeyboardInterrupt()
    with monkeypatch.context() as patch:
        patch.setattr(research, 'append_csv', fail_after_uncommitted_update)
        with pytest.raises(KeyboardInterrupt):
            train(train_args(split, 8))
    train(train_args(split, 8, True))
    one = torch.load(full / 'latest.checkpoint', weights_only=False)
    two = torch.load(split / 'latest.checkpoint', weights_only=False)
    for key in ('net', 'optimizer', 'rng', 'worker_rngs', 'updates', 'episode'):
        same(one[key], two[key])
    for filename in ('updates.csv', 'training.csv', 'validation.csv'):
        assert read_rows(full / filename) == read_rows(split / filename)


def test_final_test_requires_exact_frozen_policy_rules_budget_and_sources(tmp_path, monkeypatch):
    import rl.training.a2c_research as research
    model = tmp_path / 'selected.pt'
    ResearchA2CAgent().save(model)
    manifest = {'checkpoint_sha256': research.sha256(model), 'phase': 'test',
        'mode': 'pure', 'rules': 'freestyle', 'tactical_config': None,
        'seeds': {'black': 8_100_000, 'white': 8_200_000},
        'games_per_color': 100, 'opponents': ['random', 'greedy'],
        'source_sha256': research.source_hashes()}
    selection = tmp_path / 'selection.json'
    def never_evaluate(*args, **kwargs):
        pytest.fail('Invalid frozen manifest reached actual test games')
    monkeypatch.setattr(research, 'evaluate', never_evaluate)
    for number, key in enumerate(manifest):
        damaged = {**manifest, key: 'incorrect'}
        selection.write_text(json.dumps(damaged), encoding='utf-8')
        args = parser().parse_args(['evaluate', '--output', str(tmp_path / f'bad{number}'),
            '--model', str(model), '--phase', 'test', '--selection-manifest', str(selection)])
        with pytest.raises(ValueError, match='Frozen selection'):
            research.frozen_evaluate(args)
    selection.write_text(json.dumps(manifest), encoding='utf-8')
    args = parser().parse_args(['evaluate', '--output', str(tmp_path / 'valid'),
        '--model', str(model), '--phase', 'test', '--selection-manifest', str(selection)])
    monkeypatch.setattr(research, 'evaluate', lambda *a, **k: [])
    research.frozen_evaluate(args)
    with pytest.raises(ValueError, match='already exists'):
        research.frozen_evaluate(args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA integration coverage')
def test_real_cuda_eight_env_five_step_rollout_preserves_actual_policy_likelihood(tmp_path):
    checkpoint = Path(__file__).resolve().parents[1] / 'runs/production/a2c4_seed22/best.pt'
    if not checkpoint.exists():
        pytest.skip('Local trained seed22 weights required for the observed TF32 regression')
    flags = ['train', '--output', str(tmp_path / 'cuda8x5'), '--initial-model', str(checkpoint),
        '--episodes', '16', '--device', 'cuda', '--seed', '22', '--num-envs', '8',
        '--rollout-steps', '5', '--save-every', '16', '--eval-every', '16',
        '--candidate-every', '16', '--validation-games', '1']
    train(parser().parse_args(flags))
    audit = torch.load(tmp_path / 'cuda8x5' / 'first_rollout.audit.pt', weights_only=True)
    assert audit['num_envs'] == 8 and audit['rollout_steps'] == 5
    assert len(audit['actual_actions']) == 40
    assert torch.equal(audit['sampled_actions'], audit['actual_actions'])
    assert audit['precision']['cuda_matmul_tf32'] is False
    assert audit['precision']['cudnn_tf32'] is False
    error = (audit['behavior_log_probs'] - audit['recomputed_log_probs']).abs().max()
    assert error < 2e-5
    assert float(error) == audit['metrics']['behavior_max_abs_logprob_error']
    rows = read_rows(tmp_path / 'cuda8x5' / 'updates.csv')
    assert rows and all(float(row['behavior_max_abs_logprob_error']) < 2e-5 for row in rows)
