"""Reproducible, isolated A2C + demonstration research pipeline.

Commands: generate (new expert games only), pretrain (masked imitation), train
(genuinely on-policy A2C, optionally plus a separate imitation loss), evaluate
(explicit frozen evaluation only). Training NEVER automatically runs final tests.
Checkpoints commit at drained episode/epoch/game barriers; an interruption rolls
back the uncommitted part and reproduces it from the saved RNG states.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from rl.agents.a2c_research import (FEATURE_DESCRIPTIONS, ModelSpec,
    ResearchA2CAgent, ResearchTacticalAgent, base_to_state, encode_state)
from rl.agents.greedy_agent import GreedyAgent
from rl.agents.minimax_agent import MinimaxAgent
from rl.agents.random_agent import RandomAgent
from rl.agents.tactics import TacticalConfig
from rl.environment.gomoku import GomokuEnv
from rl.environment.state import observation, state_to_tensor4
from rl.environment.rules import relative_legal_actions
from rl.training.a2c_math import a2c_loss, masked_distribution, signed_n_step_returns
from rl.training.common import (append_csv, atomic_json, atomic_torch, choose,
    evaluate, exclusive_run, restore_rng, rng_checkpoint, seed_everything)


FORMAT = "a2c_research_v1"
DEMO_FORMAT = "a2c_expert_demonstrations_v1"
ROOT = Path(__file__).resolve().parents[2]


def configure_research_precision():
    """Use full FP32 for batch-size-independent on-policy likelihood checks.

    On CUDA, cuDNN TF32 convolution rounding depends on batch geometry. A
    rollout sampled in batches of eight and recomputed in a batch of forty can
    otherwise disagree by ~1e-4 in log probability despite unchanged weights.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    return {'cuda_matmul_tf32': False, 'cudnn_tf32': False,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'behavior_guard_atol': 2e-5, 'behavior_guard_rtol': 2e-5,
            'sampled_action_guard': 'exact identity with the recorded categorical sample'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    names = ('rl/training/a2c_research.py', 'rl/agents/a2c_research.py',
             'rl/training/a2c_math.py', 'rl/training/common.py',
             'rl/environment/gomoku.py', 'rl/environment/state.py',
             'rl/environment/rules.py', 'rl/agents/greedy_agent.py',
             'rl/agents/minimax_agent.py', 'rl/agents/q_learning.py',
             'rl/agents/random_agent.py', 'rl/agents/tactics.py')
    return {name: sha256(ROOT / name) for name in names}


def transform_spatial(tensor, symmetry):
    if not 0 <= symmetry < 8:
        raise ValueError('D4 symmetry must be 0..7')
    rotated = torch.rot90(tensor, symmetry % 4, (-2, -1))
    return rotated.flip(-1) if symmetry >= 4 else rotated


def transform_sample(inputs, mask, action, symmetry):
    size = inputs.shape[-1]
    marker = torch.zeros(size, size, dtype=torch.bool)
    marker.flatten()[int(action)] = True
    mapped_action = int(transform_spatial(marker, symmetry).flatten().nonzero()[0])
    return (transform_spatial(inputs, symmetry),
            transform_spatial(mask.reshape(size, size), symmetry).flatten(), mapped_action)


def canonical_position_id(base):
    # Include last move and side-to-move, since they are actual input features.
    versions = [transform_spatial(base.to(torch.uint8), s).contiguous().numpy().tobytes()
                for s in range(8)]
    return hashlib.sha256(min(versions)).hexdigest()


def split_position_id(base):
    """Split by board and side, even when two last-move markers differ.

    The teacher may ignore last move. Keeping those boards in separate splits
    would leak its labels despite differing complete neural observations.
    """
    return canonical_position_id(base[[0, 1, 3]])


def reject_seed_overlap(train_seeds, validation_seeds):
    if set(train_seeds) & set(validation_seeds):
        raise ValueError('Training and validation game seeds overlap')
    # This research protocol reserves these independent evaluation families.
    reserved = ((2_000_000, 2_400_000), (3_000_000, 3_500_000),
                (8_000_000, 8_500_000))
    if any(lo <= seed < hi for seed in (*train_seeds, *validation_seeds)
           for lo, hi in reserved):
        raise ValueError('Expert games overlap a reserved evaluation seed family')


def fresh_output(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'latest.checkpoint'
    if checkpoint.exists() and not args.resume:
        raise ValueError('Existing checkpoint requires --resume')
    if not checkpoint.exists() and any(path.name != 'run.lock' for path in out.iterdir()):
        raise ValueError('Use a new empty output directory')
    return out, checkpoint


def protocol(args, **extra):
    return {'format': FORMAT, 'config': vars(args).copy(), 'sources': source_hashes(), **extra}


def verify_resume(saved, current, ignored=()):
    if saved['sources'] != current['sources']:
        raise ValueError('Source code changed; use a new experiment')
    ignored = {'output', 'resume', *ignored}
    left = {key: value for key, value in saved['config'].items() if key not in ignored}
    right = {key: value for key, value in current['config'].items() if key not in ignored}
    if left != right:
        changed = sorted(key for key in left.keys() | right.keys()
                         if left.get(key) != right.get(key))
        raise ValueError(f'Cannot resume with changed settings: {changed}')
    # Includes teacher/checkpoint provenance, not just CLI settings.
    for key in (saved.keys() | current.keys()) - {'config', 'sources'}:
        if saved.get(key) != current.get(key):
            raise ValueError(f'Resume input changed: {key}')


def trim_csv(path, field, maximum):
    if not path.exists():
        return
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames
        rows = [row for row in reader if int(row[field]) <= maximum]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _pack_rows(rows):
    if not rows:
        raise ValueError('Expert split has no usable observations')
    return {'base': torch.stack([row[0] for row in rows]).to(torch.uint8),
            'masks': torch.stack([row[1] for row in rows]),
            'actions': torch.tensor([row[2] for row in rows]),
            'game_seeds': torch.tensor([row[3] for row in rows]),
            'position_ids': [row[4] for row in rows],
            'split_ids': [row[5] for row in rows]}


def generate(args, expert_factory=None, expert_metadata=None):
    """Generate full fresh games, never importing old evaluation trajectories."""
    if min(args.train_games, args.validation_games, args.save_every,
           args.expert_depth, args.expert_candidates) <= 0:
        raise ValueError('Dataset game counts and expert settings must be positive')
    if not 0 <= args.expert_probability <= 1 or args.random_opening < 0:
        raise ValueError('Invalid demonstration behavior mixture')
    train_seeds = list(range(args.train_seed, args.train_seed + args.train_games))
    val_seeds = list(range(args.validation_seed, args.validation_seed + args.validation_games))
    reject_seed_overlap(train_seeds, val_seeds)
    out, ckpath = fresh_output(args)
    if expert_factory is not None and not expert_metadata:
        raise ValueError('An injected teacher requires explicit provenance metadata')
    metadata = protocol(args, source='fresh_generated_expert_games',
                        expert=expert_metadata or {'kind': 'minimax',
                            'depth': args.expert_depth, 'candidates': args.expert_candidates},
                        external_trajectories_allowed=False,
                        split='disjoint game seeds; board+side D4 validation overlaps removed regardless of last move')
    rows, seen = {'train': [], 'validation': []}, set()
    completed, dropped = 0, 0
    if ckpath.exists():
        ck = torch.load(ckpath, weights_only=False, map_location='cpu')
        verify_resume(ck['protocol'], metadata)
        completed, rows, seen, dropped = ck['completed_games'], ck['rows'], ck['seen'], ck['dropped']
    schedule = [('train', seed) for seed in train_seeds] + [('validation', seed) for seed in val_seeds]
    for number in range(completed, len(schedule)):
        split, game_seed = schedule[number]
        seed_everything(game_seed)
        rng, env = random.Random(game_seed), GomokuEnv(rules=args.rules)
        teacher = (expert_factory(game_seed, args.rules) if expert_factory is not None else
                   MinimaxAgent(depth=args.expert_depth, candidate_limit=args.expert_candidates,
                                rules=args.rules))
        greedy = GreedyAgent(seed=game_seed + 100_000_000)
        # Diverse fresh legal prefixes, not final-test positions.
        for _ in range(rng.randrange(args.random_opening + 1)):
            if not env.done:
                env.step(rng.choice(env.get_legal_actions()))
        while not env.done:
            state, legal = observation(env), env.get_legal_actions()
            action = choose(teacher, env)
            base = state_to_tensor4(state).to(torch.uint8)
            identity = canonical_position_id(base)
            split_id = split_position_id(base)
            mask = torch.zeros(81, dtype=torch.bool)
            mask[legal] = True
            if split == 'validation' and split_id in seen:
                dropped += 1
            else:
                rows[split].append((base, mask, action, game_seed, identity, split_id))
            if split == 'train':
                seen.add(split_id)
            sample = rng.random()
            if sample < args.expert_probability:
                behavior_action = action
            elif sample < (1 + args.expert_probability) / 2:
                behavior_action = greedy.select_action(env.get_state(), legal)
            else:
                behavior_action = rng.choice(legal)
            env.step(behavior_action)
        completed = number + 1
        if completed % args.save_every == 0 or completed == len(schedule):
            atomic_torch(ckpath, {'protocol': metadata, 'completed_games': completed,
                'rows': rows, 'seen': seen, 'dropped': dropped})
            atomic_json(out / 'status.json', {'state': 'generating', 'pid': os.getpid(),
                'games': completed, 'target_games': len(schedule),
                'train_positions': len(rows['train']), 'validation_positions': len(rows['validation'])})
            print(f'Expert games {completed}/{len(schedule)}', flush=True)
    dataset = {'format': DEMO_FORMAT, 'protocol': metadata, 'rules': args.rules,
               'train': _pack_rows(rows['train']), 'validation': _pack_rows(rows['validation']),
               'validation_d4_overlap_removed': dropped}
    atomic_torch(out / 'demonstrations.pt', dataset)
    atomic_json(out / 'manifest.json', {'format': DEMO_FORMAT,
        'sha256': sha256(out / 'demonstrations.pt'), 'protocol': metadata,
        'train_samples': len(rows['train']), 'validation_samples': len(rows['validation']),
        'validation_d4_overlap_removed': dropped,
        'train_seed_range': [train_seeds[0], train_seeds[-1]],
        'validation_seed_range': [val_seeds[0], val_seeds[-1]]})
    atomic_json(out / 'status.json', {'state': 'complete', 'pid': os.getpid(),
                                     'games': completed, 'target_games': len(schedule)})


def load_demonstrations(path, rules):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('format') != DEMO_FORMAT or data.get('rules') != rules:
        raise ValueError('Not a compatible generated expert dataset')
    if data.get('protocol', {}).get('source') != 'fresh_generated_expert_games':
        raise ValueError('Only fresh generated expert data are accepted')
    for split in ('train', 'validation'):
        part = data[split]
        count = len(part['actions'])
        if count == 0 or part['base'].shape != (count, 4, 9, 9) or \
                part['masks'].shape != (count, 81):
            raise ValueError('Invalid demonstration tensor dimensions')
        actions = part['actions']
        if bool(((actions < 0) | (actions >= 81)).any()) or not \
                bool(part['masks'].gather(1, actions[:, None]).all()):
            raise ValueError('Illegal expert target')
        ids = [canonical_position_id(base) for base in part['base']]
        if ids != part['position_ids']:
            raise ValueError('Demonstration position hash mismatch')
        if [split_position_id(base) for base in part['base']] != part['split_ids']:
            raise ValueError('Demonstration split hash mismatch')
        for base, mask in zip(part['base'], part['masks']):
            state = base_to_state(base)
            if not torch.equal(state_to_tensor4(state).to(torch.uint8), base):
                raise ValueError('Invalid board observation planes')
            expected_mask = torch.zeros(81, dtype=torch.bool)
            expected_mask[relative_legal_actions(state[0], rules)] = True
            if not torch.equal(expected_mask, mask):
                raise ValueError('Stored legal mask differs from rule engine')
    reject_seed_overlap(data['train']['game_seeds'].tolist(),
                        data['validation']['game_seeds'].tolist())
    if set(data['train']['split_ids']) & set(data['validation']['split_ids']):
        raise ValueError('D4 equivalent board leaked across demonstration splits')
    return data


def feature_split(part, spec, rules):
    if spec.features == 'base4':
        return part['base'].float()
    return torch.stack([encode_state(base_to_state(base), spec, rules)
                        for base in part['base']])


def expert_batch(inputs, part, indices, rng=None):
    x, masks, actions = inputs[indices].clone(), part['masks'][indices].clone(), part['actions'][indices].clone()
    if rng is not None:
        for index in range(len(indices)):
            x[index], masks[index], actions[index] = transform_sample(
                x[index], masks[index], actions[index], rng.randrange(8))
    return x, masks, actions


def masked_expert_loss(logits, masks, actions):
    distribution = masked_distribution(logits, masks)
    actions = actions.to(device=logits.device, dtype=torch.long)
    if bool(((actions < 0) | (actions >= logits.shape[1])).any()) or \
            not bool(masks.to(logits.device).gather(1, actions[:, None]).all()):
        raise ValueError('Expert action must be legal')
    return -distribution.log_prob(actions).mean()


def assert_on_policy(logits, masks, actions, recorded_log_probs, sampled_actions=None):
    """Fail if an action override or an intervening optimizer update is detected."""
    if sampled_actions is not None and not torch.equal(
            actions.detach().cpu(), sampled_actions.detach().cpu()):
        raise RuntimeError('Rollout action override: actual action differs from recorded policy sample')
    current = masked_distribution(logits, masks).log_prob(actions)
    recorded = recorded_log_probs.to(current.device)
    errors = (current.detach() - recorded).abs()
    statistics = {
        'behavior_max_abs_logprob_error': float(errors.max()),
        'behavior_max_relative_logprob_error': float(
            (errors / recorded.abs().clamp_min(torch.finfo(recorded.dtype).eps)).max()),
        'behavior_mean_abs_logprob_error': float(errors.mean()),
    }
    if not torch.allclose(current.detach(), recorded,
                          atol=2e-5, rtol=2e-5):
        raise RuntimeError('Rollout behavior differs from the current sampled policy; '
                           + ', '.join(f'{key}={value:.9g}' for key, value in statistics.items()))
    return statistics


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _initial_agent(args):
    spec = ModelSpec(architecture=args.architecture, features=args.features)
    agent = ResearchA2CAgent(spec, args.device, args.seed, args.rules)
    if args.initial_model:
        agent.load(args.initial_model)
    return agent


def _training_protocol(args, agent):
    return protocol(args, spec=asdict(agent.spec),
        parameters=sum(p.numel() for p in agent.net.parameters()),
        feature_names=FEATURE_DESCRIPTIONS[:agent.spec.input_channels],
        initial_sha256=sha256(args.initial_model) if args.initial_model else None,
        dataset_sha256=sha256(args.dataset) if args.dataset else None,
        optimizer_initialization='new Adam; no source optimizer state inherited',
        final_test_automatic=False, torch_version=str(torch.__version__))


def _check_loss_step(loss, agent, optimizer, max_norm):
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Non-finite training loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(agent.net.parameters(), max_norm)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Non-finite gradient')
    optimizer.step()
    return float(norm)


def pretrain(args):
    if min(args.epochs, args.batch_size) < 1 or args.learning_rate <= 0:
        raise ValueError('Invalid supervised training budget')
    configure_research_precision()
    out, ckpath = fresh_output(args)
    data = load_demonstrations(args.dataset, args.rules)
    seed_everything(args.seed)
    agent, sampler = _initial_agent(args), random.Random(args.seed + 777)
    optimizer = torch.optim.Adam(agent.net.parameters(), lr=args.learning_rate)
    metadata = _training_protocol(args, agent)
    metadata['precision'] = configure_research_precision()
    epoch, updates, best_ce, best_epoch, best_weights = 0, 0, math.inf, None, None
    if ckpath.exists():
        ck = torch.load(ckpath, map_location='cpu', weights_only=False)
        verify_resume(ck['protocol'], metadata, ('epochs',))
        if args.epochs < ck['epoch']:
            raise ValueError('Target epoch precedes checkpoint')
        agent.net.load_state_dict(ck['net'])
        optimizer.load_state_dict(ck['optimizer'])
        _optimizer_to_device(optimizer, agent.device)
        epoch, updates, best_ce, best_epoch, best_weights = (ck[k] for k in
            ('epoch', 'updates', 'best_ce', 'best_epoch', 'best_weights'))
        sampler.setstate(ck['sampler_rng'])
        restore_rng(ck['rng'])
        trim_csv(out / 'pretraining.csv', 'epoch', epoch)
    inputs = {split: feature_split(data[split], agent.spec, args.rules)
              for split in ('train', 'validation')}
    atomic_json(out / 'protocol.json', metadata)

    def save():
        atomic_torch(ckpath, {'protocol': metadata, 'net': agent.net.state_dict(),
            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'updates': updates,
            'best_ce': best_ce, 'best_epoch': best_epoch, 'best_weights': best_weights,
            'sampler_rng': sampler.getstate(), 'rng': rng_checkpoint()})
        agent.save(out / 'current.pt', stage='imitation', epoch=epoch)
        atomic_json(out / 'status.json', {'state': 'pretraining', 'pid': os.getpid(),
            'epoch': epoch, 'target_epochs': args.epochs, 'updates': updates,
            'best_validation_ce': best_ce if math.isfinite(best_ce) else None})

    if not ckpath.exists():
        save()
    while epoch < args.epochs:
        agent.net.train()
        indices = list(range(len(data['train']['actions'])))
        sampler.shuffle(indices)
        total, correct = 0.0, 0
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            x, masks, actions = expert_batch(inputs['train'], data['train'], batch, sampler)
            logits, _ = agent.net(x.to(agent.device))
            loss = masked_expert_loss(logits, masks.to(agent.device), actions.to(agent.device))
            _check_loss_step(loss, agent, optimizer, args.max_gradient_norm)
            total += float(loss.detach()) * len(batch)
            updates += 1
        agent.net.eval()
        validation_ce = 0.0
        with torch.inference_mode():
            count = len(data['validation']['actions'])
            for start in range(0, count, args.batch_size):
                batch = list(range(start, min(start + args.batch_size, count)))
                x, masks, actions = expert_batch(inputs['validation'], data['validation'], batch)
                logits, _ = agent.net(x.to(agent.device))
                validation_ce += float(masked_expert_loss(logits, masks.to(agent.device),
                                                         actions.to(agent.device))) * len(batch)
                predictions = masked_distribution(logits, masks.to(agent.device)).logits.argmax(-1)
                correct += int((predictions.cpu() == actions).sum())
        epoch += 1
        validation_ce /= count
        if validation_ce < best_ce:
            best_ce, best_epoch = validation_ce, epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in agent.net.state_dict().items()}
        append_csv(out / 'pretraining.csv', {'epoch': epoch, 'updates': updates,
            'train_ce': total / len(indices), 'validation_ce': validation_ce,
            'validation_accuracy': correct / count, 'best_epoch': best_epoch})
        save()
        print(f'Imitation epoch={epoch}/{args.epochs} val_ce={validation_ce:.5f}', flush=True)
    agent.save(out / 'final.pt', stage='imitation', epoch=epoch)
    agent.net.load_state_dict(best_weights)
    agent.save(out / 'best.pt', stage='imitation', epoch=best_epoch,
               selection='held-out expert cross entropy only')
    atomic_json(out / 'status.json', {'state': 'complete', 'pid': os.getpid(),
        'epoch': epoch, 'target_epochs': args.epochs, 'updates': updates,
        'best_epoch': best_epoch, 'best_validation_ce': best_ce})


def opponent_kind(sample):
    """Fixed per-episode mixture; reported transition shares may differ."""
    return 'greedy' if sample < .6 else ('random' if sample < .8 else 'selfplay')


def train(args):
    positive = (args.episodes, args.num_envs, args.rollout_steps, args.save_every,
                args.eval_every, args.candidate_every, args.validation_games,
                args.imitation_batch_size)
    if min(positive) < 1 or not 0 < args.gamma <= 1 or args.learning_rate <= 0 or \
            min(args.value_coef, args.entropy_coef, args.imitation_coef) < 0:
        raise ValueError('Invalid A2C budget or coefficients')
    if args.imitation_coef and not args.dataset:
        raise ValueError('Imitation regularization requires a fresh expert dataset')
    configure_research_precision()
    out, ckpath = fresh_output(args)
    seed_everything(args.seed)
    agent = _initial_agent(args)
    agent.net.train()
    net, device = agent.net, agent.device
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    metadata = _training_protocol(args, agent)
    metadata['precision'] = configure_research_precision()
    metadata.update(algorithm='A2C+imitation' if args.imitation_coef else 'A2C',
        opponent_episode_probabilities={'greedy': .6, 'random': .2, 'selfplay': .2},
        rollout_behavior='unmodified legal-mask categorical policy samples',
        value_target='signed n-step returns; selfplay sign=-1, fixed opponent sign=+1',
        gamma_unit='learner transition', reward='terminal only',
        best_pure_selection='mean black/white Greedy validation score only',
        best_tactical_selection='external frozen tactical development evaluation; NOT best_pure')
    dataset = load_demonstrations(args.dataset, args.rules) if args.dataset else None
    expert_inputs = feature_split(dataset['train'], agent.spec, args.rules) if dataset else None
    worker_rngs = [random.Random(args.seed * 10007 + i) for i in range(args.num_envs)]
    imitation_rng = random.Random(args.seed + 888)
    completed = allocated = transitions = plies = updates = 0
    best, best_episode, best_weights = -1.0, None, None
    opponent_counts, transition_counts, validation = {}, {}, []
    wall_offset, started = 0.0, time.perf_counter()
    if ckpath.exists():
        ck = torch.load(ckpath, map_location='cpu', weights_only=False)
        verify_resume(ck['protocol'], metadata, ('episodes', 'skip_final'))
        if args.episodes < ck['episode']:
            raise ValueError('Target episodes precede durable checkpoint')
        net.load_state_dict(ck['net'])
        optimizer.load_state_dict(ck['optimizer'])
        _optimizer_to_device(optimizer, device)
        completed, transitions, plies, updates, best, best_episode, best_weights = (
            ck[k] for k in ('episode', 'learner_transitions', 'environment_plies',
                           'updates', 'best_pure_score', 'best_pure_episode', 'best_pure_weights'))
        allocated = completed
        opponent_counts, transition_counts, validation = (ck[k] for k in
            ('opponent_counts', 'transition_counts', 'validation'))
        for rng, state in zip(worker_rngs, ck['worker_rngs']):
            rng.setstate(state)
        imitation_rng.setstate(ck['imitation_rng'])
        agent.rng.setstate(ck['agent_rng'])
        restore_rng(ck['rng'])
        wall_offset = ck['wall_seconds']
        for name, field, maximum in (('training.csv', 'completion_index', completed),
                                     ('updates.csv', 'update', updates),
                                     ('validation.csv', 'episode', completed)):
            trim_csv(out / name, field, maximum)
    atomic_json(out / 'protocol.json', metadata)

    def checkpoint_payload():
        return {'protocol': metadata, 'net': net.state_dict(),
            'optimizer': optimizer.state_dict(), 'episode': completed,
            'learner_transitions': transitions, 'environment_plies': plies, 'updates': updates,
            'best_pure_score': best, 'best_pure_episode': best_episode,
            'best_pure_weights': best_weights, 'opponent_counts': opponent_counts,
            'transition_counts': transition_counts, 'validation': validation,
            'worker_rngs': [rng.getstate() for rng in worker_rngs],
            'imitation_rng': imitation_rng.getstate(), 'agent_rng': agent.rng.getstate(),
            'rng': rng_checkpoint(), 'wall_seconds': wall_offset + time.perf_counter() - started}

    def save(candidate=False):
        payload = checkpoint_payload()
        atomic_torch(ckpath, payload)
        agent.save(out / 'current.pt', stage=metadata['algorithm'], episode=completed)
        if best_weights is not None:
            best_payload = agent.payload(stage=metadata['algorithm'], episode=best_episode,
                                         selection=metadata['best_pure_selection'])
            best_payload['state_dict'] = best_weights
            atomic_torch(out / 'best_pure.pt', best_payload)
        if candidate:
            # Re-running after a crash may recreate only the SAME committed state.
            path = out / 'candidates' / f'episode_{completed:06d}.pt'
            if path.exists():
                old = torch.load(path, map_location='cpu', weights_only=True)
                if any(not torch.equal(old['state_dict'][k], v.detach().cpu())
                       for k, v in net.state_dict().items()):
                    raise RuntimeError('Immutable candidate differs from replayed state')
            else:
                agent.save(path, stage=metadata['algorithm'], episode=completed)
            cp = path.with_suffix('.checkpoint')
            if not cp.exists():
                atomic_torch(cp, payload)
        atomic_json(out / 'status.json', {'state': 'training', 'pid': os.getpid(),
            'episode': completed, 'target_episodes': args.episodes, 'updates': updates,
            'learner_transitions': transitions, 'environment_plies': plies,
            'best_pure_score': best, 'best_pure_episode': best_episode,
            'opponent_counts': opponent_counts, 'transition_counts': transition_counts,
            'wall_seconds': payload['wall_seconds']})

    def validate():
        nonlocal best, best_episode, best_weights
        rows = evaluate(agent, args.validation_games, args.validation_seed, args.rules,
                        out / f'validation_{completed:06d}_games.csv')
        score = sum(row['score'] for row in rows if row['opponent'] == 'greedy') / 2
        validation.extend({'episode': completed, **row} for row in rows)
        for row in rows:
            append_csv(out / 'validation.csv', {'episode': completed, **row})
        if score > best:
            best, best_episode = score, completed
            best_weights = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    def allocate(index, boundary):
        nonlocal allocated, plies
        if allocated >= boundary:
            return None
        allocated += 1
        rng = worker_rngs[index]
        kind = opponent_kind(rng.random())
        opponent = (GreedyAgent(seed=rng.randrange(2**31)) if kind == 'greedy' else
                    RandomAgent(seed=rng.randrange(2**31)) if kind == 'random' else None)
        color = 1 if allocated % 2 else -1
        worker = {'env': GomokuEnv(rules=args.rules), 'opponent': opponent,
                  'kind': kind, 'episode': allocated, 'color': color,
                  'moves': 0, 'transitions': 0}
        if opponent is not None and color == -1:
            worker['env'].step(choose(opponent, worker['env']))
            worker['moves'] += 1
            plies += 1
        return worker

    if not ckpath.exists():
        validate()
        save(candidate=True)
    # Repair derived artifacts from the last committed checkpoint on resume.
    else:
        save(candidate=completed % args.candidate_every == 0)
    while completed < args.episodes:
        boundary = min(args.episodes, *[(completed // interval + 1) * interval
            for interval in (args.save_every, args.eval_every, args.candidate_every)])
        workers = [allocate(i, boundary) for i in range(args.num_envs)]
        while any(worker is not None for worker in workers):
            shape = (args.rollout_steps, args.num_envs)
            inputs = torch.zeros(*shape, agent.spec.input_channels, 9, 9)
            masks = torch.zeros(*shape, 81, dtype=torch.bool)
            actions = torch.zeros(shape, dtype=torch.long)
            sampled_actions = torch.zeros(shape, dtype=torch.long)
            rewards, behavior = torch.zeros(shape), torch.zeros(shape)
            dones, valid = torch.zeros(shape, dtype=torch.bool), torch.zeros(shape, dtype=torch.bool)
            signs, finished = torch.ones(shape), []
            for step in range(args.rollout_steps):
                active = [i for i, worker in enumerate(workers) if worker is not None]
                if not active:
                    break
                for index in active:
                    env = workers[index]['env']
                    inputs[step, index] = agent.encode(observation(env))
                    masks[step, index, env.get_legal_actions()] = True
                    valid[step, index] = True
                with torch.no_grad():
                    logits, _ = net(inputs[step, active].to(device))
                    distribution = masked_distribution(logits, masks[step, active].to(device))
                    picked = distribution.sample()
                    sampled_actions[step, active] = picked.cpu()
                    behavior[step, active] = distribution.log_prob(picked).cpu()
                for index, action in zip(active, picked.cpu().tolist()):
                    worker = workers[index]
                    env, opponent = worker['env'], worker['opponent']
                    _, reward, done = env.step(action)
                    actions[step, index] = action
                    worker['moves'] += 1
                    worker['transitions'] += 1
                    transitions += 1
                    plies += 1
                    transition_counts[worker['kind']] = transition_counts.get(worker['kind'], 0) + 1
                    if opponent is None:
                        signs[step, index] = -1
                    else:
                        if not done:
                            env.step(choose(opponent, env))
                            worker['moves'] += 1
                            plies += 1
                        done = env.done
                        reward = (0.0 if not done or env.winner == 0 else
                                  1.0 if env.winner == worker['color'] else -1.0)
                    rewards[step, index], dones[step, index] = reward, done
                    if done:
                        completed += 1
                        label = f"{worker['kind']}:{worker['color'] if opponent else 'both'}"
                        opponent_counts[label] = opponent_counts.get(label, 0) + 1
                        finished.append({'episode': worker['episode'],
                            'completion_index': completed, 'kind': worker['kind'],
                            'color': worker['color'] if opponent else 0,
                            'winner': env.winner, 'moves': worker['moves'],
                            'game_transitions': worker['transitions']})
                        workers[index] = allocate(index, boundary)
            bootstrap = torch.zeros(args.num_envs, device=device)
            active = [i for i, worker in enumerate(workers) if worker is not None]
            if active:
                with torch.no_grad():
                    bx = torch.stack([agent.encode(observation(workers[i]['env'])) for i in active])
                    _, values = net(bx.to(device))
                    bootstrap[active] = values
            targets = signed_n_step_returns(rewards.to(device), dones.to(device),
                signs.to(device), bootstrap, valid.to(device), args.gamma)
            logits, values = net(inputs[valid].to(device))
            legal, chosen = masks[valid].to(device), actions[valid].to(device)
            behavior_metrics = assert_on_policy(logits, legal, chosen, behavior[valid],
                                               sampled_actions[valid])
            if updates == 0:
                atomic_torch(out / 'first_rollout.audit.pt', {
                    'inputs': inputs[valid].clone(), 'legal_masks': masks[valid].clone(),
                    'actual_actions': actions[valid].clone(),
                    'sampled_actions': sampled_actions[valid].clone(),
                    'behavior_log_probs': behavior[valid].clone(),
                    'recomputed_log_probs': masked_distribution(logits, legal).log_prob(chosen).detach().cpu(),
                    'precision': metadata['precision'], 'metrics': behavior_metrics,
                    'optimizer_updates_before_collection': 0,
                    'num_envs': args.num_envs, 'rollout_steps': args.rollout_steps})
            loss, metrics = a2c_loss(logits, values, chosen, targets[valid.to(device)],
                                    legal, args.value_coef, args.entropy_coef)
            ce = torch.zeros((), device=device)
            if args.imitation_coef:
                count = len(dataset['train']['actions'])
                indices = [imitation_rng.randrange(count) for _ in range(args.imitation_batch_size)]
                x, lm, expert_actions = expert_batch(expert_inputs, dataset['train'], indices, imitation_rng)
                expert_logits, _ = net(x.to(device))
                ce = masked_expert_loss(expert_logits, lm.to(device), expert_actions.to(device))
                loss = loss + args.imitation_coef * ce
            gradient = _check_loss_step(loss, agent, optimizer, args.max_gradient_norm)
            updates += 1
            append_csv(out / 'updates.csv', {'update': updates, 'episode': completed,
                'samples': int(valid.sum()), **metrics, 'imitation_ce': float(ce.detach()),
                'imitation_coef': args.imitation_coef, 'combined_loss': float(loss.detach()),
                'gradient_norm': gradient, 'learner_transitions': transitions,
                'environment_plies': plies, 'behavior': 'masked_policy_sample',
                **behavior_metrics})
            for row in finished:
                append_csv(out / 'training.csv', {**row, 'updates': updates,
                    'learner_transitions': transitions, 'environment_plies': plies})
        if completed != boundary or allocated != boundary:
            raise RuntimeError('Episode barrier failed to drain exactly')
        if completed % args.eval_every == 0:
            validate()
        save(candidate=completed % args.candidate_every == 0 or completed == args.episodes)
        print(f'Research A2C {completed}/{args.episodes} updates={updates} '
              f'best_pure={best:.3f} transition_counts={transition_counts}', flush=True)
    agent.save(out / 'final.pt', stage=metadata['algorithm'], episode=completed)
    status = json.loads((out / 'status.json').read_text(encoding='utf-8'))
    atomic_json(out / 'status.json', {**status, 'state': 'complete',
        'final_test': 'not run; requires explicit evaluate after selection is frozen'})


def frozen_evaluate(args):
    precision = configure_research_precision()
    if not 1 <= args.games < 100_000:
        raise ValueError('Evaluation counts must fit the disjoint color seed ranges')
    out = Path(args.output)
    if out.exists() and any(p.name != 'run.lock' for p in out.iterdir()):
        raise ValueError('Frozen evaluation output already exists; do not rerun final tests')
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'test' and not args.selection_manifest:
        raise ValueError('Final testing requires a frozen selection manifest')
    if args.phase == 'test' and args.seed != (8_100_000 if args.rules == 'freestyle' else 8_300_000):
        raise ValueError('This study reserves test seeds 8.1m/8.2m or Renju 8.3m/8.4m')
    selection = None
    expected = {'checkpoint_sha256': sha256(args.model), 'phase': args.phase,
        'mode': 'tactical' if args.tactical else 'pure', 'rules': args.rules,
        'tactical_config': asdict(TacticalConfig()) if args.tactical else None,
        'seeds': {'black': args.seed, 'white': args.seed + 100_000},
        'games_per_color': args.games, 'opponents': ['random', 'greedy'],
        'source_sha256': source_hashes()}
    if args.selection_manifest:
        selection = json.loads(Path(args.selection_manifest).read_text(encoding='utf-8'))
        for key, value in expected.items():
            if selection.get(key) != value:
                raise ValueError(f'Frozen selection does not match evaluation {key}')
    payload = torch.load(args.model, map_location='cpu', weights_only=True)
    spec = ModelSpec.from_dict(payload['spec'])
    seed_everything(args.seed)
    agent = ResearchA2CAgent(spec, args.device, args.seed, args.rules)
    agent.load(args.model, allow_rule_transfer=args.allow_rule_transfer)
    evaluated = ResearchTacticalAgent(agent) if args.tactical else agent
    atomic_json(out / 'protocol.json', protocol(args,
        **expected, selection=selection,
        training_rules=payload['rules'], evaluation_rules=args.rules,
        precision=precision,
        policy='tactical wrapper' if args.tactical else 'pure network'))
    rows = evaluate(evaluated, args.games, args.seed, args.rules, out / 'games.csv')
    if expected['source_sha256'] != source_hashes() or expected['checkpoint_sha256'] != sha256(args.model):
        raise RuntimeError('Frozen code or checkpoint changed during evaluation')
    atomic_json(out / 'summary.json', rows)
    atomic_json(out / 'status.json', {'state': 'complete', 'pid': os.getpid()})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--output', required=True)
    gen.add_argument('--train-games', type=int, default=2000)
    gen.add_argument('--validation-games', type=int, default=200)
    gen.add_argument('--train-seed', type=int, default=5_000_000)
    gen.add_argument('--validation-seed', type=int, default=5_100_000)
    gen.add_argument('--expert-depth', type=int, default=2)
    gen.add_argument('--expert-candidates', type=int, default=8)
    gen.add_argument('--expert-probability', type=float, default=.8)
    gen.add_argument('--random-opening', type=int, default=6)
    gen.add_argument('--save-every', type=int, default=25)
    for cmd in (gen,):
        cmd.add_argument('--rules', choices=('freestyle', 'renju'), default='freestyle')
        cmd.add_argument('--resume', action='store_true')
    pre, tr = sub.add_parser('pretrain'), sub.add_parser('train')
    for cmd in (pre, tr):
        cmd.add_argument('--output', required=True)
        cmd.add_argument('--initial-model')
        cmd.add_argument('--dataset', required=cmd is pre)
        cmd.add_argument('--architecture', choices=('flat', 'residual'), default='flat')
        cmd.add_argument('--features', choices=('base4', 'tactical10'), default='base4')
        cmd.add_argument('--rules', choices=('freestyle', 'renju'), default='freestyle')
        cmd.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
        cmd.add_argument('--seed', type=int, default=11)
        cmd.add_argument('--learning-rate', type=float, default=3e-4)
        cmd.add_argument('--max-gradient-norm', type=float, default=5.0)
        cmd.add_argument('--resume', action='store_true')
    pre.add_argument('--epochs', type=int, default=20)
    pre.add_argument('--batch-size', type=int, default=256)
    tr.add_argument('--episodes', type=int, default=30000)
    tr.add_argument('--num-envs', type=int, default=8)
    tr.add_argument('--rollout-steps', type=int, default=5)
    tr.add_argument('--gamma', type=float, default=.99)
    tr.add_argument('--value-coef', type=float, default=.5)
    tr.add_argument('--entropy-coef', type=float, default=.01)
    tr.add_argument('--imitation-coef', type=float, default=0.)
    tr.add_argument('--imitation-batch-size', type=int, default=64)
    tr.add_argument('--save-every', type=int, default=250)
    tr.add_argument('--eval-every', type=int, default=5000)
    tr.add_argument('--candidate-every', type=int, default=5000)
    tr.add_argument('--validation-games', type=int, default=40)
    tr.add_argument('--validation-seed', type=int, default=3_000_000)
    tr.add_argument('--skip-final', action='store_true', default=True,
                    help='Final tests are always separate; retained as an explicit protocol flag')
    ev = sub.add_parser('evaluate')
    ev.add_argument('--output', required=True)
    ev.add_argument('--model', required=True)
    ev.add_argument('--phase', choices=('validation', 'test'), required=True)
    ev.add_argument('--selection-manifest')
    ev.add_argument('--tactical', action='store_true')
    ev.add_argument('--allow-rule-transfer', action='store_true')
    ev.add_argument('--games', type=int, default=100)
    ev.add_argument('--seed', type=int, default=8_100_000)
    ev.add_argument('--rules', choices=('freestyle', 'renju'), default='freestyle')
    ev.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    return p


def main():
    args = parser().parse_args()
    with exclusive_run(Path(args.output) / 'run.lock'):
        try:
            {'generate': generate, 'pretrain': pretrain, 'train': train,
             'evaluate': frozen_evaluate}[args.command](args)
        except BaseException as error:
            if Path(args.output).exists():
                atomic_json(Path(args.output) / 'failure.json', {
                    'pid': os.getpid(), 'type': type(error).__name__, 'message': str(error),
                    'resume': 'latest.checkpoint, rolling back the uncommitted barrier'})
                status_path = Path(args.output) / 'status.json'
                if status_path.exists():
                    status = json.loads(status_path.read_text(encoding='utf-8'))
                    atomic_json(status_path, {**status,
                        'state': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        'error_type': type(error).__name__, 'message': str(error)})
            raise


if __name__ == '__main__':
    main()
