"""Synchronous masked A2C with signed zero-sum returns and barrier checkpoints."""

import argparse
import copy
import csv
import os
from pathlib import Path
import random
import statistics
import time

import torch

from rl.agents.a2c import A2CAgent
from rl.agents.greedy_agent import GreedyAgent
from rl.agents.random_agent import RandomAgent
from rl.environment.gomoku import GomokuEnv
from rl.environment.state import observation, state_to_tensor4
from rl.training.a2c_math import a2c_loss, masked_distribution, signed_n_step_returns
from rl.training.common import (
    append_csv, atomic_json, atomic_torch, check_resume_config, choose,
    curriculum_ratio, evaluate, exclusive_run, prepare_resume_logs,
    restore_rng, rng_checkpoint, seed_everything,
)


class FrozenOpponent:
    needs_last_move = True

    def __init__(self, network, device, seed):
        self.net, self.device = network, device
        self.rng = random.Random(seed)

    def select_action(self, state, legal_actions):
        if self.rng.random() < 0.05:
            return self.rng.choice(legal_actions)
        with torch.inference_mode():
            logits, _ = self.net(state_to_tensor4(state).unsqueeze(0).to(self.device))
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[0, legal_actions] = True
            return int(logits.masked_fill(~mask, float('-inf')).argmax(1).item())


def trim_update_log(path, last_update):
    if not path.exists():
        return
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row['update']) <= last_update]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def measure_latency(agent, rules):
    """Frozen batch-one action latency including encoding and device transfers."""
    rng = random.Random(1729)
    positions = []
    for moves in (4, 12, 20, 28):
        env = GomokuEnv(rules=rules)
        for _ in range(moves):
            env.step(rng.choice(env.get_legal_actions()))
            if env.done:
                break
        if not env.done:
            positions.append((observation(env), env.get_legal_actions()))
    for state, legal in positions:
        agent.select_action(state, legal)
    samples = []
    for _ in range(12):
        for state, legal in positions:
            start = time.perf_counter()
            agent.select_action(state, legal)
            samples.append(1000 * (time.perf_counter() - start))
    samples.sort()
    return {'device': str(agent.device), 'batch_size': 1, 'samples': len(samples),
            'median_ms': statistics.median(samples),
            'p95_ms': samples[min(len(samples) - 1, int(0.95 * len(samples)))],
            'protocol': 'fixed synthetic legal positions; encoding, transfer and argmax included'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='runs/production/a2c4_seed11')
    p.add_argument('--episodes', type=int, default=30000)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--rules', choices=['freestyle', 'renju'], default='freestyle')
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--rollout-steps', type=int, default=5)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--value-coef', type=float, default=0.5)
    p.add_argument('--entropy-coef', type=float, default=0.01)
    p.add_argument('--max-gradient-norm', type=float, default=5.0)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--history-every', type=int, default=1000)
    p.add_argument('--validation-games', type=int, default=40)
    p.add_argument('--final-games', type=int, default=100)
    p.add_argument('--resume', action='store_true')
    return p


def run(args):
    wall_started = time.perf_counter()
    previous_wall_seconds = 0.0
    positive = ['episodes', 'num_envs', 'rollout_steps', 'save_every', 'eval_every',
                'history_every', 'validation_games', 'final_games']
    if any(getattr(args, key) <= 0 for key in positive):
        raise ValueError('Episode, rollout and evaluation counts must be positive')
    if not 0 < args.gamma <= 1 or args.learning_rate <= 0:
        raise ValueError('Invalid discount or learning rate')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ckpath = out / 'latest.checkpoint'
    if ckpath.exists() and not args.resume:
        raise ValueError('Existing checkpoint requires --resume; use a new output otherwise')
    seed_everything(args.seed)
    agent = A2CAgent(device=args.device, seed=args.seed)
    net, device = agent.net, agent.device
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    workers_rng = [random.Random(args.seed * 10007 + index) for index in range(args.num_envs)]
    completed = transitions = plies = updates = 0
    allocated = 0
    saved_episode = 0
    ratio, best = 0.1, -1.0
    best_episode = None
    training_seconds = validation_seconds = test_seconds = 0.0
    history_pool, history_networks = [], []
    opponent_counts = {}
    checkpoint = None
    if ckpath.exists():
        checkpoint = torch.load(ckpath, map_location='cpu', weights_only=False)
        check_resume_config(checkpoint['config'], vars(args), [
            'seed', 'device', 'rules', 'num_envs', 'rollout_steps', 'gamma',
            'learning_rate', 'value_coef', 'entropy_coef', 'max_gradient_norm',
            'save_every', 'eval_every', 'history_every', 'validation_games', 'final_games',
        ])
        completed = checkpoint['episode']
        if args.episodes < completed:
            raise ValueError('Target episodes cannot precede the durable checkpoint')
        allocated = completed
        saved_episode = completed
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        transitions, plies, updates = [checkpoint[key] for key in
                                      ['learner_transitions', 'environment_plies', 'updates']]
        ratio, best, best_episode = [checkpoint[key] for key in
                                    ['ratio', 'best', 'best_episode']]
        training_seconds, validation_seconds = [checkpoint[key] for key in
                                                ['training_seconds', 'validation_seconds']]
        previous_wall_seconds = checkpoint.get('wall_seconds', 0.0)
        opponent_counts = checkpoint['opponent_counts']
        history_pool = checkpoint['history_pool']
        for weights in history_pool:
            frozen = copy.deepcopy(net).eval()
            frozen.load_state_dict(weights)
            history_networks.append(frozen)
        for rng, state in zip(workers_rng, checkpoint['worker_rngs']):
            rng.setstate(state)
        agent.rng.setstate(checkpoint['agent_rng'])
        prepare_resume_logs(out, completed)
        trim_update_log(out / 'updates.csv', updates)
        restore_rng(checkpoint['rng'])
    atomic_json(out / 'config.json', {
        **vars(args), 'parameters': sum(p.numel() for p in net.parameters()),
        'channels': [32, 64, 64], 'hidden_units': 256, 'input_channels': 4,
        'device_used': str(device), 'algorithm': 'synchronous_n_step_A2C',
        'gamma_unit': 'learner transition; signed for alternating self-play',
        'reward_shaping': False, 'torch_version': str(torch.__version__),
    })

    def save():
        nonlocal saved_episode
        agent.save(out / 'current.pt')
        atomic_torch(ckpath, {
            'net': net.state_dict(), 'optimizer': optimizer.state_dict(),
            'episode': completed, 'completed_episode_ids': list(range(1, completed + 1)),
            'next_episode_id': completed + 1, 'learner_transitions': transitions,
            'environment_plies': plies, 'updates': updates, 'ratio': ratio,
            'best': best, 'best_episode': best_episode,
            'history_pool': history_pool, 'worker_rngs': [rng.getstate() for rng in workers_rng],
            'agent_rng': agent.rng.getstate(), 'rng': rng_checkpoint(),
            'training_seconds': training_seconds, 'validation_seconds': validation_seconds,
            'wall_seconds': previous_wall_seconds + time.perf_counter() - wall_started,
            'opponent_counts': opponent_counts, 'config': vars(args),
        })
        saved_episode = completed
        atomic_json(out / 'status.json', {
            'state': 'training', 'episode': completed, 'target_episodes': args.episodes,
            'pid': os.getpid(),
            'device': str(device), 'steps': transitions, 'environment_plies': plies,
            'learner_transitions': transitions, 'actual_rollout_samples': transitions,
            'updates': updates, 'ratio': ratio, 'validation_score': best,
            'best_episode': best_episode, 'training_seconds': training_seconds,
            'validation_seconds': validation_seconds,
            'wall_seconds': previous_wall_seconds + time.perf_counter() - wall_started,
            'opponent_counts_by_kind_and_color': opponent_counts,
        })

    def allocate(index, boundary):
        nonlocal allocated, plies
        if allocated >= boundary:
            return None
        allocated += 1
        rng = workers_rng[index]
        sample = rng.random()
        opponent = None
        if sample < ratio:
            kind, opponent = 'greedy', GreedyAgent(seed=rng.randrange(2**31))
        elif sample < ratio + 0.15:
            kind, opponent = 'random', RandomAgent(seed=rng.randrange(2**31))
        elif history_networks and sample < ratio + 0.30:
            kind = 'history'
            opponent = FrozenOpponent(rng.choice(history_networks), device, rng.randrange(2**31))
        else:
            kind = 'selfplay'
        color = 1 if allocated % 2 else -1
        env = GomokuEnv(rules=args.rules)
        worker = {'env': env, 'opponent': opponent, 'kind': kind, 'episode': allocated,
                  'color': color, 'moves': 0, 'transitions': 0, 'ratio': ratio}
        if opponent is not None and color == -1:
            env.step(choose(opponent, env))
            worker['moves'] += 1
            plies += 1
        return worker

    if checkpoint is None:
        save()
    print(f'A2C resume={completed} target={args.episodes} seed={args.seed} device={device}', flush=True)
    try:
        net.train()
        while completed < args.episodes:
            boundary = min(args.episodes, *[
                (completed // interval + 1) * interval
                for interval in [args.save_every, args.eval_every, args.history_every]
            ])
            allocation_started = time.perf_counter()
            workers = [allocate(index, boundary) for index in range(args.num_envs)]
            training_seconds += time.perf_counter() - allocation_started
            while any(worker is not None for worker in workers):
                started = time.perf_counter()
                shape = (args.rollout_steps, args.num_envs)
                inputs = torch.zeros(*shape, 4, 9, 9)
                masks = torch.zeros(*shape, 81, dtype=torch.bool)
                actions = torch.zeros(shape, dtype=torch.long)
                rewards = torch.zeros(shape)
                dones = torch.zeros(shape, dtype=torch.bool)
                signs = torch.ones(shape)
                valid = torch.zeros(shape, dtype=torch.bool)
                finished = []
                for step in range(args.rollout_steps):
                    active = [index for index, worker in enumerate(workers) if worker is not None]
                    if not active:
                        break
                    for index in active:
                        env = workers[index]['env']
                        inputs[step, index] = state_to_tensor4(observation(env))
                        masks[step, index, env.get_legal_actions()] = True
                        valid[step, index] = True
                    with torch.no_grad():
                        logits, _ = net(inputs[step, active].to(device))
                        picked = masked_distribution(logits, masks[step, active].to(device)).sample().cpu().tolist()
                    for index, action in zip(active, picked):
                        worker = workers[index]
                        env, opponent = worker['env'], worker['opponent']
                        _, reward, done = env.step(action)
                        actions[step, index] = action
                        worker['moves'] += 1
                        worker['transitions'] += 1
                        transitions += 1
                        plies += 1
                        if opponent is None:
                            signs[step, index] = -1
                        else:
                            if not done:
                                env.step(choose(opponent, env))
                                worker['moves'] += 1
                                plies += 1
                            done = env.done
                            reward = (0.0 if not done or env.winner == 0 else
                                      (1.0 if env.winner == worker['color'] else -1.0))
                        rewards[step, index], dones[step, index] = reward, done
                        if done:
                            completed += 1
                            label = f"{worker['kind']}:{worker['color'] if opponent else 'both'}"
                            opponent_counts[label] = opponent_counts.get(label, 0) + 1
                            finished.append({
                                'episode': worker['episode'], 'completion_index': completed,
                                'kind': worker['kind'], 'color': worker['color'] if opponent else 0,
                                'winner': env.winner, 'moves': worker['moves'],
                                'game_transitions': worker['transitions'], 'greedy_ratio': worker['ratio'],
                            })
                            workers[index] = allocate(index, boundary)
                bootstrap = torch.zeros(args.num_envs, device=device)
                active = [index for index, worker in enumerate(workers) if worker is not None]
                if active:
                    with torch.no_grad():
                        bx = torch.stack([state_to_tensor4(observation(workers[index]['env'])) for index in active])
                        _, values = net(bx.to(device))
                        bootstrap[active] = values
                returns = signed_n_step_returns(
                    rewards.to(device), dones.to(device), signs.to(device), bootstrap,
                    valid.to(device), args.gamma,
                )
                x, legal = inputs[valid].to(device), masks[valid].to(device)
                logits, values = net(x)
                loss, metrics = a2c_loss(logits, values, actions[valid].to(device),
                                         returns[valid.to(device)], legal,
                                         args.value_coef, args.entropy_coef)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError('Non-finite A2C loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_gradient_norm)
                if not bool(torch.isfinite(gradient).item()):
                    raise FloatingPointError('Non-finite A2C gradient')
                optimizer.step()
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize()
                training_seconds += time.perf_counter() - started
                updates += 1
                append_csv(out / 'updates.csv', {
                    'update': updates, 'episode': completed, 'samples': int(valid.sum()),
                    'policy_loss': metrics['policy_loss'], 'value_loss': metrics['value_loss'],
                    'entropy': metrics['entropy'], 'loss': float(loss.detach().cpu()),
                    'gradient_norm': float(gradient), 'learner_transitions': transitions,
                    'environment_plies': plies, 'training_seconds': training_seconds,
                })
                for row in finished:
                    append_csv(out / 'training.csv', {
                        **row, 'steps': transitions, 'environment_plies': plies,
                        'updates': updates, 'seconds': training_seconds + validation_seconds,
                        'training_seconds': training_seconds, 'loss': float(loss.detach().cpu()),
                    })
            if completed != boundary or allocated != boundary:
                raise RuntimeError('Barrier did not drain exactly the assigned episodes')
            if completed % args.history_every == 0:
                history_pool.append({key: value.detach().cpu().clone() for key, value in net.state_dict().items()})
                history_pool = history_pool[-4:]
                history_networks.append(copy.deepcopy(net).eval())
                history_networks = history_networks[-4:]
            if completed % args.eval_every == 0 or (completed == args.episodes and best_episode is None):
                started = time.perf_counter()
                net.eval()
                rows = evaluate(agent, args.validation_games, 300000, args.rules,
                                out / f'validation_{completed}_games.csv')
                net.train()
                validation_seconds += time.perf_counter() - started
                score = sum(row['score'] for row in rows if row['opponent'] == 'greedy') / 2
                for row in rows:
                    append_csv(out / 'validation.csv', {'episode': completed, **row})
                if score > best:
                    best, best_episode = score, completed
                    agent.save(out / 'best.pt')
                ratio = curriculum_ratio(ratio, rows)
            save()
            print(f'A2C seed={args.seed} {completed}/{args.episodes} transitions={transitions} '
                  f'greedy score={best:.4f} ratio={ratio:.2f} train_seconds={training_seconds:.1f}', flush=True)
        agent.save(out / 'final.pt')
        agent.load(out / 'best.pt')
        net = agent.net
        net.eval()
        atomic_json(out / 'status.json', {'state': 'evaluating', 'episode': completed,
                                        'target_episodes': args.episodes, 'device': str(device),
                                        'pid': os.getpid()})
        atomic_json(out / 'final_evaluation_model.json', {
            'checkpoint': 'best.pt', 'selection': 'validation greedy score only',
            'selected_episode': best_episode, 'test_seed': 800000,
        })
        started = time.perf_counter()
        rows = evaluate(agent, args.final_games, 800000, args.rules, out / 'final_games.csv')
        test_seconds += time.perf_counter() - started
        atomic_json(out / 'final_evaluation.json', rows)
        latency = measure_latency(agent, args.rules)
        atomic_json(out / 'inference_latency.json', latency)
        atomic_json(out / 'status.json', {
            'state': 'complete', 'episode': completed, 'target_episodes': args.episodes,
            'pid': os.getpid(),
            'device': str(device), 'steps': transitions, 'environment_plies': plies,
            'learner_transitions': transitions, 'actual_rollout_samples': transitions,
            'updates': updates, 'best_episode': best_episode, 'validation_score': best,
            'training_seconds': training_seconds, 'validation_seconds': validation_seconds,
            'test_seconds': test_seconds, 'opponent_counts': opponent_counts,
            'inference_latency_ms': latency['median_ms'],
            'wall_seconds': previous_wall_seconds + time.perf_counter() - wall_started,
            'opponent_counts_by_kind_and_color': opponent_counts,
        })
    except BaseException as error:
        atomic_json(out / 'status.json', {
            'state': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            'pid': os.getpid(),
            'episode': saved_episode, 'target_episodes': args.episodes,
            'attempted_completed_episodes': completed, 'resume_from': 'latest.checkpoint',
            'error_type': type(error).__name__, 'message': str(error),
        })
        raise


def main():
    args = parser().parse_args()
    with exclusive_run(Path(args.output) / 'run.lock'):
        run(args)


if __name__ == '__main__':
    main()
