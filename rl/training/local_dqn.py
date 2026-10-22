"""Masked Double DQN with mixed transition perspectives and resumable runs."""

from pathlib import Path
from collections import deque
import argparse, random, time, os
import torch
from rl.agents.dqn import DQNAgent, QNetwork, state_to_tensor
from rl.agents.greedy_agent import GreedyAgent
from rl.agents.random_agent import RandomAgent
from rl.environment.gomoku import GomokuEnv
from rl.environment.state import observation
from rl.training.common import (
    seed_everything,
    atomic_json,
    atomic_torch,
    append_csv,
    choose,
    evaluate,
    curriculum_ratio,
    rng_checkpoint,
    restore_rng,
    prepare_resume_logs,
    check_resume_config,
)


def double_dqn_target(online, target, next_x, mask, rewards, dones, signs, gamma=0.99):
    with torch.no_grad():
        scores = online(next_x).masked_fill(~mask, float("-inf"))
        actions = scores.argmax(1)
        values = target(next_x).gather(1, actions[:, None]).squeeze(1)
        bootstrap = torch.where(mask.any(1) & ~dones, values, torch.zeros_like(values))
        return rewards + gamma * signs * bootstrap


def train_batch(net, target, optimizer, batch, device):
    x, actions, rewards, nx, done, mask, sign = zip(*batch)
    x = torch.stack(x).to(device)
    nx = torch.stack(nx).to(device)
    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(rewards, device=device)
    done = torch.tensor(done, device=device)
    mask = torch.stack(mask).to(device)
    sign = torch.tensor(sign, device=device)
    values = net(x).gather(1, actions[:, None]).squeeze(1)
    targets = double_dqn_target(net, target, nx, mask, rewards, done, sign)
    loss = torch.nn.functional.smooth_l1_loss(values, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["GOMOKU_RULES"] = args.rules
    seed_everything(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device=device, seed=args.seed, input_channels=args.input_channels)
    net = agent.net
    target = QNetwork(input_channels=args.input_channels).to(device)
    target.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    replay = deque(maxlen=args.replay_capacity)
    history_pool = []
    episode = steps = updates = 0
    best = -1.0
    ratio = args.fixed_ratio if args.fixed_ratio is not None else 0.1
    checkpoint = out / "latest.checkpoint"
    config = vars(args).copy()
    atomic_json(out / "config.json", config)
    if args.resume and checkpoint.exists():
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ck["rules"] != args.rules or ck["input_channels"] != args.input_channels:
            raise ValueError("Cannot resume a different rule or architecture")
        check_resume_config(
            ck.get("config", {}),
            vars(args),
            (
                "seed",
                "fixed_ratio",
                "epsilon_decay",
                "replay_capacity",
                "batch_size",
                "eval_every",
                "validation_games",
            ),
        )
        net.load_state_dict(ck["net"])
        target.load_state_dict(ck["target"])
        optimizer.load_state_dict(ck["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        replay.extend(ck["replay"])
        history_pool = ck["history_pool"]
        episode = ck["episode"]
        steps = ck["steps"]
        updates = ck["updates"]
        ratio = ck["ratio"]
        best = ck["best"]
        agent.rng.setstate(ck["agent_rng"])
        restore_rng(ck["rng"])
        prepare_resume_logs(out, episode)
    elif args.warm_start:
        agent.load(args.warm_start)
        if agent.net.input_channels != args.input_channels:
            raise ValueError("DQN channel change needs fresh training")
        net = agent.net
        target.load_state_dict(net.state_dict())
        optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    started = time.perf_counter()

    def save():
        agent.save(out / "current.pt")
        atomic_torch(
            checkpoint,
            {
                "net": net.state_dict(),
                "target": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "replay": list(replay),
                "history_pool": history_pool,
                "episode": episode,
                "steps": steps,
                "updates": updates,
                "ratio": ratio,
                "best": best,
                "agent_rng": agent.rng.getstate(),
                "rng": rng_checkpoint(),
                "rules": args.rules,
                "input_channels": args.input_channels,
                "config": vars(args),
            },
        )
        atomic_json(
            out / "status.json",
            {
                "state": "training",
                "episode": episode,
                "target_episodes": args.episodes,
                "steps": steps,
                "ratio": ratio,
                "validation_score": best,
                "device": device,
            },
        )

    def remember(state, action, reward, nxt, done, legal, sign):
        nonlocal steps, updates
        mask = torch.zeros(81, dtype=torch.bool)
        mask[list(legal)] = True
        replay.append(
            (
                state_to_tensor(state, input_channels=args.input_channels),
                action,
                float(reward),
                state_to_tensor(nxt, input_channels=args.input_channels),
                bool(done),
                mask,
                float(sign),
            )
        )
        steps += 1
        if len(replay) >= args.batch_size * 4 and steps % 4 == 0:
            loss = train_batch(
                net,
                target,
                optimizer,
                random.sample(list(replay), args.batch_size),
                device,
            )
            updates += 1
            if updates % 200 == 0:
                target.load_state_dict(net.state_dict())
            return loss
        return float("nan")

    if not checkpoint.exists():
        save()
    try:
        while episode < args.episodes:
            episode += 1
            env = GomokuEnv(rules=args.rules)
            moves = 0
            loss = float("nan")
            agent.epsilon = max(0.05, 1.0 - 0.95 * episode / max(args.epsilon_decay, 1))
            sample = random.random()
            if sample < ratio:
                kind = "greedy"
                opponent = GreedyAgent(seed=random.randrange(2**31))
            elif sample < ratio + 0.15:
                kind = "random"
                opponent = RandomAgent(seed=random.randrange(2**31))
            elif history_pool and sample < ratio + 0.3:
                kind = "history"
                opponent = DQNAgent(
                    epsilon=0.05,
                    device=device,
                    input_channels=args.input_channels,
                    seed=random.randrange(2**31),
                )
                opponent.net.load_state_dict(random.choice(history_pool))
            else:
                kind = "selfplay"
                opponent = None
            color = 1 if episode % 2 else -1
            if opponent is None:
                while not env.done:
                    state = observation(env)
                    action = choose(agent, env)
                    _, reward, done = env.step(action)
                    moves += 1
                    loss = remember(
                        state,
                        action,
                        reward,
                        observation(env),
                        done,
                        env.get_legal_actions(),
                        -1.0,
                    )
            else:
                if color == -1:
                    env.step(choose(opponent, env))
                    moves += 1
                while not env.done:
                    state = observation(env)
                    action = choose(agent, env)
                    env.step(action)
                    moves += 1
                    if not env.done:
                        env.step(choose(opponent, env))
                        moves += 1
                    reward = (
                        0.0
                        if not env.done or env.winner == 0
                        else (1.0 if env.winner == color else -1.0)
                    )
                    loss = remember(
                        state,
                        action,
                        reward,
                        observation(env),
                        env.done,
                        env.get_legal_actions(),
                        1.0,
                    )
            append_csv(
                out / "training.csv",
                {
                    "episode": episode,
                    "kind": kind,
                    "color": color,
                    "winner": env.winner,
                    "moves": moves,
                    "epsilon": agent.epsilon,
                    "greedy_ratio": ratio,
                    "loss": loss,
                    "steps": steps,
                    "seconds": time.perf_counter() - started,
                },
            )
            if episode % 1000 == 0:
                history_pool.append(
                    {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                )
                history_pool = history_pool[-4:]
            if episode % args.eval_every == 0:
                rows = evaluate(
                    agent,
                    args.validation_games,
                    300000,
                    args.rules,
                    out / f"validation_{episode}_games.csv",
                )
                score = sum(r["score"] for r in rows if r["opponent"] == "greedy") / 2
                for r in rows:
                    append_csv(out / "validation.csv", {"episode": episode, **r})
                if score > best:
                    best = score
                    agent.save(out / "best.pt")
                if args.fixed_ratio is None:
                    ratio = curriculum_ratio(ratio, rows)
                print(
                    f"DQN seed={args.seed} channels={args.input_channels} {episode}/{args.episodes} greedy score={score:.3f} ratio={ratio:.2f}",
                    flush=True,
                )
            if episode % args.save_every == 0:
                save()
        save()
        agent.save(out / "final.pt")
        # Select with validation only; final tests do not choose a checkpoint.
        if (out / "best.pt").exists():
            agent.load(out / "best.pt")
        atomic_json(
            out / "final_evaluation_model.json",
            {
                "checkpoint": "best.pt" if (out / "best.pt").exists() else "final.pt",
                "selection": "validation greedy score only"
                if (out / "best.pt").exists()
                else "no validation was run",
                "test_seed": 800000,
            },
        )
        rows = evaluate(
            agent, args.final_games, 800000, args.rules, out / "final_games.csv"
        )
        atomic_json(out / "final_evaluation.json", rows)
        atomic_json(
            out / "status.json",
            {"state": "complete", "episode": episode, "steps": steps, "device": device},
        )
    except KeyboardInterrupt:
        atomic_json(
            out / "status.json",
            {
                "state": "interrupted",
                "attempted_episode": episode,
                "resumable": True,
                "resume_from": "latest.checkpoint",
                "partial_game_discarded": True,
            },
        )
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="runs/dqn4")
    p.add_argument("--episodes", type=int, default=30000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--input-channels", type=int, choices=[3, 4], default=4)
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--rules", choices=["freestyle", "renju"], default="freestyle")
    p.add_argument("--fixed-ratio", type=float)
    p.add_argument("--epsilon-decay", type=int, default=15000)
    p.add_argument("--replay-capacity", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--validation-games", type=int, default=40)
    p.add_argument("--final-games", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--warm-start")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
