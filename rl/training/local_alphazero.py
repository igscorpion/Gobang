"""Architecture ablation, explicit warm transfer and resumable AlphaZero training."""

from pathlib import Path
from collections import deque
import copy, argparse, random, os, time, hashlib
import torch
from rl.agents.alphazero import AlphaZeroAgent, AlphaZeroNetwork
from rl.agents.greedy_agent import GreedyAgent
from rl.training.az_games import (
    selfplay_game,
    vs_opponent_game,
    _batch_tensor,
    _policy_probs,
    train_step,
)
from rl.training.common import (
    seed_everything,
    atomic_json,
    atomic_torch,
    append_csv,
    evaluate,
    curriculum_ratio,
    rng_checkpoint,
    restore_rng,
    prepare_resume_logs,
    check_resume_config,
)


def transfer_weights(net, source):
    """Copy unchanged tensors; prefix-copy widened channels; retain random new slices."""
    old = torch.load(source, map_location="cpu", weights_only=True)
    old = old.get("state_dict", old)
    new = net.state_dict()
    records = []
    for name, value in new.items():
        if name not in old:
            raise ValueError(f"Missing tensor: {name}")
        previous = old[name]
        if value.shape == previous.shape:
            value.copy_(previous)
            mode = "exact"
        elif len(value.shape) == len(previous.shape) and all(
            a >= b for a, b in zip(value.shape, previous.shape)
        ):
            value[tuple(slice(0, n) for n in previous.shape)].copy_(previous)
            mode = "prefix"
        else:
            raise ValueError(f"Incompatible transfer: {name}")
        records.append(
            {
                "tensor": name,
                "source_shape": list(previous.shape),
                "new_shape": list(value.shape),
                "mode": mode,
            }
        )
    net.load_state_dict(new)
    return {
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "tensors": records,
        "optimizer": "fresh Adam, no state imported",
        "new_slices": "random initialization",
    }


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["GOMOKU_RULES"] = args.rules
    seed_everything(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = AlphaZeroNetwork(channels=tuple(args.channels)).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    buffer = deque(maxlen=40000)
    ckpath = out / "latest.checkpoint"
    episode = 0
    best = -1.0
    multiplier = 1.0
    ratio = args.fixed_ratio if args.fixed_ratio is not None else 0.1
    checkpoint = None
    if args.resume and ckpath.exists():
        checkpoint = torch.load(ckpath, map_location="cpu", weights_only=False)
        if (
            checkpoint["channels"] != tuple(args.channels)
            or checkpoint["rules"] != args.rules
        ):
            raise ValueError("Checkpoint configuration differs")
        check_resume_config(
            checkpoint.get("config", {}),
            vars(args),
            (
                "seed",
                "sims",
                "fixed_ratio",
                "batch_size",
                "eval_every",
                "validation_games",
            ),
        )
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        episode = checkpoint["episode"]
        best = checkpoint["best"]
        ratio = checkpoint["ratio"]
        multiplier = checkpoint["multiplier"]
        buffer.extend(checkpoint["buffer"])
        prepare_resume_logs(out, episode)
    elif args.warm_start:
        atomic_json(out / "transfer.json", transfer_weights(net, args.warm_start))
    ema = copy.deepcopy(net)
    for p in ema.parameters():
        p.requires_grad_(False)
    if checkpoint:
        ema.load_state_dict(checkpoint["ema"])
        restore_rng(checkpoint["rng"])
    greedy = GreedyAgent(seed=args.seed)
    lengths = []
    started = time.perf_counter()
    if checkpoint:
        lengths = checkpoint.get("lengths", [])
        if "greedy_rng" in checkpoint:
            greedy.rng.setstate(checkpoint["greedy_rng"])
    atomic_json(
        out / "config.json",
        {
            **vars(args),
            "parameters": sum(p.numel() for p in net.parameters()),
            "device_used": device,
        },
    )

    def weights():
        return {"state_dict": ema.state_dict(), "channels": net.channels}

    def save():
        atomic_torch(out / "current.pt", weights())
        atomic_torch(
            ckpath,
            {
                "net": net.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "buffer": list(buffer),
                "episode": episode,
                "ratio": ratio,
                "best": best,
                "multiplier": multiplier,
                "rng": rng_checkpoint(),
                "channels": tuple(args.channels),
                "rules": args.rules,
                "config": vars(args),
                "greedy_rng": greedy.rng.getstate(),
                "lengths": lengths,
            },
        )
        atomic_json(
            out / "status.json",
            {
                "state": "training",
                "episode": episode,
                "target_episodes": args.episodes,
                "device": device,
                "ratio": ratio,
                "validation_score": best,
                "sims": args.sims,
            },
        )

    def eval_agent():
        rng = rng_checkpoint()
        agent = AlphaZeroAgent(
            channels=tuple(args.channels),
            device=device,
            num_sims=args.sims,
            rules=args.rules,
            seed=args.seed,
        )
        agent.net.load_state_dict(ema.state_dict())
        agent.net.eval()
        restore_rng(rng)
        return agent

    if not ckpath.exists():
        save()
    try:
        if episode == 0 and best < 0:
            rows = evaluate(
                eval_agent(),
                args.validation_games,
                300000,
                args.rules,
                out / "validation_0_games.csv",
            )
            for row in rows:
                append_csv(out / "validation.csv", {"episode": 0, **row})
            best = sum(r["score"] for r in rows if r["opponent"] == "greedy") / 2
            atomic_torch(out / "best.pt", weights())
            atomic_json(out / "initial_validation.json", rows)
            save()
        while episode < args.episodes:
            episode += 1
            t = time.perf_counter()
            if random.random() < ratio:
                kind = "greedy"
                data, winner, color = vs_opponent_game(
                    net, device, 9, args.sims, 1.4, 1.0, greedy
                )
            else:
                kind = "selfplay"
                data, winner = selfplay_game(net, device, 9, args.sims, 1.4, 1.0)
                color = 0
                lengths.append(len(data) // 8)
            buffer.extend(data)
            p_loss = v_loss = kl = float("nan")
            lr = 0.0005 * 0.5 ** (episode // 2000) * multiplier
            if len(buffer) >= args.batch_size:
                batch = random.sample(list(buffer), args.batch_size)
                xs = _batch_tensor(batch, device, 9)
                old = _policy_probs(net, xs)
                for _ in range(5):
                    p_loss, v_loss = train_step(net, optimizer, batch, device, 9, lr)
                    with torch.no_grad():
                        for ep, p in zip(ema.parameters(), net.parameters()):
                            ep.mul_(0.999).add_(p, alpha=0.001)
                    new = _policy_probs(net, xs)
                    kl = float(
                        (old * ((old + 1e-10).log() - (new + 1e-10).log()))
                        .sum(1)
                        .mean()
                    )
                    if kl > 0.08:
                        break
                if kl > 0.04:
                    multiplier = max(0.1, multiplier / 1.5)
                elif kl < 0.01:
                    multiplier = min(10.0, multiplier * 1.5)
            append_csv(
                out / "training.csv",
                {
                    "episode": episode,
                    "kind": kind,
                    "color": color,
                    "winner": winner,
                    "recorded_positions": len(data) // 8,
                    "greedy_ratio": ratio,
                    "policy_loss": p_loss,
                    "value_loss": v_loss,
                    "kl": kl,
                    "lr": lr,
                    "seconds": time.perf_counter() - t,
                    "elapsed": time.perf_counter() - started,
                },
            )
            print(
                f"AZ channels={args.channels} {episode}/{args.episodes} {kind} {len(data) // 8} samples steps {time.perf_counter() - t:.1f}s",
                flush=True,
            )
            if episode % args.eval_every == 0:
                agent = eval_agent()
                rows = evaluate(
                    agent,
                    args.validation_games,
                    300000,
                    args.rules,
                    out / f"validation_{episode}_games.csv",
                )
                for r in rows:
                    append_csv(out / "validation.csv", {"episode": episode, **r})
                score = sum(r["score"] for r in rows if r["opponent"] == "greedy") / 2
                if score > best:
                    best = score
                    atomic_torch(out / "best.pt", weights())
                short = (
                    sum(n <= 11 for n in lengths) / len(lengths) if lengths else None
                )
                if args.fixed_ratio is None:
                    ratio = curriculum_ratio(ratio, rows, short)
                atomic_json(
                    out / f"health_{episode}.json",
                    {
                        "selfplay_lengths": lengths,
                        "short_fraction": short,
                        "new_ratio": ratio,
                    },
                )
                lengths = []
            if episode % args.save_every == 0:
                save()
        save()
        atomic_torch(out / "final.pt", weights())
        test_agent = eval_agent()
        # Freeze the validation-selected model before evaluating the held-out seeds.
        if (out / "best.pt").exists():
            test_agent.load(out / "best.pt")
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
            test_agent, args.final_games, 800000, args.rules, out / "final_games.csv"
        )
        atomic_json(out / "final_evaluation.json", rows)
        atomic_json(
            out / "status.json",
            {"state": "complete", "episode": episode, "device": device},
        )
    except KeyboardInterrupt:
        # A half-played game is not a completed episode. Keep the durable checkpoint.
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
    p.add_argument("--output", default="runs/az_wide")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--channels", type=int, nargs=3, default=[64, 64, 128])
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--fixed-ratio", type=float)
    p.add_argument("--warm-start")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rules", choices=["freestyle", "renju"], default="freestyle")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--validation-games", type=int, default=40)
    p.add_argument("--final-games", type=int, default=100)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
