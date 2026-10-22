"""Shared experiment IO, seeded evaluation and validation-only curriculum."""

from pathlib import Path
from contextlib import contextmanager
import json, csv, random
import numpy as np
import torch
from rl.environment.gomoku import GomokuEnv
from rl.environment.state import observation
from rl.agents.greedy_agent import GreedyAgent
from rl.agents.random_agent import RandomAgent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def atomic_torch(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def append_csv(path, row):
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def prepare_resume_logs(out, episode):
    """Discard log rows beyond the last durable checkpoint after a stopped process."""
    for name in ("training.csv", "validation.csv"):
        path = Path(out) / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = [r for r in reader if int(r["episode"]) <= episode]
        with path.open("w", newline="", encoding="utf-8") as f:
            if fields:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


def check_resume_config(saved, current, keys):
    for key in keys:
        if key in saved and saved[key] != current[key]:
            raise ValueError(
                f"Cannot resume with different {key}: {saved[key]} -> {current[key]}"
            )


@contextmanager
def exclusive_run(path):
    """OS-held lock released even after a crashed launcher; no stale PID lock."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if path.stat().st_size == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "This experiment suite is already running; do not launch a second writer."
            ) from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def choose(agent, env):
    return agent.select_action(
        observation(env) if agent.needs_last_move else env.get_state(),
        env.get_legal_actions(),
    )


def evaluate(agent, games=40, seed=300000, rules="freestyle", log_path=None):
    """Frozen per-color/per-opponent evaluation preserving training RNGs."""
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    trng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    agent_rng = agent.rng.getstate() if hasattr(agent, "rng") else None
    epsilon = getattr(agent, "epsilon", None)
    policy_training = getattr(agent, "training", None)
    network = getattr(agent, "net", None)
    network_training = getattr(network, "training", None)
    if epsilon is not None:
        agent.epsilon = 0.0
    if policy_training is not None:
        agent.training = False
    if network_training is not None:
        network.eval()
    rows = []
    if log_path:
        Path(log_path).write_text("", encoding="utf-8")
    try:
        for name in ("random", "greedy"):
            for color in (1, -1):
                results = []
                for game in range(games):
                    gs = seed + game + (100000 if color == -1 else 0)
                    random.seed(gs)
                    torch.manual_seed(gs)
                    if agent_rng is not None:
                        agent.rng.seed(gs + 500000)
                    opponent = (RandomAgent if name == "random" else GreedyAgent)(
                        seed=gs
                    )
                    env = GomokuEnv(rules=rules)
                    moves = 0
                    if hasattr(agent, "mcts"):
                        agent.mcts.root = None
                    while not env.done:
                        env.step(
                            choose(
                                agent if env.current_player == color else opponent, env
                            )
                        )
                        moves += 1
                    outcome = (
                        0 if env.winner == 0 else (1 if env.winner == color else -1)
                    )
                    results.append(outcome)
                    if log_path:
                        append_csv(
                            log_path,
                            {
                                "opponent": name,
                                "color": color,
                                "game": game,
                                "seed": gs,
                                "outcome": outcome,
                                "moves": moves,
                                "rules": rules,
                            },
                        )
                wins = results.count(1)
                draws = results.count(0)
                p = wins / games
                z = 1.96
                center = (p + z * z / (2 * games)) / (1 + z * z / games)
                radius = (
                    z
                    * (p * (1 - p) / games + z * z / (4 * games * games)) ** 0.5
                    / (1 + z * z / games)
                )
                rows.append(
                    {
                        "opponent": name,
                        "color": color,
                        "games": games,
                        "wins": wins,
                        "draws": draws,
                        "losses": games - wins - draws,
                        "win_rate": p,
                        "score": (wins + 0.5 * draws) / games,
                        "wilson_low": center - radius,
                        "wilson_high": center + radius,
                        "rules": rules,
                    }
                )
    finally:
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(trng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if agent_rng is not None:
            agent.rng.setstate(agent_rng)
        if epsilon is not None:
            agent.epsilon = epsilon
        if policy_training is not None:
            agent.training = policy_training
        if network_training is not None:
            network.train(network_training)
        if hasattr(agent, "mcts"):
            agent.mcts.root = None
    return rows


def curriculum_ratio(current, rows, short_fraction=None):
    greedy = np.mean([r["win_rate"] for r in rows if r["opponent"] == "greedy"])
    target = (
        0.6
        if greedy >= 0.65
        else (0.4 if greedy >= 0.4 else (0.2 if greedy >= 0.15 else 0.1))
    )
    if short_fraction is not None and short_fraction >= 0.8:
        target = max(target, 0.4)
    return float(np.clip(target, current - 0.1, current + 0.1))


def rng_checkpoint():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
