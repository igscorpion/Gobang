"""Publish completed A2C/DQN controls; validation alone selects model candidates."""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (11, 22, 33)
CELLS = tuple((opponent, color) for opponent in ("random", "greedy") for color in (1, -1))
COST_FIELDS = (
    "environment_plies", "learner_transitions", "updates", "gradient_updates",
    "actual_rollout_samples", "training_seconds", "validation_seconds", "test_seconds",
    "inference_latency_ms", "parameters", "parameter_count", "opponent_counts_by_kind_and_color",
)


@contextmanager
def report_lock(path):
    """Serialize independent publishers without taking the long-lived suite lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + 5
        while True:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Another A2C report publisher holds the report lock")
                time.sleep(0.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_atomic(path, text):
    """Unique temporary names also make replacement safe after interrupted reports."""
    path = Path(path)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def selected_validation(directory):
    """Earliest maximum of paired Greedy scores; initial policies are excluded."""
    groups = {}
    with (directory / "validation.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode"])
            if row["opponent"] != "greedy" or episode <= 0:
                continue
            color = int(row["color"])
            score = float(row["score"])
            if color not in (1, -1) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid validation Greedy score/color")
            colors = groups.setdefault(episode, {})
            if color in colors:
                raise ValueError("Duplicate validation color in one episode")
            colors[color] = score
    scores = {episode: statistics.mean(values.values()) for episode, values in groups.items() if len(values) == 2}
    if not scores:
        raise ValueError("No complete positive-episode Greedy validation pair")
    best = max(scores.values())
    return min(episode for episode, score in scores.items() if score == best), best


def normalize_test(rows):
    """Recompute rates and Wilson intervals from independent test counts."""
    normalized = {}
    for row in rows:
        key = (row["opponent"], int(row["color"]))
        if key not in CELLS or key in normalized:
            raise ValueError("Expected one test row per opponent and color")
        games, wins, draws, losses = (int(row[field]) for field in ("games", "wins", "draws", "losses"))
        if games != 100 or min(wins, draws, losses) < 0 or wins + draws + losses != games:
            raise ValueError("Expected 100 games and consistent W/D/L per test cell")
        rate, z = wins / games, 1.96
        center = (rate + z * z / (2 * games)) / (1 + z * z / games)
        radius = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games)) / (1 + z * z / games)
        normalized[key] = {
            "opponent": key[0], "color": key[1], "games": games,
            "wins": wins, "draws": draws, "losses": losses,
            "win_rate": rate, "score": (wins + 0.5 * draws) / games,
            "wilson_low": max(0.0, center - radius), "wilson_high": min(1.0, center + radius),
            "rules": row.get("rules", "freestyle"),
        }
    if set(normalized) != set(CELLS):
        raise ValueError("Final test missing opponent/color cells")
    return [normalized[key] for key in CELLS]


def last_csv_row(path):
    last = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                last = row
    return last


def collect_run(out, family, seed):
    name = f"{family}_seed{seed}"
    directory = out / name
    item = {"name": name, "family": family, "seed": seed, "state": "pending", "issues": []}
    try:
        status = read_json(directory / "status.json", {})
        item.update({"state": status.get("state", "pending"), "episode": status.get("episode"), "target_episodes": status.get("target_episodes", 30000)})
        config = read_json(directory / "config.json", {})
        item["config"] = config
        # Costs are optional; missing metrics stay absent rather than becoming zero.
        item["cost_metrics"] = {key: source[key] for source in (config, status) for key in COST_FIELDS if key in source}
        if item["state"] != "complete":
            return item
        metadata = read_json(directory / "final_evaluation_model.json", {})
        if metadata.get("checkpoint") != "best.pt" or metadata.get("test_seed") != 800000:
            raise ValueError("Final evaluation must identify validation-selected best.pt and independent test_seed=800000")
        if "validation" not in metadata.get("selection", "").lower():
            raise ValueError("Final evaluation model selection must use validation")
        item["best_episode"], item["validation_score"] = selected_validation(directory)
        checkpoint = directory / "best.pt"
        if not checkpoint.exists():
            raise ValueError("Selected best.pt is missing")
        item["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        item["evaluation_model"] = metadata
        item["test"] = normalize_test(read_json(directory / "final_evaluation.json", []))
        for source in (last_csv_row(directory / "training.csv"), last_csv_row(directory / "updates.csv")):
            for key in COST_FIELDS:
                if key in source and key not in item["cost_metrics"]:
                    try:
                        value = float(source[key])
                        if math.isfinite(value):
                            item["cost_metrics"][key] = value
                    except (ValueError, TypeError):
                        pass
    except (OSError, ValueError, TypeError, KeyError) as exc:
        item.pop("test", None)
        item["issues"].append(str(exc))
    return item


def sample_summary(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None, "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def summarize(experiments):
    summaries, pairs = [], []
    completed = {(item["family"], item["seed"]): item for item in experiments if "test" in item}
    for family in ("a2c4", "dqn4"):
        for opponent, color in CELLS:
            matching = [next(row for row in item["test"] if (row["opponent"], row["color"]) == (opponent, color)) for (f, _), item in completed.items() if f == family]
            summaries.append({"family": family, "opponent": opponent, "color": color, "win_rate": sample_summary([row["win_rate"] for row in matching]), "score": sample_summary([row["score"] for row in matching])})
    paired_seeds = [seed for seed in SEEDS if all((family, seed) in completed for family in ("a2c4", "dqn4"))]
    for opponent, color in CELLS:
        rows = []
        for seed in paired_seeds:
            a2c, dqn = (next(row for row in completed[(family, seed)]["test"] if (row["opponent"], row["color"]) == (opponent, color)) for family in ("a2c4", "dqn4"))
            rows.append({"seed": seed, "win_rate_difference_percentage_points": 100 * (a2c["win_rate"] - dqn["win_rate"]), "score_difference": a2c["score"] - dqn["score"]})
        pairs.append({"opponent": opponent, "color": color, "pairs": rows, "win_rate_difference_percentage_points": sample_summary([row["win_rate_difference_percentage_points"] for row in rows]), "score_difference": sample_summary([row["score_difference"] for row in rows])})
    return summaries, pairs


def format_summary(summary, multiplier=100, suffix="%", decimals=1):
    if not summary["n"]:
        return "—"
    value = f"{summary['mean'] * multiplier:.{decimals}f}{suffix}"
    return value + (f" ± {summary['sample_sd'] * multiplier:.{decimals}f}{suffix}" if summary["sample_sd"] is not None else "（单种子）")


def publish(out=ROOT / "runs/production", root=ROOT):
    out, root = Path(out), Path(root)
    with report_lock(out / "a2c_report.lock"):
        experiments = [collect_run(out, family, seed) for seed in SEEDS for family in ("a2c4", "dqn4")]
        summaries, pairs = summarize(experiments)
        candidates = [item for item in experiments if item["family"] == "a2c4" and "test" in item]
        # Stable seed order breaks ties; neither test wins nor test scores are read here.
        selected = max(candidates, key=lambda item: item["validation_score"], default=None)
        models = {}
        if selected:
            checkpoint = out / selected["name"] / "best.pt"
            try:
                checkpoint_label = checkpoint.relative_to(root).as_posix()
            except ValueError:
                checkpoint_label = str(checkpoint.resolve())
            models["a2c4"] = {"algorithm": "a2c", "checkpoint": checkpoint_label, "selection": "validation score only", "validation_score": selected["validation_score"], "seed": selected["seed"], "selected_episode": selected["best_episode"]}
        payload = {"experiment": "a2c4_vs_dqn4", "a2c_complete": all("test" in item for item in experiments if item["family"] == "a2c4"), "experiments": experiments, "across_seeds": summaries, "paired_a2c_minus_dqn": pairs, "model_selection": "Completed best.pt candidates ranked by mean validation Greedy score only; earliest validation checkpoint and stable seed order break ties.", "limitations": ["Same game budget, observation, backbone and curriculum mechanism; realized opponent ratios and historical policies can differ.", "A2C and DQN have different learner transitions, optimization, replay and compute costs; this is not strict compute-budget matching.", "Three seeds provide limited replication; Wilson intervals describe per-seed test win rates, not across-training-seed uncertainty.", "Initial validation is never reported as final test and does not select the A2C checkpoint."]}
        lines = ["# A2C4 与 DQN4 对照", "", "原方案：4 通道，32/64/64 卷积＋FC256；每种方法每个种子训练 30,000 局，种子 11/22/33。", "A2C 同步 8 环境、5-step；测试使用验证选出的 best.pt，对 Random / Greedy 分黑白各 100 局。", "只展示已完成且最终测试资料完整的结果；第 0 局或训练中验证不会冒充最终成绩。", "", "| 阶段 | 状态 / 进度 | 验证选中局数 | Greedy 验证平均 score |", "|---|---|---:|---:|"]
        for item in experiments:
            phase = item["state"] + (f" {item['episode']} / {item['target_episodes']}" if item.get("episode") is not None else "")
            if item["issues"]:
                phase += "；结果未采纳：" + "; ".join(item["issues"])
            lines.append(f"| {item['name']} | {phase} | {item.get('best_episode', '—')} | {format(item['validation_score'], '.3f') if 'test' in item else '—'} |")
        lines += ["", "## 独立最终测试", "", "| 阶段 | 对手 / 执子 | 胜 / 和 / 负 | 胜率（95% Wilson 区间） | score |", "|---|---|---|---|---:|"]
        for item in experiments:
            for row in item.get("test", []):
                lines.append(f"| {item['name']} | {row['opponent']} / {'黑' if row['color'] == 1 else '白'} | {row['wins']} / {row['draws']} / {row['losses']} | {100 * row['win_rate']:.1f}% ({100 * row['wilson_low']:.1f}–{100 * row['wilson_high']:.1f}%) | {row['score']:.3f} |")
        lines += ["", "## 跨种子均值与样本标准差", "", "| 方法 | 对手 / 执子 | 已完成种子 | 胜率均值 ± 样本标准差 | score 均值 ± 样本标准差 |", "|---|---|---:|---|---|"]
        for row in summaries:
            lines.append(f"| {row['family']} | {row['opponent']} / {'黑' if row['color'] == 1 else '白'} | {row['win_rate']['n']} / 3 | {format_summary(row['win_rate'])} | {format_summary(row['score'], 1, '', 3)} |")
        lines += ["", "## 同种子 A2C 减 DQN", "", "| 对手 / 执子 | 已配对种子 | 胜率差均值 ± 样本标准差（百分点） | score 差均值 ± 样本标准差 |", "|---|---:|---|---|"]
        for row in pairs:
            lines.append(f"| {row['opponent']} / {'黑' if row['color'] == 1 else '白'} | {len(row['pairs'])} / 3 | {format_summary(row['win_rate_difference_percentage_points'], 1, '')} | {format_summary(row['score_difference'], 1, '', 3)} |")
        lines += ["", "## 可用计算成本指标", "", "缺失指标保留为空，不能根据参数量推定训练成本相同。JSON 保留已记录的成本指标。", "", "| 阶段 | 参数量 | 环境落子数 | 学习方转移数 | 梯度更新数 | 训练 / 验证 / 测试秒数 |", "|---|---:|---:|---:|---:|---|"]
        for item in experiments:
            cost = item.get("cost_metrics", {})
            lines.append(f"| {item['name']} | {cost.get('parameters', cost.get('parameter_count', '—'))} | {cost.get('environment_plies', '—')} | {cost.get('learner_transitions', '—')} | {cost.get('gradient_updates', cost.get('updates', '—'))} | {' / '.join(str(cost.get(key, '—')) for key in ('training_seconds', 'validation_seconds', 'test_seconds'))} |")
        lines += ["", "## 口径与限制", "", "score=胜率+0.5×和率；跨种子标准差使用样本标准差（n−1），单种子不填标准差。", "按已完成模型的 Greedy 黑白验证平均 score 选择网页候选，测试成绩不参与 checkpoint 或种子选择。", "相同输入、近似容量、局数和课程机制；实际对手比例、历史策略、更新次数及计算成本可能不同，不称严格计算公平。", "三种子仍属有限重复；每格 Wilson 区间描述单个冻结模型的测试胜率，不代表跨训练种子的泛化区间。", "跨方法同种子差采用配对汇总，测试开局种子相同也不能当成训练轨迹相同。"]
        if selected:
            lines += ["", f"当前 A2C4 候选：{selected['name']} 的第 {selected['best_episode']} 局 best.pt，验证 score={selected['validation_score']:.3f}。"]
        for filename, value in (("a2c_dqn_comparison.json", payload), ("a2c_models.json", models)):
            write_atomic(out / filename, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        write_atomic(out / "a2c_dqn_comparison.md", "\n".join(lines) + "\n")
        return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/production")
    args = parser.parse_args()
    result = publish(args.output)
    print(json.dumps({"a2c_complete": result["a2c_complete"], "report": str(args.output / "a2c_dqn_comparison.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
