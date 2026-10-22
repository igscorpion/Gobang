"""Publish completed AlphaZero capacity results without using tests for selection."""

from pathlib import Path
import argparse
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
MARKER = "<!-- alphazero-capacity-results"


def write_if_changed(path, text):
    path = Path(path)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def selected_episode(directory):
    with (directory / "validation.csv").open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    groups = {}
    for row in records:
        if row["opponent"] == "greedy":
            groups.setdefault(int(row["episode"]), []).append(float(row["score"]))
    scores = {
        episode: sum(values) / 2
        for episode, values in groups.items()
        if len(values) == 2
    }
    # Trainers keep the first checkpoint attaining the highest validation score.
    best = max(scores.values())
    return min(episode for episode, value in scores.items() if value == best), best


def publish(out=ROOT / "runs/production", root=ROOT):
    out, root = Path(out), Path(root)
    records = []
    lines = [
        "# AlphaZero 网络容量与 Greedy 对照",
        "",
        "两组均新增训练 1,000 局，seed=11，400 次 MCTS 模拟，Greedy 对手比例固定 .6，规则为 freestyle。",
        "测试使用验证集选出的 best.pt；每色 100 局，测试种子 800000/900000，与验证种子分离。",
        "未完成阶段只显示状态，不把第 0 局验证成绩当成最终测试。",
        "",
        "| 实验 | 通道 / 参数量 | 状态 | Greedy 黑胜率 | Greedy 白胜率 |",
        "|---|---|---|---:|---:|",
    ]
    for width in (64, 128):
        name = f"az_width{width}_fixed"
        directory = out / name
        status_path = directory / "status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.exists()
            else {"state": "pending"}
        )
        item = {
            "name": name,
            "channels": [64, 64, width],
            "parameters": 94935 if width == 64 else 132055,
            "state": status["state"],
            "episode": status.get("episode"),
        }
        result_path = directory / "final_evaluation.json"
        rates = ["—", "—"]
        if status["state"] == "complete" and result_path.exists():
            item["config"] = json.loads(
                (directory / "config.json").read_text(encoding="utf-8")
            )
            item["evaluation_model"] = json.loads(
                (directory / "final_evaluation_model.json").read_text(encoding="utf-8")
            )
            item["best_episode"], item["validation_score"] = selected_episode(directory)
            item["checkpoint_sha256"] = hashlib.sha256(
                (directory / item["evaluation_model"]["checkpoint"]).read_bytes()
            ).hexdigest()
            results = json.loads(result_path.read_text(encoding="utf-8"))
            item["greedy"] = [
                next(
                    r
                    for r in results
                    if r["opponent"] == "greedy" and r["color"] == color
                )
                for color in (1, -1)
            ]
            rates = [f"{r['win_rate'] * 100:.1f}%" for r in item["greedy"]]
        label = status["state"] + (
            f" {status['episode']} / 1000" if "episode" in status else ""
        )
        lines.append(
            f"| {name} | 64/64/{width} · {item['parameters']:,} | {label} | {rates[0]} | {rates[1]} |"
        )
        records.append(item)

    complete = all("greedy" in item for item in records)
    if complete:
        for key in (
            "seed",
            "episodes",
            "sims",
            "fixed_ratio",
            "rules",
            "batch_size",
            "eval_every",
            "validation_games",
            "final_games",
        ):
            if records[0]["config"][key] != records[1]["config"][key]:
                raise ValueError(f"Capacity controls differ: {key}")
        for item in records:
            if item["evaluation_model"]["test_seed"] != 800000:
                raise ValueError("Unexpected test seed")
            if any(r["games"] != 100 for r in item["greedy"]):
                raise ValueError("Expected 100 Greedy games per color")
        lines += [
            "",
            "## 最终测试明细",
            "",
            "| 实验 | 执子 | 胜 / 和 / 负 | 胜率（95% Wilson 区间） | score | 验证选中局数 |",
            "|---|---|---|---|---:|---:|",
        ]
        for item in records:
            for r in item["greedy"]:
                lines.append(
                    f"| {item['name']} | {'黑' if r['color'] == 1 else '白'} | {r['wins']} / {r['draws']} / {r['losses']} | {r['win_rate'] * 100:.1f}% ({r['wilson_low'] * 100:.1f}–{r['wilson_high'] * 100:.1f}%) | {r['score']:.3f} | {item['best_episode']} |"
                )
        deltas = [
            100 * (wide["win_rate"] - small["win_rate"])
            for small, wide in zip(records[0]["greedy"], records[1]["greedy"])
        ]
        lines += [
            "",
            f"64/64/128 减 64/64/64：黑棋胜率差 {deltas[0]:+.1f} 个百分点，白棋 {deltas[1]:+.1f} 个百分点，两色平均 {(sum(deltas) / 2):+.1f} 个百分点。",
            "这只是 seed=11 的单次容量对照；不能据此断言加宽网络稳定更强或更弱。两组同源暖启动，新增训练预算相同。",
            "best.pt 可能来自暖启动后的第 0 局验证；表中的选中局数用于区分训练终点与实际测试权重。score=胜率+0.5×和率。",
        ]
    else:
        deltas = None
        lines += ["", "加宽组仍未完成最终测试，当前不作扩容效果结论。"]
    text = "\n".join(lines) + "\n"
    write_if_changed(out / "alphazero_capacity_comparison.md", text)
    payload = {
        "complete": complete,
        "experiments": records,
        "win_rate_differences_percentage_points": deltas,
        "interpretation": "Single-seed controlled comparison; final tests do not select models.",
    }
    write_if_changed(
        out / "alphazero_capacity_comparison.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    for relative, link in (
        ("README.md", "runs/production/alphazero_capacity_comparison.md"),
        ("rl/README.md", "../runs/production/alphazero_capacity_comparison.md"),
        ("docs/experiments.md", "../runs/production/alphazero_capacity_comparison.md"),
    ):
        path = root / relative
        if not path.exists():
            continue
        body = path.read_text(encoding="utf-8")
        start, end = MARKER + ":start -->", MARKER + ":end -->"
        brief = (
            "两组训练与最终测试均已完成。"
            if complete
            else "64/64/64 已完成；64/64/128 正在续训，最终成绩待完成后写入。"
        )
        block = f"{start}\n\n## AlphaZero 容量对照结果\n\n{brief} [查看 Greedy 分色结果与测试口径]({link})。\n\n{end}"
        if start in body:
            before, tail = body.split(start, 1)
            _, after = tail.split(end, 1)
            body = before + block + after
        else:
            body = body.rstrip() + "\n\n" + block + "\n"
        write_if_changed(path, body)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "runs/production")
    args = p.parse_args()
    result = publish(args.output)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "report": str(args.output / "alphazero_capacity_comparison.md"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
