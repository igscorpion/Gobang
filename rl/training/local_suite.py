"""Sequential long-run plan. Resume skips complete stages and never uploads."""

from pathlib import Path
import argparse, sys, subprocess, time, json, csv, statistics
from rl.training.common import atomic_json, exclusive_run
from rl.evaluation.report_capacity import publish as publish_capacity

ROOT = Path(__file__).resolve().parents[2]


def stages():
    az_source = "backups/upstream-2711cca/models/alphazero.pt"
    result = []
    # Only input channels change between each pair; both use corrected targets.
    for seed in (11, 22, 33):
        for channels in (3, 4):
            name = f"dqn{channels}_seed{seed}"
            result.append(
                (
                    name,
                    "rl.training.local_dqn",
                    [
                        "--input-channels",
                        str(channels),
                        "--episodes",
                        "30000",
                        "--seed",
                        str(seed),
                        "--resume",
                    ],
                )
            )
    # Same warm source, seed, budget, simulations and opponent ratio: width alone changes.
    for width in (64, 128):
        name = f"az_width{width}_fixed"
        result.append(
            (
                name,
                "rl.training.local_alphazero",
                [
                    "--channels",
                    "64",
                    "64",
                    str(width),
                    "--episodes",
                    "1000",
                    "--sims",
                    "400",
                    "--fixed-ratio",
                    ".6",
                    "--seed",
                    "11",
                    "--warm-start",
                    az_source,
                    "--resume",
                ],
            )
        )
    # Paired second stage from the same widened model, fresh optimizer in both arms.
    for schedule in ("fixed", "adaptive"):
        args = [
            "--episodes",
            "1000",
            "--sims",
            "400",
            "--seed",
            "22",
            "--warm-start",
            "runs/production/az_width128_fixed/final.pt",
            "--resume",
        ]
        if schedule == "fixed":
            args += ["--fixed-ratio", ".6"]
        result.append(
            (f"az_wide_continue_{schedule}", "rl.training.local_alphazero", args)
        )
    # Deliver a trained four-channel model early, then alternate algorithm families.
    by_name = {item[0]: item for item in result}
    order = [
        "dqn4_seed11",
        "az_width64_fixed",
        "az_width128_fixed",
        "dqn3_seed11",
        "az_wide_continue_fixed",
        "az_wide_continue_adaptive",
        "dqn3_seed22",
        "dqn4_seed22",
        "dqn3_seed33",
        "dqn4_seed33",
    ]
    return [by_name[name] for name in order]


def update_report(out):
    lines = [
        "# 本地开发与训练结果",
        "",
        "结果仅来自已完成阶段；训练中阶段不填入预测成绩。",
        "",
        "| 阶段 | 状态 | Random 黑 / 白胜率 | Greedy 黑 / 白胜率 |",
        "|---|---|---:|---:|",
    ]
    candidates = {}
    completed = {}
    for name, _, _ in stages():
        directory = out / name
        p = directory / "status.json"
        status = (
            json.loads(p.read_text(encoding="utf-8"))
            if p.exists()
            else {"state": "pending"}
        )
        phase = status["state"]
        detail = phase + (
            " " + str(status.get("episode", 0)) if "episode" in status else ""
        )
        result = directory / "final_evaluation.json"
        if phase == "complete" and result.exists():
            rows = json.loads(result.read_text(encoding="utf-8"))
            completed[name] = rows
            rates = [
                " / ".join(
                    f"{r['win_rate'] * 100:.1f}%"
                    for r in rows
                    if r["opponent"] == opponent
                )
                for opponent in ("random", "greedy")
            ]
            val = directory / "validation.csv"
            if val.exists():
                with val.open(encoding="utf-8") as f:
                    records = list(csv.DictReader(f))
                scores = {}
                for r in records:
                    if r["opponent"] == "greedy":
                        scores.setdefault(int(r["episode"]), []).append(
                            float(r["score"])
                        )
                best = max(sum(v) / len(v) for v in scores.values())
                family = (
                    "dqn4"
                    if name.startswith("dqn4")
                    else (
                        "alphazero_wide"
                        if name == "az_width128_fixed"
                        or name.startswith("az_wide_continue")
                        else None
                    )
                )
                if family and (
                    family not in candidates or best > candidates[family][0]
                ):
                    ck = directory / "best.pt"
                    candidates[family] = (
                        best,
                        {
                            "algorithm": "dqn" if family == "dqn4" else "alphazero",
                            "checkpoint": str(ck.relative_to(ROOT)).replace("\\", "/"),
                            "selection": "validation score only",
                            "validation_score": best,
                        },
                    )
        else:
            rates = ["—", "—"]
        lines.append(f"| {name} | {detail} | {rates[0]} | {rates[1]} |")
    lines += [
        "",
        "## DQN 跨种子汇总",
        "",
        "| 输入 | 已完成种子 | 对手 / 执子 | 胜率均值 ± 样本标准差 |",
        "|---|---:|---|---:|",
    ]
    for channels in (3, 4):
        names = [
            f"dqn{channels}_seed{s}"
            for s in (11, 22, 33)
            if f"dqn{channels}_seed{s}" in completed
        ]
        for opponent in ("random", "greedy"):
            for color in (1, -1):
                values = [
                    next(
                        r["win_rate"]
                        for r in completed[n]
                        if r["opponent"] == opponent and r["color"] == color
                    )
                    for n in names
                ]
                value = (
                    f"{statistics.mean(values) * 100:.1f}% ± {statistics.stdev(values) * 100:.1f}%"
                    if len(values) > 1
                    else (f"{values[0] * 100:.1f}% (单种子)" if values else "—")
                )
                lines.append(
                    f"| {channels} | {len(values)} / 3 | {opponent} / {'黑' if color == 1 else '白'} | {value} |"
                )
    paired = [
        s for s in (11, 22, 33) if all(f"dqn{c}_seed{s}" in completed for c in (3, 4))
    ]
    if paired:
        lines += ["", "4 通道减 3 通道的配对胜率差（百分点）："]
        for opponent in ("random", "greedy"):
            for color in (1, -1):
                ds = [
                    100
                    * (
                        next(
                            r["win_rate"]
                            for r in completed[f"dqn4_seed{s}"]
                            if r["opponent"] == opponent and r["color"] == color
                        )
                        - next(
                            r["win_rate"]
                            for r in completed[f"dqn3_seed{s}"]
                            if r["opponent"] == opponent and r["color"] == color
                        )
                    )
                    for s in paired
                ]
                lines.append(
                    f"- {opponent} / {'黑' if color == 1 else '白'}：{statistics.mean(ds):+.1f}，已配对 {len(ds)} 个种子；三种子结果仍属有限重复。"
                )
    lines += [
        "",
        "## 规则对照",
        "",
        "| 策略 | 规则 | 局数 | 黑胜率（95% Wilson 区间） | 和棋率 |",
        "|---|---|---:|---:|---:|",
    ]
    for algorithm in ("greedy", "alphazero"):
        path = out / f"rule_comparison_{algorithm}" / "summary.json"
        if not path.exists():
            lines.append(f"| {algorithm} | 待完成 | — | — | — |")
            continue
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            lines.append(
                f"| {algorithm} | {row['rules']} | {row['games']} | {100 * row['black_win_rate']:.1f}% ({100 * row['wilson_low']:.1f}–{100 * row['wilson_high']:.1f}%) | {100 * row['draw_rate']:.1f}% |"
            )
    lines += [
        "",
        "## 口径与限制",
        "",
        "结果表测试验证选出的 best.pt；final.pt 为训练终点，单独保留用于续训和后续配对暖启动。",
        "DQN 配对使用种子 11/22/33，各 30,000 局；AlphaZero 容量与课程对照各阶段 1,000 局，400 次模拟，当前为单种子探索。",
        "所有评估分黑白、胜和负，验证每色每对手 40 局，最终每色每对手 100 局；不同算法计算预算不同，不称严格计算成本公平。",
        "网络扩容和课程比例分别比较；3 通道对照也使用修正后的合法掩码、Double DQN 与混合经验视角，不能把它当成上游原始 DQN。",
        "网页模型仅按验证结果选择，测试不参与选择。规则评估使用冻结权重与配对随机开局，不能证明 9×9 先手必胜。",
    ]
    path = out / "实验报告.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    if candidates:
        atomic_json(out / "models.json", {k: v[1] for k, v in candidates.items()})
    publish_capacity(out)


def run(args):
    out = ROOT / "runs/production"
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        update_report(out)
        return
    tasks = stages()
    atomic_json(
        out / "plan.json",
        {
            "dqn_training_games": 180000,
            "alphazero_training_games": 4000,
            "stages": tasks,
            "rules": "freestyle during training; frozen rule comparison afterward",
            "only_local": True,
        },
    )
    # Produce the untrained policy reference before the longer learning stages.
    directory = out / "rule_comparison_greedy"
    if not (directory / "summary.json").exists():
        directory.mkdir(exist_ok=True)
        command = [
            sys.executable,
            "-u",
            "-m",
            "rl.evaluation.compare_rules",
            "--algorithm",
            "greedy",
            "--games",
            "300",
            "--output",
            str(directory),
        ]
        atomic_json(
            out / "suite_status.json",
            {"state": "running", "stage": "rule_comparison_greedy", "command": command},
        )
        with (directory / "console.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise SystemExit(result.returncode)
        update_report(out)
    for name, module, flags in tasks:
        directory = out / name
        directory.mkdir(exist_ok=True)
        status = directory / "status.json"
        if (
            status.exists()
            and json.loads(status.read_text(encoding="utf-8")).get("state")
            == "complete"
        ):
            continue
        device = args.az_device if module.endswith("local_alphazero") else args.device
        command = [
            sys.executable,
            "-u",
            "-m",
            module,
            "--output",
            str(directory),
            "--device",
            device,
            *flags,
        ]
        atomic_json(
            out / "suite_status.json",
            {"state": "running", "stage": name, "command": command},
        )
        update_report(out)
        with (directory / "console.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            atomic_json(
                out / "suite_status.json",
                {
                    "state": "running",
                    "stage": name,
                    "child_pid": process.pid,
                    "command": command,
                },
            )
            while process.poll() is None:
                time.sleep(30)
                update_report(out)
        if process.returncode:
            atomic_json(
                out / "suite_status.json",
                {"state": "failed", "stage": name, "exit_code": process.returncode},
            )
            update_report(out)
            raise SystemExit(process.returncode)
        update_report(out)
    for algorithm, games in (("greedy", 300), ("alphazero", 100)):
        directory = out / f"rule_comparison_{algorithm}"
        if (directory / "summary.json").exists():
            continue
        command = [
            sys.executable,
            "-u",
            "-m",
            "rl.evaluation.compare_rules",
            "--algorithm",
            algorithm,
            "--games",
            str(games),
            "--output",
            str(directory),
        ]
        if algorithm == "alphazero":
            registry = json.loads((out / "models.json").read_text(encoding="utf-8"))
            command += [
                "--model",
                registry["alphazero_wide"]["checkpoint"],
                "--sims",
                "400",
                "--device",
                args.az_device,
            ]
        directory.mkdir(exist_ok=True)
        atomic_json(
            out / "suite_status.json",
            {
                "state": "running",
                "stage": f"rule_comparison_{algorithm}",
                "command": command,
            },
        )
        with (directory / "console.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise SystemExit(result.returncode)
    update_report(out)
    atomic_json(out / "suite_status.json", {"state": "complete"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--az-device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    if args.report_only:
        run(args)
    else:
        with exclusive_run(ROOT / "runs/production/suite.lock"):
            run(args)


if __name__ == "__main__":
    main()
