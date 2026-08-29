#!/usr/bin/env python3
"""Build committed D2 P1 evidence from three completed Ultralytics run directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
MAP_KEY = "metrics/mAP50-95(B)"
THRESHOLD = 0.003
T_CRITICAL_95_DF2 = 4.302652729911275
PAIRED_RUNS = {
    20260824: {
        "off": REPO_ROOT / "runs/rhino_d2/p1/off-s20260824-2",
        "on": REPO_ROOT / "runs/rhino_d2/p1/on-s20260824",
    },
    20260825: {
        "off": REPO_ROOT / "runs/rhino_d2/p1/off-s20260825",
        "on": REPO_ROOT / "runs/rhino_d2/p1/on-s20260825",
    },
    20260826: {
        "off": REPO_ROOT / "runs/rhino_d2/p1/off-s20260826",
        "on": REPO_ROOT / "runs/rhino_d2/p1/on-s20260826",
    },
}


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(run_dir: Path) -> list[dict]:
    """Read numeric rows from one Ultralytics results.csv file."""
    path = run_dir / "results.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return [{key: float(value) for key, value in row.items() if value != ""} for row in csv.DictReader(handle)]


def summarize_arm(arm: str, run_dir: Path) -> tuple[dict, list[dict]]:
    """Summarize best validation metrics and checkpoint integrity for one arm."""
    import torch

    rows = read_rows(run_dir)
    best = max(rows, key=lambda row: row[MAP_KEY])
    checkpoint = run_dir / "weights" / "best.pt"
    healthy_checkpoint = run_dir / "weights" / "last_healthy.pt"
    best_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    healthy_payload = torch.load(healthy_checkpoint, map_location="cpu", weights_only=False)
    epoch_checkpoints = sorted((run_dir / "weights").glob("epoch*.pt"))
    return (
        {
            "arm": arm,
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "epochs_completed": len(rows),
            "best_epoch": int(best["epoch"]),
            "best_map50": best["metrics/mAP50(B)"],
            "best_map50_95": best[MAP_KEY],
            "final_map50_95": rows[-1][MAP_KEY],
            "final_foundation_loss": rows[-1].get("train/foundation_loss"),
            "best_checkpoint": {
                "path": str(checkpoint.relative_to(REPO_ROOT)),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
                "load_verified": isinstance(best_payload, dict) and best_payload.get("model") is not None,
            },
            "healthy_checkpoint": {
                "path": str(healthy_checkpoint.relative_to(REPO_ROOT)),
                "bytes": healthy_checkpoint.stat().st_size,
                "sha256": sha256(healthy_checkpoint),
                "load_verified": isinstance(healthy_payload, dict) and healthy_payload.get("ema") is not None,
            },
            "epoch_checkpoint_count": len(epoch_checkpoints),
            "epoch_checkpoint_sha256": {path.name: sha256(path) for path in epoch_checkpoints},
        },
        rows,
    )


def plot_curves(rows_by_arm: dict[str, list[dict]], output: Path) -> None:
    """Plot task metric and distillation curves without depending on Polars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm, rows in rows_by_arm.items():
        axes[0].plot([row["epoch"] for row in rows], [row[MAP_KEY] for row in rows], marker="o", label=arm)
        if "train/foundation_loss" in rows[0]:
            axes[1].plot(
                [row["epoch"] for row in rows],
                [row["train/foundation_loss"] for row in rows],
                marker="o",
                label=arm,
            )
    axes[0].axhline(0.003, color="black", linestyle="--", linewidth=1, label="0.003 decision scale")
    axes[0].set(title="COCO128 validation mAP50-95", xlabel="epoch", ylabel="mAP50-95")
    axes[1].set(title="Single-stage P4 foundation loss", xlabel="epoch", ylabel="unweighted KD loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("D2 first pair and relational ablation — seed 20260824")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_multiseed(pairs: list[dict], output: Path) -> None:
    """Plot paired best metrics and deltas against the pre-registered threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = [str(pair["seed"])[-2:] for pair in pairs]
    x = list(range(len(pairs)))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar([value - width / 2 for value in x], [pair["off_best_map50_95"] for pair in pairs], width, label="off")
    axes[0].bar([value + width / 2 for value in x], [pair["on_best_map50_95"] for pair in pairs], width, label="on")
    axes[0].set(
        title="Best mAP50-95 by paired seed", xlabel="seed suffix", ylabel="mAP50-95", xticks=x, xticklabels=seeds
    )
    axes[0].legend()
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axhline(THRESHOLD, color="tab:red", linestyle="--", linewidth=1, label="go scale +0.003")
    axes[1].axhline(-THRESHOLD, color="tab:red", linestyle="--", linewidth=1, label="no-go scale -0.003")
    axes[1].scatter(x, [pair["delta"] for pair in pairs], color="tab:purple", s=45, label="on - off")
    axes[1].set(title="Paired deltas", xlabel="seed suffix", ylabel="delta mAP50-95", xticks=x, xticklabels=seeds)
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("D2 P1 three-seed same-budget result")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_summary(run_dirs: dict[str, Path]) -> dict:
    """Build the first-pair result and a deliberately conservative decision record."""
    arms = {}
    rows_by_arm = {}
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for arm, run_dir in run_dirs.items():
        arms[arm], rows_by_arm[arm] = summarize_arm(arm, run_dir)
        shutil.copyfile(run_dir / "results.csv", RESULTS_ROOT / f"d2_p1_{arm}_s20260824.csv")
    delta = arms["on"]["best_map50_95"] - arms["off"]["best_map50_95"]
    relational_delta = arms["on"]["best_map50_95"] - arms["cosine-only"]["best_map50_95"]
    curve_path = RESULTS_ROOT / "d2_p1_first_results.png"
    plot_curves(rows_by_arm, curve_path)
    return {
        "schema_version": 1,
        "status": "completed_single_seed_first_pair_and_key_ablation",
        "claim": "directional_single_seed_coco128_evidence_not_a_go_no_go_decision",
        "protocol": {"seed": 20260824, "epochs": 10, "imgsz": 256, "batch": 4, "pretrained": False},
        "arms": arms,
        "comparisons": {
            "hybrid_on_minus_off_best_map50_95": delta,
            "hybrid_on_minus_cosine_only_best_map50_95": relational_delta,
            "absolute_delta_below_0_003": abs(delta) < 0.003,
        },
        "decision": {
            "status": "pending_more_seeds",
            "reason": "one paired seed cannot estimate the required 95% confidence interval",
            "required_next_seeds": [20260825, 20260826],
            "pre_registered_threshold": 0.003,
            "confidence_interval": None,
        },
        "artifacts": {
            "curve": {"path": str(curve_path.relative_to(REPO_ROOT)), "sha256": sha256(curve_path)},
            "csv": {
                arm: {
                    "path": f"experiments/rhino_d2/results/d2_p1_{arm}_s20260824.csv",
                    "sha256": sha256(RESULTS_ROOT / f"d2_p1_{arm}_s20260824.csv"),
                }
                for arm in run_dirs
            },
        },
    }


def _decision(mean_delta: float, ci_low: float, ci_high: float) -> tuple[str, str]:
    """Apply the pre-registered decision rule without moving its threshold."""
    contains_zero = ci_low <= 0 <= ci_high
    if mean_delta >= THRESHOLD and ci_low > 0:
        return "go", "mean delta reaches 0.003 and the paired 95% CI excludes zero"
    if abs(mean_delta) < THRESHOLD and contains_zero:
        return "no_go", "absolute mean delta is below 0.003 and the paired 95% CI contains zero"
    return "inconclusive", "the paired result satisfies neither the pre-registered go nor no-go rule"


def build_multiseed_summary(paired_runs: dict[int, dict[str, Path]]) -> dict:
    """Build paired three-seed statistics and independently verifiable evidence."""
    pairs = []
    arms = {}
    artifacts = {}
    for seed, run_dirs in paired_runs.items():
        arms[str(seed)] = {}
        artifacts[str(seed)] = {}
        for arm, run_dir in run_dirs.items():
            summary, _ = summarize_arm(arm, run_dir.resolve())
            arms[str(seed)][arm] = summary
            output_csv = RESULTS_ROOT / f"d2_p1_{arm}_s{seed}.csv"
            shutil.copyfile(run_dir / "results.csv", output_csv)
            artifacts[str(seed)][arm] = {
                "path": str(output_csv.relative_to(REPO_ROOT)),
                "sha256": sha256(output_csv),
            }
        off = arms[str(seed)]["off"]
        on = arms[str(seed)]["on"]
        pairs.append(
            {
                "seed": seed,
                "off_best_epoch": off["best_epoch"],
                "on_best_epoch": on["best_epoch"],
                "off_best_map50_95": off["best_map50_95"],
                "on_best_map50_95": on["best_map50_95"],
                "delta": on["best_map50_95"] - off["best_map50_95"],
            }
        )

    deltas = [pair["delta"] for pair in pairs]
    mean_delta = statistics.fmean(deltas)
    sample_std = statistics.stdev(deltas)
    standard_error = sample_std / math.sqrt(len(deltas))
    half_width = T_CRITICAL_95_DF2 * standard_error
    ci_low, ci_high = mean_delta - half_width, mean_delta + half_width
    status, reason = _decision(mean_delta, ci_low, ci_high)

    aggregate_csv = RESULTS_ROOT / "d2_p1_three_seed_results.csv"
    with aggregate_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    curve_path = RESULTS_ROOT / "d2_p1_three_seed_results.png"
    plot_multiseed(pairs, curve_path)
    runtime_records = []
    for seed in (20260825, 20260826):
        for arm in ("off", "on"):
            manifest_path = REPO_ROOT / "runs/rhino_d2/manifests" / f"{arm}-s{seed}.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            log_path = Path(manifest["log"])
            runtime_records.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "returncode": manifest["returncode"],
                    "started_utc": manifest["started_utc"],
                    "finished_utc": manifest["finished_utc"],
                    "config_sha256": manifest["config_sha256"],
                    "log": str(log_path.relative_to(REPO_ROOT)),
                    "log_bytes": log_path.stat().st_size,
                    "log_sha256": manifest["log_sha256"],
                    "log_hash_verified": sha256(log_path) == manifest["log_sha256"],
                }
            )
    return {
        "schema_version": 1,
        "status": "completed_three_seed_same_budget_pair",
        "claim": "formal_p1_decision_on_coco128_10_epoch_from_scratch_protocol_not_a_general_accuracy_claim",
        "protocol": {
            "seeds": list(paired_runs),
            "epochs": 10,
            "imgsz": 256,
            "batch": 4,
            "pretrained": False,
            "paired_metric": "best metrics/mAP50-95(B) within the fixed 10-epoch budget",
        },
        "pairs": pairs,
        "arms": arms,
        "statistics": {
            "n": len(deltas),
            "paired_deltas": deltas,
            "mean_delta": mean_delta,
            "sample_standard_deviation": sample_std,
            "standard_error": standard_error,
            "t_critical_95_df2": T_CRITICAL_95_DF2,
            "confidence_interval_95": [ci_low, ci_high],
            "confidence_interval_contains_zero": ci_low <= 0 <= ci_high,
        },
        "decision": {
            "status": status,
            "reason": reason,
            "pre_registered_threshold": THRESHOLD,
            "threshold_percentage_points": THRESHOLD * 100,
            "rule_unchanged_after_observing_results": True,
        },
        "artifacts": {
            "per_run_csv": artifacts,
            "aggregate_csv": {"path": str(aggregate_csv.relative_to(REPO_ROOT)), "sha256": sha256(aggregate_csv)},
            "curve": {"path": str(curve_path.relative_to(REPO_ROOT)), "sha256": sha256(curve_path)},
            "runtime_records": runtime_records,
            "runtime_record_coverage": {
                "complete_console_tee": [20260825, 20260826],
                "seed_20260824": "results_csv_args_and_checkpoint_hashes_only; original console was not tee-captured",
            },
        },
    }


def main() -> None:
    """Parse paths and write the JSON evidence file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", type=Path, default=REPO_ROOT / "runs/rhino_d2/p1/off-s20260824-2")
    parser.add_argument("--on", type=Path, default=REPO_ROOT / "runs/rhino_d2/p1/on-s20260824")
    parser.add_argument("--cosine-only", type=Path, default=REPO_ROOT / "runs/rhino_d2/p1/cosine-only-s20260824")
    args = parser.parse_args()
    result = build_summary(
        {"off": args.off.resolve(), "on": args.on.resolve(), "cosine-only": args.cosine_only.resolve()}
    )
    output = RESULTS_ROOT / "d2_p1_first_results.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    multiseed = build_multiseed_summary(PAIRED_RUNS)
    multiseed_output = RESULTS_ROOT / "d2_p1_three_seed_results.json"
    multiseed_output.write_text(json.dumps(multiseed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(multiseed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
