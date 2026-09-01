#!/usr/bin/env python3
"""Archive and summarize the frozen DINOv3-S formal P1 paired runs."""

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
PRECISION_KEY = "metrics/precision(B)"
RECALL_KEY = "metrics/recall(B)"
FOUNDATION_KEY = "train/foundation_loss"
FOUNDATION_RATIO_KEY = "train/foundation_task_ratio"
LATE_WINDOW = 10
REQUIRED_EPOCHS = 50
THRESHOLD = 0.003
T_CRITICAL_95_DF2 = 4.302652729911275
SEEDS = (20260824, 20260825, 20260826)


def sha256(path: Path) -> str:
    """Return one file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, float]]:
    """Read numeric Ultralytics CSV rows."""
    with path.open(encoding="utf-8", newline="") as handle:
        return [{key: float(value) for key, value in row.items() if value != ""} for row in csv.DictReader(handle)]


def archive(source: Path, destination: Path) -> dict[str, object]:
    """Copy one immutable experiment artifact and describe it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {
        "path": str(destination.relative_to(REPO_ROOT)).replace("\\", "/"),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def summarize_arm(arm: str, seed: int, run_dir: Path, manifest_path: Path) -> tuple[dict, list[dict[str, float]]]:
    """Verify, archive, and summarize one arm without selecting a best epoch."""
    rows = read_rows(run_dir / "results.csv")
    if len(rows) != REQUIRED_EPOCHS:
        raise ValueError(f"{arm} expected {REQUIRED_EPOCHS} epochs, found {len(rows)}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log_path = Path(manifest["log"])
    config_path = REPO_ROOT / manifest["config"]
    checks = {
        "returncode_zero": manifest["returncode"] == 0,
        "completed_50_epochs": len(rows) == REQUIRED_EPOCHS,
        "experiment_inputs_clean": manifest["source_state"]["experiment_inputs_dirty"] is False,
        "config_hash_verified": sha256(config_path) == manifest["config_sha256"],
        "log_hash_verified": sha256(log_path) == manifest["log_sha256"],
        "checkpoint_manifest_complete": len(manifest["checkpoints"]) == REQUIRED_EPOCHS + 3,
    }
    if not all(checks.values()):
        raise RuntimeError(f"{arm} evidence failed closed: {checks}")

    prefix = f"d2_v3_p1_{arm}_s{seed}"
    artifacts = {
        "results": archive(run_dir / "results.csv", RESULTS_ROOT / f"{prefix}.csv"),
        "args": archive(run_dir / "args.yaml", RESULTS_ROOT / f"{prefix}.args.yaml"),
        "complete_log": archive(log_path, RESULTS_ROOT / f"{prefix}.log"),
        "runtime_manifest": archive(manifest_path, RESULTS_ROOT / f"{prefix}.manifest.json"),
    }
    late = rows[-LATE_WINDOW:]
    foundation_values = [row.get(FOUNDATION_KEY, 0.0) for row in rows]
    return (
        {
            "arm": arm,
            "seed": seed,
            "epochs_completed": len(rows),
            "metric_contract": f"median of epochs {REQUIRED_EPOCHS - LATE_WINDOW + 1}-{REQUIRED_EPOCHS}",
            "late_median_map50_95": statistics.median(row[MAP_KEY] for row in late),
            "late_median_precision": statistics.median(row[PRECISION_KEY] for row in late),
            "late_median_recall": statistics.median(row[RECALL_KEY] for row in late),
            "final_map50_95": rows[-1][MAP_KEY],
            "best_map50_95_audit_only": max(row[MAP_KEY] for row in rows),
            "foundation_loss_first": foundation_values[0],
            "foundation_loss_final": foundation_values[-1],
            "foundation_loss_nonzero": any(value > 0 for value in foundation_values),
            "late_median_foundation_task_ratio": statistics.median(row.get(FOUNDATION_RATIO_KEY, 0.0) for row in late),
            "source_commit": manifest["source_state"]["commit"],
            "checks": checks,
            "artifacts": artifacts,
        },
        rows,
    )


def plot_pair(seed: int, off_rows: list[dict[str, float]], on_rows: list[dict[str, float]], output: Path) -> None:
    """Plot the full metric trace and ON distillation loss."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in off_rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [row[MAP_KEY] for row in off_rows], label="OFF")
    axes[0].plot(epochs, [row[MAP_KEY] for row in on_rows], label="ON")
    axes[0].axvspan(41, 50, alpha=0.12, color="tab:green", label="registered last-10 window")
    axes[0].set(title="Validation mAP50-95", xlabel="epoch", ylabel="mAP50-95")
    axes[1].plot(epochs, [row[FOUNDATION_KEY] for row in on_rows], color="tab:purple", label="ON foundation loss")
    axes[1].axvspan(41, 50, alpha=0.12, color="tab:green")
    axes[1].set(title="Single-stage P4 cosine loss", xlabel="epoch", ylabel="unweighted loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(f"D2 DINOv3-S formal P1 — paired seed {seed}")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_three_seed(pairs: list[dict], output: Path) -> None:
    """Plot registered per-arm summaries and paired deltas."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(pairs)))
    labels = [str(pair["seed"])[-2:] for pair in pairs]
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].bar(
        [value - width / 2 for value in x],
        [pair["off_late_median_map50_95"] for pair in pairs],
        width,
        label="OFF",
    )
    axes[0].bar(
        [value + width / 2 for value in x],
        [pair["on_late_median_map50_95"] for pair in pairs],
        width,
        label="ON",
    )
    axes[0].set(
        title="Registered last-10 median by seed",
        xlabel="seed suffix",
        ylabel="mAP50-95",
        xticks=x,
        xticklabels=labels,
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axhline(THRESHOLD, color="tab:red", linestyle="--", linewidth=1, label="±0.003 scale")
    axes[1].axhline(-THRESHOLD, color="tab:red", linestyle="--", linewidth=1)
    axes[1].scatter(x, [pair["delta"] for pair in pairs], color="tab:purple", s=55, label="ON − OFF")
    axes[1].set(
        title="Paired deltas",
        xlabel="seed suffix",
        ylabel="Δ mAP50-95",
        xticks=x,
        xticklabels=labels,
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("D2 DINOv3-S formal P1 — three paired seeds")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def decision(mean_delta: float, ci_low: float, ci_high: float) -> tuple[str, str]:
    """Apply the frozen DINOv3-S P1 Go/No-Go rule."""
    contains_zero = ci_low <= 0 <= ci_high
    if mean_delta >= THRESHOLD and ci_low > 0:
        return "go", "mean delta reaches 0.003 and the paired 95% CI excludes zero"
    if abs(mean_delta) < THRESHOLD and contains_zero:
        return "no_go", "absolute mean delta is below 0.003 and the paired 95% CI contains zero"
    return "inconclusive", "the result satisfies neither the frozen go nor no-go rule"


def build_three_seed_summary() -> dict:
    """Build the formal three-seed paired decision from registered arm summaries."""
    pair_paths = {
        20260824: RESULTS_ROOT / "d2_v3_p1_first_pair.json",
        20260825: RESULTS_ROOT / "d2_v3_p1_pair_s20260825.json",
        20260826: RESULTS_ROOT / "d2_v3_p1_pair_s20260826.json",
    }
    pair_results = {seed: json.loads(path.read_text(encoding="utf-8")) for seed, path in pair_paths.items()}
    pairs = [
        {
            "seed": seed,
            "off_late_median_map50_95": pair_results[seed]["pair"]["off_late_median_map50_95"],
            "on_late_median_map50_95": pair_results[seed]["pair"]["on_late_median_map50_95"],
            "delta": pair_results[seed]["pair"]["delta"],
        }
        for seed in SEEDS
    ]
    deltas = [pair["delta"] for pair in pairs]
    mean_delta = statistics.fmean(deltas)
    sample_std = statistics.stdev(deltas)
    standard_error = sample_std / math.sqrt(len(deltas))
    half_width = T_CRITICAL_95_DF2 * standard_error
    ci_low, ci_high = mean_delta - half_width, mean_delta + half_width
    status, reason = decision(mean_delta, ci_low, ci_high)

    manifests = {
        (seed, arm): json.loads((RESULTS_ROOT / f"d2_v3_p1_{arm}_s{seed}.manifest.json").read_text(encoding="utf-8"))
        for seed in SEEDS
        for arm in ("off", "on")
    }
    input_checks = {
        "all_runs_returncode_zero": all(manifest["returncode"] == 0 for manifest in manifests.values()),
        "all_experiment_inputs_clean": all(
            manifest["source_state"]["experiment_inputs_dirty"] is False for manifest in manifests.values()
        ),
        "off_config_hash_constant": len({manifests[(seed, "off")]["config_sha256"] for seed in SEEDS}) == 1,
        "on_config_hash_constant": len({manifests[(seed, "on")]["config_sha256"] for seed in SEEDS}) == 1,
        "runner_hash_constant": len({manifest["runner_sha256"] for manifest in manifests.values()}) == 1,
        "paired_metric_constant": all(
            result["protocol"]["paired_metric"] == "median_last10(metrics/mAP50-95(B))"
            and result["protocol"]["uses_best_epoch"] is False
            for result in pair_results.values()
        ),
    }
    if not all(input_checks.values()):
        raise RuntimeError(f"three-seed input audit failed closed: {input_checks}")

    aggregate_csv = RESULTS_ROOT / "d2_v3_p1_three_seed_results.csv"
    with aggregate_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    curve_path = RESULTS_ROOT / "d2_v3_p1_three_seed_results.png"
    plot_three_seed(pairs, curve_path)
    return {
        "schema_version": 1,
        "status": "completed_three_seed_formal_p1",
        "claim": "formal_decision_for_frozen_dinov3_s_coco_mini_50_epoch_protocol_not_a_general_kd_claim",
        "protocol": {
            "seeds": list(SEEDS),
            "epochs": REQUIRED_EPOCHS,
            "late_window": LATE_WINDOW,
            "paired_metric": "median_last10(metrics/mAP50-95(B))",
            "uses_best_epoch": False,
            "pre_registered_threshold": THRESHOLD,
            "t_critical_95_df2": T_CRITICAL_95_DF2,
            "summarizer_sha256": sha256(Path(__file__)),
        },
        "input_audit": input_checks,
        "pairs": pairs,
        "statistics": {
            "n": len(deltas),
            "paired_deltas": deltas,
            "mean_delta": mean_delta,
            "sample_standard_deviation": sample_std,
            "standard_error": standard_error,
            "confidence_interval_95": [ci_low, ci_high],
            "confidence_interval_contains_zero": ci_low <= 0 <= ci_high,
        },
        "decision": {
            "status": status,
            "reason": reason,
            "rule_unchanged_after_observing_results": True,
            "expand_to_multistage_or_vitl": False,
        },
        "artifacts": {
            "per_seed_json": {
                str(seed): {"path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha256(path)}
                for seed, path in pair_paths.items()
            },
            "aggregate_csv": {
                "path": str(aggregate_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
                "bytes": aggregate_csv.stat().st_size,
                "sha256": sha256(aggregate_csv),
            },
            "curve": {
                "path": str(curve_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "bytes": curve_path.stat().st_size,
                "sha256": sha256(curve_path),
            },
        },
    }


def build_summary(seed: int, run_root: Path, manifest_root: Path) -> dict:
    """Build a conservative first-pair record; one seed can never decide Go/No-Go."""
    arms: dict[str, dict] = {}
    rows: dict[str, list[dict[str, float]]] = {}
    for arm in ("off", "on"):
        name = f"v3-p1-{arm}-s{seed}"
        arms[arm], rows[arm] = summarize_arm(arm, seed, run_root / name, manifest_root / f"{name}.json")

    if arms["off"]["source_commit"] != arms["on"]["source_commit"]:
        raise RuntimeError("paired arms were not run from the same source commit")
    if arms["off"]["foundation_loss_nonzero"]:
        raise RuntimeError("OFF unexpectedly contains non-zero foundation loss")
    if not arms["on"]["foundation_loss_nonzero"]:
        raise RuntimeError("ON did not activate foundation loss")

    delta = arms["on"]["late_median_map50_95"] - arms["off"]["late_median_map50_95"]
    stem = "d2_v3_p1_first_pair" if seed == 20260824 else f"d2_v3_p1_pair_s{seed}"
    pair_csv = RESULTS_ROOT / f"{stem}.csv"
    with pair_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "off_late_median_map50_95", "on_late_median_map50_95", "delta"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seed": seed,
                "off_late_median_map50_95": arms["off"]["late_median_map50_95"],
                "on_late_median_map50_95": arms["on"]["late_median_map50_95"],
                "delta": delta,
            }
        )
    curve_path = RESULTS_ROOT / f"{stem}.png"
    plot_pair(seed, rows["off"], rows["on"], curve_path)
    required_next_seeds = [value for value in (20260824, 20260825, 20260826) if value > seed]
    ordinal = {20260824: "first", 20260825: "second", 20260826: "third"}.get(seed, "additional")
    individual_status = "pending_more_seeds" if required_next_seeds else "pending_three_seed_aggregate"
    individual_reason = (
        "one paired seed cannot estimate the pre-registered paired 95% confidence interval"
        if required_next_seeds
        else "an individual pair cannot issue the decision; aggregate all three registered paired seeds"
    )
    return {
        "schema_version": 1,
        "status": f"completed_{ordinal}_of_three_paired_seeds",
        "claim": "single_seed_directional_evidence_not_a_go_no_go_decision",
        "protocol": {
            "seed": seed,
            "epochs": REQUIRED_EPOCHS,
            "late_window": LATE_WINDOW,
            "paired_metric": "median_last10(metrics/mAP50-95(B))",
            "uses_best_epoch": False,
            "pre_registered_threshold": THRESHOLD,
            "summarizer_sha256": sha256(Path(__file__)),
        },
        "arms": arms,
        "pair": {
            "off_late_median_map50_95": arms["off"]["late_median_map50_95"],
            "on_late_median_map50_95": arms["on"]["late_median_map50_95"],
            "delta": delta,
            "absolute_delta_below_0p003": abs(delta) < THRESHOLD,
        },
        "decision": {
            "status": individual_status,
            "reason": individual_reason,
            "required_next_seeds": required_next_seeds,
            "confidence_interval_95": None,
            "go_no_go_allowed": False,
            "rule_unchanged_after_observing_results": True,
        },
        "artifacts": {
            "pair_csv": {
                "path": str(pair_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
                "bytes": pair_csv.stat().st_size,
                "sha256": sha256(pair_csv),
            },
            "curve": {
                "path": str(curve_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "bytes": curve_path.stat().st_size,
                "sha256": sha256(curve_path),
            },
        },
    }


def main() -> None:
    """Parse paths and persist first-pair evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "runs/rhino_d2/v3_p1")
    parser.add_argument("--manifest-root", type=Path, default=REPO_ROOT / "runs/rhino_d2/manifests")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        result = build_three_seed_summary()
        output = RESULTS_ROOT / "d2_v3_p1_three_seed_results.json"
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    result = build_summary(args.seed, args.run_root.resolve(), args.manifest_root.resolve())
    stem = "d2_v3_p1_first_pair" if args.seed == 20260824 else f"d2_v3_p1_pair_s{args.seed}"
    output = RESULTS_ROOT / f"{stem}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
