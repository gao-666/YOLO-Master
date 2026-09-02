#!/usr/bin/env python3
"""Summarize the pre-registered DINOv3-S P2-02 ON128 diagnosis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
RESULTS_ROOT = EXPERIMENT_ROOT / "results/p2_align_dim"
P1_RESULT = EXPERIMENT_ROOT / "results/d2_v3_p1_three_seed_results.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs/rhino_d2/v3_p2_align_dim"
DEFAULT_MANIFEST_ROOT = REPO_ROOT / "runs/rhino_d2/p2_align_dim/manifests"
MAP_KEY = "metrics/mAP50-95(B)"
FOUNDATION_KEY = "train/foundation_loss"
FOUNDATION_RATIO_KEY = "train/foundation_task_ratio"
SEEDS = (20260824, 20260825, 20260826)
REQUIRED_EPOCHS = 50
LATE_WINDOW = 10
THRESHOLD = 0.003
T_CRITICAL_95_DF2 = 4.302652729911275


def sha256(path: Path) -> str:
    """Return one file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, float]]:
    """Read numeric Ultralytics result rows."""
    with path.open(encoding="utf-8", newline="") as handle:
        return [{key: float(value) for key, value in row.items() if value != ""} for row in csv.DictReader(handle)]


def archive(source: Path, destination: Path) -> dict[str, Any]:
    """Copy one immutable evidence artifact and describe it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {
        "path": destination.relative_to(REPO_ROOT).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def paired_interval(values: list[float]) -> dict[str, Any]:
    """Compute the frozen paired 95% t interval for three seeds."""
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    standard_error = sample_std / math.sqrt(len(values))
    half_width = T_CRITICAL_95_DF2 * standard_error
    low, high = mean - half_width, mean + half_width
    return {
        "n": len(values),
        "paired_deltas": values,
        "mean_delta": mean,
        "sample_standard_deviation": sample_std,
        "standard_error": standard_error,
        "confidence_interval_95": [low, high],
        "confidence_interval_contains_zero": low <= 0 <= high,
    }


def decision(mean_delta: float, ci_low: float, ci_high: float) -> tuple[str, str]:
    """Apply the pre-registered three-level P2-02 rule."""
    contains_zero = ci_low <= 0 <= ci_high
    if mean_delta >= THRESHOLD and ci_low > 0:
        return "support", "mean delta reaches +0.003 and the paired 95% CI is entirely above zero"
    if abs(mean_delta) < THRESHOLD and contains_zero:
        return "no_support", "absolute mean delta is below 0.003 and the paired 95% CI contains zero"
    return "inconclusive", "the comparison satisfies neither the pre-registered support nor no-support rule"


def summarize_seed(seed: int, run_root: Path, manifest_root: Path) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Fail closed while archiving one ON128 seed."""
    name = f"v3-p2-align128-s{seed}"
    run_dir = run_root / name
    manifest_path = manifest_root / f"{name}.json"
    results_path = run_dir / "results.csv"
    rows = read_rows(results_path)
    if len(rows) != REQUIRED_EPOCHS:
        raise ValueError(f"{name} expected {REQUIRED_EPOCHS} epochs, found {len(rows)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = REPO_ROOT / manifest["config"]
    log = Path(manifest["log"])
    checks = {
        "returncode_zero": manifest["returncode"] == 0,
        "completed_50_epochs": len(rows) == REQUIRED_EPOCHS,
        "arm_is_on128": manifest["arm"] == "on128",
        "seed_matches": manifest["seed"] == seed,
        "config_hash_verified": sha256(config) == manifest["config_sha256"],
        "preflight_hash_verified": sha256(REPO_ROOT / manifest["preflight"]) == manifest["preflight_sha256"],
        "log_hash_verified": sha256(log) == manifest["log_sha256"],
        "has_epoch9_24_49_checkpoints": all(
            any(item["name"] == checkpoint for item in manifest["checkpoints"])
            for checkpoint in ("epoch9.pt", "epoch24.pt", "epoch49.pt")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{name} evidence failed closed: {checks}")
    prefix = f"d2_v3_p2_align128_s{seed}"
    artifacts = {
        "results": archive(results_path, RESULTS_ROOT / f"{prefix}.csv"),
        "args": archive(run_dir / "args.yaml", RESULTS_ROOT / f"{prefix}.args.yaml"),
        "complete_log": archive(log, RESULTS_ROOT / f"{prefix}.log"),
        "runtime_manifest": archive(manifest_path, RESULTS_ROOT / f"{prefix}.manifest.json"),
    }
    late = rows[-LATE_WINDOW:]
    return (
        {
            "seed": seed,
            "epochs_completed": len(rows),
            "late_median_map50_95": statistics.median(row[MAP_KEY] for row in late),
            "late_median_foundation_loss": statistics.median(row.get(FOUNDATION_KEY, 0.0) for row in late),
            "late_median_foundation_task_ratio": statistics.median(row.get(FOUNDATION_RATIO_KEY, 0.0) for row in late),
            "final_map50_95": rows[-1][MAP_KEY],
            "best_map50_95_audit_only": max(row[MAP_KEY] for row in rows),
            "source_commit": manifest["source_state"]["commit"],
            "checks": checks,
            "artifacts": artifacts,
        },
        rows,
    )


def plot_results(pairs: list[dict[str, Any]], output: Path) -> None:
    """Plot registered late-window metrics and paired effects."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(pairs)))
    labels = [str(row["seed"])[-2:] for row in pairs]
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for offset, key, label in (
        (-width, "off", "OFF"),
        (0, "on64", "ON64"),
        (width, "on128", "ON128"),
    ):
        axes[0].bar([value + offset for value in x], [row[key] for row in pairs], width, label=label)
    axes[0].set(
        title="Registered last-10 median", xlabel="seed suffix", ylabel="mAP50-95", xticks=x, xticklabels=labels
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axhline(THRESHOLD, color="tab:red", linestyle="--", linewidth=1, label="±0.003 scale")
    axes[1].axhline(-THRESHOLD, color="tab:red", linestyle="--", linewidth=1)
    axes[1].scatter(x, [row["delta_dim"] for row in pairs], label="ON128 − ON64", s=55)
    axes[1].scatter(x, [row["delta_off"] for row in pairs], label="ON128 − OFF", marker="x", s=65)
    axes[1].set(
        title="Pre-registered paired effects", xlabel="seed suffix", ylabel="Δ mAP50-95", xticks=x, xticklabels=labels
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("D2 DINOv3-S P2-02 alignment-dimension diagnosis")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_summary(run_root: Path, manifest_root: Path) -> dict[str, Any]:
    """Build and write the formal P2-02 result."""
    p1 = json.loads(P1_RESULT.read_text(encoding="utf-8"))
    if p1["decision"]["status"] != "no_go":
        raise RuntimeError("frozen P1 reference is not No-Go")
    references = {row["seed"]: row for row in p1["pairs"]}
    seed_summaries = []
    pairs = []
    for seed in SEEDS:
        summary, _rows = summarize_seed(seed, run_root, manifest_root)
        seed_summaries.append(summary)
        reference = references[seed]
        on128 = summary["late_median_map50_95"]
        pairs.append(
            {
                "seed": seed,
                "off": reference["off_late_median_map50_95"],
                "on64": reference["on_late_median_map50_95"],
                "on128": on128,
                "delta_dim": on128 - reference["on_late_median_map50_95"],
                "delta_off": on128 - reference["off_late_median_map50_95"],
            }
        )
    dim_stats = paired_interval([row["delta_dim"] for row in pairs])
    off_stats = paired_interval([row["delta_off"] for row in pairs])
    dim_status, dim_reason = decision(dim_stats["mean_delta"], *dim_stats["confidence_interval_95"])
    off_status, off_reason = decision(off_stats["mean_delta"], *off_stats["confidence_interval_95"])
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    aggregate_csv = RESULTS_ROOT / "d2_v3_p2_align_dim_three_seed_results.csv"
    with aggregate_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    plot_path = RESULTS_ROOT / "d2_v3_p2_align_dim_three_seed_results.png"
    plot_results(pairs, plot_path)
    payload = {
        "schema_version": 1,
        "status": "completed_three_seed_p2_02",
        "claim": "single_variable_align_dim_diagnosis_not_a_revision_of_p1",
        "protocol": {
            "seeds": list(SEEDS),
            "epochs": REQUIRED_EPOCHS,
            "late_window": LATE_WINDOW,
            "paired_metric": "median_last10(metrics/mAP50-95(B))",
            "uses_best_epoch": False,
            "only_training_change": "foundation_align_dim: 64 -> 128",
            "threshold": THRESHOLD,
            "t_critical_95_df2": T_CRITICAL_95_DF2,
            "summarizer_sha256": sha256(Path(__file__)),
        },
        "p1_reference": {"path": P1_RESULT.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(P1_RESULT)},
        "seeds": seed_summaries,
        "pairs": pairs,
        "primary_on128_minus_on64": {"statistics": dim_stats, "decision": dim_status, "reason": dim_reason},
        "secondary_on128_minus_off": {"statistics": off_stats, "decision": off_status, "reason": off_reason},
        "boundaries": {
            "p1_no_go_unchanged": True,
            "p2_01_inconclusive_unchanged": True,
            "no_256_or_512_search_if_no_support": True,
            "inconclusive_requires_more_seeds_not_more_dimensions": True,
            "gradient_probe_is_follow_up_mechanism_evidence": True,
        },
        "artifacts": {
            "aggregate_csv": {"path": aggregate_csv.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(aggregate_csv)},
            "plot": {"path": plot_path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(plot_path)},
        },
    }
    output = RESULTS_ROOT / "d2_v3_p2_align_dim_result.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    """Summarize completed runs under explicit roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(build_summary(args.run_root.resolve(), args.manifest_root.resolve()), indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
