#!/usr/bin/env python3
"""Archive and summarize the frozen DINOv3-S formal P1 paired runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
            "status": "pending_more_seeds",
            "reason": "one paired seed cannot estimate the pre-registered paired 95% confidence interval",
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
    args = parser.parse_args()
    result = build_summary(args.seed, args.run_root.resolve(), args.manifest_root.resolve())
    stem = "d2_v3_p1_first_pair" if args.seed == 20260824 else f"d2_v3_p1_pair_s{args.seed}"
    output = RESULTS_ROOT / f"{stem}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
