#!/usr/bin/env python3
"""Assess the pre-registered DINOv3 OFF-only baseline engineering sanity gate."""

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
DEFAULT_RUN = REPO_ROOT / "runs/rhino_d2/v3_baseline_sanity/v3-baseline-sanity-s20260824"


def sha256(path: Path) -> str:
    """Return one file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, float]]:
    """Read all numeric training rows using the standard library only."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [{key: float(value) for key, value in row.items() if value != ""} for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def main() -> None:
    """Evaluate a robust late-window gate without selecting the best validation epoch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--late-window", type=int, default=10)
    parser.add_argument("--map-threshold", type=float, default=0.01)
    parser.add_argument("--loss-retention-max", type=float, default=0.90)
    args = parser.parse_args()
    if args.late_window <= 0 or args.map_threshold <= 0 or not 0 < args.loss_retention_max < 1:
        parser.error("gate values must be positive and loss-retention-max must be below 1")
    run_dir = args.run_dir.resolve()
    if REPO_ROOT not in run_dir.parents:
        raise ValueError("run directory must stay inside the repository")
    results_path = run_dir / "results.csv"
    args_path = run_dir / "args.yaml"
    runtime_manifest_path = REPO_ROOT / "runs/rhino_d2/manifests" / f"{run_dir.name}.json"
    complete_log_path = REPO_ROOT / "runs/rhino_d2/logs" / f"{run_dir.name}.log"
    for required in (results_path, args_path, runtime_manifest_path, complete_log_path):
        if not required.is_file():
            raise FileNotFoundError(f"required baseline artifact is missing: {required}")
    rows = read_rows(results_path)
    if len(rows) < args.late_window:
        raise ValueError(f"need at least {args.late_window} epochs, found {len(rows)}")
    late = rows[-args.late_window :]
    loss_key = ("train/box_loss", "train/cls_loss", "train/dfl_loss")
    initial_loss = sum(rows[0][key] for key in loss_key)
    final_loss = sum(rows[-1][key] for key in loss_key)
    late_median_map = statistics.median(row["metrics/mAP50-95(B)"] for row in late)
    late_median_precision = statistics.median(row["metrics/precision(B)"] for row in late)
    late_median_recall = statistics.median(row["metrics/recall(B)"] for row in late)
    checks = {
        "completed_50_epochs": len(rows) == 50,
        "late_median_map_at_least_0p01": late_median_map >= args.map_threshold,
        "late_median_precision_nonzero": late_median_precision > 0,
        "late_median_recall_nonzero": late_median_recall > 0,
        "final_detection_loss_down_at_least_10pct": final_loss / initial_loss <= args.loss_retention_max,
    }
    artifact_prefix = f"d2_{run_dir.name.replace('-', '_')}"
    archive_paths = {
        "results": EXPERIMENT_ROOT / "results" / f"{artifact_prefix}.csv",
        "args": EXPERIMENT_ROOT / "results" / f"{artifact_prefix}.args.yaml",
        "complete_log": EXPERIMENT_ROOT / "results" / f"{artifact_prefix}.log",
        "runtime_manifest": EXPERIMENT_ROOT / "results" / f"{artifact_prefix}.manifest.json",
    }
    (EXPERIMENT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    for source, destination in zip(
        (results_path, args_path, complete_log_path, runtime_manifest_path), archive_paths.values(), strict=True
    ):
        shutil.copy2(source, destination)
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 2,
        "status": "passed" if all(checks.values()) else "failed",
        "claim": "engineering_pipeline_sanity_gate_not_foundation_efficacy",
        "pre_registered_gate": {
            "late_window": args.late_window,
            "late_median_map50_95_threshold": args.map_threshold,
            "late_median_precision_must_be_nonzero": True,
            "late_median_recall_must_be_nonzero": True,
            "final_detection_loss_retention_max": args.loss_retention_max,
            "uses_best_epoch": False,
        },
        "checks": checks,
        "observations": {
            "epochs": len(rows),
            "late_median_map50_95": late_median_map,
            "late_median_precision": late_median_precision,
            "late_median_recall": late_median_recall,
            "initial_detection_loss": initial_loss,
            "final_detection_loss": final_loss,
            "loss_retention": final_loss / initial_loss,
            "final_map50_95": rows[-1]["metrics/mAP50-95(B)"],
        },
        "artifacts": {
            name: {"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(path)}
            for name, path in archive_paths.items()
        },
        "runtime": {
            "source_commit": runtime_manifest["source_state"]["commit"],
            "experiment_inputs_dirty": runtime_manifest["source_state"]["experiment_inputs_dirty"],
            "returncode": runtime_manifest["returncode"],
            "checkpoint_count": len(runtime_manifest["checkpoints"]),
            "config_sha256": runtime_manifest["config_sha256"],
            "runner_sha256": runtime_manifest["runner_sha256"],
        },
        "decision": {
            "formal_p1_unlocked": all(checks.values()),
            "next_action": (
                "freeze_formal_p1_protocol"
                if all(checks.values())
                else "improve_baseline_task_learning_signal_as_one_controlled_protocol_change"
            ),
            "prohibited_until_gate_passes": [
                "formal_off_vs_dinov3_s_training",
                "validation_ap_based_kd_weight_selection",
                "dinov3_l_capacity_experiment",
                "new_loss_or_multistage_expansion",
            ],
        },
    }
    output = EXPERIMENT_ROOT / "results" / "d2_v3_baseline_sanity.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
