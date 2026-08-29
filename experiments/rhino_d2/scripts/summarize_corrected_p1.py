#!/usr/bin/env python3
"""Summarize the corrected D2 weight-calibrated three-seed experiment."""

from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
from pathlib import Path

from summarize_p1 import (
    REPO_ROOT,
    RESULTS_ROOT,
    T_CRITICAL_95_DF2,
    THRESHOLD,
    _decision,
    plot_multiseed,
    read_rows,
    sha256,
    summarize_arm,
)

RUNS = {
    20260824: {
        "off": REPO_ROOT / "runs/rhino_d2/p1/off-s20260824-2",
        "on-calibrated": REPO_ROOT / "runs/rhino_d2/p1_corrected/on-calibrated-s20260824",
    },
    20260825: {
        "off": REPO_ROOT / "runs/rhino_d2/p1/off-s20260825",
        "on-calibrated": REPO_ROOT / "runs/rhino_d2/p1_corrected/on-calibrated-s20260825",
    },
    20260826: {
        "off": REPO_ROOT / "runs/rhino_d2/p1/off-s20260826",
        "on-calibrated": REPO_ROOT / "runs/rhino_d2/p1_corrected/on-calibrated-s20260826",
    },
}


def build_summary() -> dict:
    """Build paired statistics, checkpoint verification, and provenance."""
    arms, pairs, artifacts, runtime_records = {}, [], {}, []
    for seed, run_dirs in RUNS.items():
        arms[str(seed)] = {}
        artifacts[str(seed)] = {}
        for arm, run_dir in run_dirs.items():
            summary, _ = summarize_arm(arm, run_dir)
            arms[str(seed)][arm] = summary
            output_csv = RESULTS_ROOT / f"d2_p1_corrected_{arm}_s{seed}.csv"
            shutil.copyfile(run_dir / "results.csv", output_csv)
            artifacts[str(seed)][arm] = {
                "path": str(output_csv.relative_to(REPO_ROOT)),
                "sha256": sha256(output_csv),
            }
        off, on = arms[str(seed)]["off"], arms[str(seed)]["on-calibrated"]
        pairs.append(
            {
                "seed": seed,
                "off_best_epoch": off["best_epoch"],
                "on_best_epoch": on["best_epoch"],
                "off_best_map50_95": off["best_map50_95"],
                "on_best_map50_95": on["best_map50_95"],
                "delta": on["best_map50_95"] - off["best_map50_95"],
                "on_mean_foundation_task_ratio": statistics.fmean(
                    row["train/foundation_task_ratio"] for row in read_rows(run_dirs["on-calibrated"])
                ),
            }
        )
        manifest_path = REPO_ROOT / f"runs/rhino_d2/manifests/on-calibrated-s{seed}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        log_path = Path(manifest["log"])
        runtime_records.append(
            {
                "seed": seed,
                "returncode": manifest["returncode"],
                "source_state": manifest["source_state"],
                "config_sha256": manifest["config_sha256"],
                "log": str(log_path.relative_to(REPO_ROOT)),
                "log_sha256": manifest["log_sha256"],
                "log_hash_verified": sha256(log_path) == manifest["log_sha256"],
            }
        )

    deltas = [pair["delta"] for pair in pairs]
    mean_delta = statistics.fmean(deltas)
    sample_std = statistics.stdev(deltas)
    standard_error = sample_std / math.sqrt(len(deltas))
    half_width = T_CRITICAL_95_DF2 * standard_error
    ci_low, ci_high = mean_delta - half_width, mean_delta + half_width
    status, reason = _decision(mean_delta, ci_low, ci_high)
    aggregate_csv = RESULTS_ROOT / "d2_p1_corrected_results.csv"
    with aggregate_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    curve_path = RESULTS_ROOT / "d2_p1_corrected_results.png"
    plot_multiseed(pairs, curve_path, title="D2 P1 corrected weight 0.10 — three paired seeds")
    original = json.loads((RESULTS_ROOT / "d2_p1_three_seed_results.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "completed_corrected_three_seed_pair",
        "claim": "weight_calibrated_protocol_decision_not_a_general_foundation_kd_claim",
        "correction": {
            "problem": "original weight 0.05 produced only about 1.7% foundation/task ratio",
            "selection_used_validation_ap": False,
            "selected_weight": 0.1,
            "calibration_evidence": "experiments/rhino_d2/results/d2_p1_weight_calibration.json",
            "baseline_reuse_justification": "resolved args audit verified identical non-distillation settings",
        },
        "protocol": {"seeds": list(RUNS), "epochs": 10, "imgsz": 256, "batch": 4, "pretrained": False},
        "pairs": pairs,
        "arms": arms,
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
            "pre_registered_threshold": THRESHOLD,
            "rule_unchanged_after_observing_results": True,
        },
        "comparison_to_original": {
            "original_weight": 0.05,
            "original_mean_delta": original["statistics"]["mean_delta"],
            "corrected_weight": 0.1,
            "corrected_mean_delta": mean_delta,
            "interpretation": "weight-too-small explanation is weakened only if corrected CI still contains zero",
        },
        "artifacts": {
            "per_run_csv": artifacts,
            "aggregate_csv": {"path": str(aggregate_csv.relative_to(REPO_ROOT)), "sha256": sha256(aggregate_csv)},
            "curve": {"path": str(curve_path.relative_to(REPO_ROOT)), "sha256": sha256(curve_path)},
            "runtime_records": runtime_records,
        },
    }


def main() -> None:
    """Persist the corrected experiment report."""
    result = build_summary()
    output = RESULTS_ROOT / "d2_p1_corrected_results.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
