#!/usr/bin/env python3
"""Calibrate DINOv3-S KD weight from training signal only, without validation AP."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
from run_p1 import EXPERIMENT_ROOT, REPO_ROOT, git_state, resolve_yolo, sha256

CONFIG = EXPERIMENT_ROOT / "configs" / "d2_v3_p1_calibration.yaml"
DEFAULT_WEIGHTS = (0.01, 0.025, 0.05, 0.10)


def slug(weight: float) -> str:
    """Build a filesystem-safe weight label."""
    return f"{weight:g}".replace(".", "p")


def read_last_row(path: Path) -> dict[str, float]:
    """Read the final training-only row."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return {key: float(value) for key, value in rows[-1].items() if value != ""}


def run_candidate(weight: float, seed: int, project: Path) -> dict:
    """Run one identical one-epoch candidate with validation disabled."""
    name = f"v3-calibration-w{slug(weight)}-s{seed}"
    run_dir = project / name
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite calibration run: {run_dir}")
    log_dir = REPO_ROOT / "runs/rhino_d2/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    command = [
        resolve_yolo(),
        f"cfg={CONFIG.relative_to(REPO_ROOT)}",
        f"foundation_loss_weight={weight}",
        f"seed={seed}",
        f"name={name}",
        f"project={project}",
        "exist_ok=False",
    ]
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / "runs/rhino_d2/config"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTHONUTF8", "1")
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        returncode = process.wait()
    if returncode:
        raise SystemExit(returncode)
    row = read_last_row(run_dir / "results.csv")
    batch = int(yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["batch"])
    reconstructed = row["train/foundation_cosine_raw"] * weight * batch
    observed = row["train/foundation_loss"]
    return {
        "weight": weight,
        "seed": seed,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "args_sha256": sha256(run_dir / "args.yaml"),
        "results_sha256": sha256(run_dir / "results.csv"),
        "log": str(log_path.relative_to(REPO_ROOT)),
        "log_sha256": sha256(log_path),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "uses_validation": False,
        "foundation_task_ratio": row["train/foundation_task_ratio"],
        "foundation_loss": observed,
        "task_loss": observed / row["train/foundation_task_ratio"],
        "mechanism_identity": {
            "formula": "cosine_raw * weight * batch_size = foundation_loss",
            "reconstructed": reconstructed,
            "observed": observed,
            "absolute_error": abs(reconstructed - observed),
            "passed": abs(reconstructed - observed) < 1e-5,
        },
    }


def main() -> None:
    """Run the frozen candidates and select the smallest in-band weight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--target-low", type=float, default=0.03)
    parser.add_argument("--target-high", type=float, default=0.06)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs/rhino_d2/v3_p1_weight_calibration")
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "results/d2_v3_p1_weight_calibration.json")
    args = parser.parse_args()
    if not 0 < args.target_low < args.target_high < 1:
        parser.error("target band must satisfy 0 < low < high < 1")
    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    records = [run_candidate(weight, args.seed, project) for weight in DEFAULT_WEIGHTS]
    eligible = [
        record["weight"]
        for record in records
        if args.target_low <= record["foundation_task_ratio"] <= args.target_high
        and record["mechanism_identity"]["passed"]
    ]
    selected = min(eligible) if eligible else None
    payload = {
        "schema_version": 1,
        "status": "selected" if selected is not None else "no_candidate_in_band",
        "claim": "train_only_signal_calibration_not_detection_or_kd_efficacy",
        "selection_contract": {
            "uses_validation_metric": False,
            "epochs": 1,
            "same_seed_initialization_data_and_order": True,
            "candidate_weights": list(DEFAULT_WEIGHTS),
            "target_foundation_task_ratio": [args.target_low, args.target_high],
            "rule": "smallest candidate inside the frozen ratio band with mechanism identity passing",
        },
        "source_state": git_state(),
        "config": {"path": str(CONFIG.relative_to(REPO_ROOT)), "sha256": sha256(CONFIG)},
        "records": records,
        "selected_weight": selected,
        "formal_p1_training_allowed": selected is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
