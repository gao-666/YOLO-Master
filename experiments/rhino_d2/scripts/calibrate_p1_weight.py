#!/usr/bin/env python3
"""Calibrate D2 KD weight from training-signal ratios without consulting validation AP."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from run_p1 import EXPERIMENT_ROOT, REPO_ROOT, git_state, resolve_yolo, sha256

CONFIG = EXPERIMENT_ROOT / "configs" / "d2_on.yaml"


def parse_weights(value: str) -> list[float]:
    """Parse a non-empty list of positive candidate weights."""
    try:
        weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weights must contain numbers") from exc
    if not weights or any(weight <= 0 for weight in weights):
        raise argparse.ArgumentTypeError("weights must be positive")
    return weights


def slug(weight: float) -> str:
    """Build a filesystem-safe weight label."""
    return f"{weight:g}".replace(".", "p")


def read_last_row(path: Path) -> dict[str, float]:
    """Read the final numeric training row."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no training rows in {path}")
    return {key: float(value) for key, value in rows[-1].items() if value != ""}


def run_candidate(weight: float, seed: int, project: Path) -> dict:
    """Run one same-initialization, one-epoch calibration candidate."""
    name = f"calibration-w{slug(weight)}-s{seed}"
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
        "epochs=1",
        "val=False",
        "save=False",
        f"seed={seed}",
        f"name={name}",
        f"project={project}",
        "exist_ok=False",
    ]
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
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
    batch = 4
    reconstructed = (row["train/foundation_cosine_raw"] + row["train/foundation_relational_raw"]) * weight * batch
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
        "foundation_task_ratio": row["train/foundation_task_ratio"],
        "foundation_loss": observed,
        "mixture_aux_loss": row["train/mixture_aux_loss"],
        "foundation_to_mixture_ratio": observed / row["train/mixture_aux_loss"],
        "mechanism_identity": {
            "formula": "(cosine_raw + relational_raw) * weight * batch_size = foundation_loss",
            "reconstructed": reconstructed,
            "observed": observed,
            "absolute_error": abs(reconstructed - observed),
            "passed": abs(reconstructed - observed) < 1e-5,
        },
    }


def choose_weight(records: list[dict], target_low: float, target_high: float) -> float | None:
    """Choose the smallest candidate inside the pre-registered task-ratio band."""
    eligible = [
        record["weight"]
        for record in records
        if target_low <= record["foundation_task_ratio"] <= target_high and record["mechanism_identity"]["passed"]
    ]
    return min(eligible) if eligible else None


def main() -> None:
    """Run candidates and persist the no-AP calibration decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=parse_weights, default=[0.01, 0.05, 0.1, 0.15])
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--target-low", type=float, default=0.03)
    parser.add_argument("--target-high", type=float, default=0.06)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs/rhino_d2/weight_calibration")
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "results" / "d2_p1_weight_calibration.json",
    )
    args = parser.parse_args()
    if not 0 < args.target_low < args.target_high < 1:
        parser.error("target band must satisfy 0 < low < high < 1")
    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    records = [run_candidate(weight, args.seed, project) for weight in args.weights]
    selected = choose_weight(records, args.target_low, args.target_high)
    payload = {
        "schema_version": 1,
        "status": "selected" if selected is not None else "no_candidate_in_band",
        "question": "what is the smallest KD weight producing a visible but non-dominant training signal?",
        "selection_contract": {
            "uses_validation_metric": False,
            "epochs": 1,
            "same_seed_and_initialization": True,
            "target_foundation_task_ratio": [args.target_low, args.target_high],
            "rule": "smallest candidate inside the frozen ratio band with mechanism identity passing",
        },
        "source_state": git_state(),
        "config": {"path": str(CONFIG.relative_to(REPO_ROOT)), "sha256": sha256(CONFIG)},
        "records": records,
        "selected_weight": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
