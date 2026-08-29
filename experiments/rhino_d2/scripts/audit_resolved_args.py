#!/usr/bin/env python3
"""Audit the args actually resolved by completed paired D2 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_PAIRS = {
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
CORRECTED_PAIRS = {
    seed: {
        "off": run_dirs["off"],
        "on": REPO_ROOT / f"runs/rhino_d2/p1_corrected/on-calibrated-s{seed}",
    }
    for seed, run_dirs in DEFAULT_PAIRS.items()
}
ALLOWED_DIFFERENCES = {
    "foundation_enabled",
    "foundation_loss_weight",
    "foundation_model",
    "foundation_revision",
    "foundation_teacher",
    "name",
    "project",
    "save_dir",
}


def sha256(path: Path) -> str:
    """Return one file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_args(run_dir: Path) -> tuple[dict, Path]:
    """Load the post-resolution args emitted by the trainer."""
    path = run_dir / "args.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing resolved args: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"resolved args must be a mapping: {path}")
    return payload, path


def audit_pair(seed: int, run_dirs: dict[str, Path]) -> dict:
    """Compare one completed off/on pair after trainer resolution."""
    off, off_path = load_args(run_dirs["off"])
    on, on_path = load_args(run_dirs["on"])
    differences = {
        key: {"off": off.get(key), "on": on.get(key)}
        for key in sorted(set(off) | set(on))
        if off.get(key) != on.get(key)
    }
    unexpected = {key: value for key, value in differences.items() if key not in ALLOWED_DIFFERENCES}
    required_equalities = {
        "seed": off.get("seed") == on.get("seed") == seed,
        "optimizer": off.get("optimizer") == on.get("optimizer"),
        "effective_optimizer": off.get("effective_optimizer") == on.get("effective_optimizer"),
        "effective_optimizer_lrs": off.get("effective_optimizer_lrs") == on.get("effective_optimizer_lrs"),
        "model": off.get("model") == on.get("model"),
        "data": off.get("data") == on.get("data"),
        "epochs": off.get("epochs") == on.get("epochs"),
        "batch": off.get("batch") == on.get("batch"),
        "imgsz": off.get("imgsz") == on.get("imgsz"),
        "pretrained": off.get("pretrained") == on.get("pretrained"),
    }
    passed = not unexpected and all(required_equalities.values())
    return {
        "seed": seed,
        "status": "passed" if passed else "failed",
        "resolved_differences": differences,
        "unexpected_differences": unexpected,
        "required_equalities": required_equalities,
        "files": {
            "off": {"path": str(off_path.relative_to(REPO_ROOT)), "sha256": sha256(off_path)},
            "on": {"path": str(on_path.relative_to(REPO_ROOT)), "sha256": sha256(on_path)},
        },
    }


def build_audit(pairs: dict[int, dict[str, Path]]) -> dict:
    """Audit every paired seed and return one fail-closed report."""
    records = [audit_pair(seed, run_dirs) for seed, run_dirs in pairs.items()]
    passed = all(record["status"] == "passed" for record in records)
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "question": "did off/on actually run with equal resolved non-distillation settings?",
        "allowed_differences": sorted(ALLOWED_DIFFERENCES),
        "records": records,
    }


def main() -> None:
    """Write the resolved-args audit for the completed P1 runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "results" / "d2_p1_resolved_args_audit.json",
    )
    parser.add_argument("--corrected", action="store_true", help="Audit the calibrated-weight treatment runs")
    args = parser.parse_args()
    pairs = CORRECTED_PAIRS if args.corrected else DEFAULT_PAIRS
    if args.corrected and args.output == EXPERIMENT_ROOT / "results" / "d2_p1_resolved_args_audit.json":
        args.output = EXPERIMENT_ROOT / "results" / "d2_p1_corrected_resolved_args_audit.json"
    result = build_audit(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
