#!/usr/bin/env python3
"""Fail closed unless formal DINOv3-S OFF/ON differ only by the KD treatment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
OFF = EXPERIMENT_ROOT / "configs/d2_v3_p1_off.yaml"
ON = EXPERIMENT_ROOT / "configs/d2_v3_p1_on.yaml"
CALIBRATION = EXPERIMENT_ROOT / "results/d2_v3_p1_weight_calibration_extension.json"
OUTPUT = EXPERIMENT_ROOT / "results/d2_v3_p1_pair_validation.json"
ALLOWED_DIFFERENCES = {"foundation_enabled", "foundation_loss_weight", "name"}


def sha256(path: Path) -> str:
    """Return one file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict:
    """Load one mapping config."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a mapping")
    return value


def main() -> None:
    """Bind configs to calibration evidence and reject every uncontrolled difference."""
    off, on = load_yaml(OFF), load_yaml(ON)
    all_keys = sorted(set(off) | set(on))
    differences = {key: {"off": off.get(key), "on": on.get(key)} for key in all_keys if off.get(key) != on.get(key)}
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    checks = {
        "differences_exactly_treatment_and_name": set(differences) == ALLOWED_DIFFERENCES,
        "calibration_selected": calibration["status"] == "selected",
        "calibration_used_no_validation_metric": calibration["correction_contract"]["uses_validation_metric"] is False,
        "on_weight_matches_calibration": on["foundation_loss_weight"] == calibration["selected_weight"] == 0.15,
        "same_student_initialization": off["pretrained"] == on["pretrained"],
        "same_data": off["data"] == on["data"],
        "same_teacher_contract": all(
            off[key] == on[key]
            for key in (
                "foundation_teacher",
                "foundation_model",
                "foundation_weights",
                "foundation_teacher_dtype",
                "foundation_target_levels",
                "foundation_align_dim",
                "foundation_loss",
            )
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "claim": "formal_p1_controlled_pair_contract_not_kd_efficacy",
        "allowed_differences": sorted(ALLOWED_DIFFERENCES),
        "observed_differences": differences,
        "checks": checks,
        "files": {
            "off": {"path": str(OFF.relative_to(REPO_ROOT)), "sha256": sha256(OFF)},
            "on": {"path": str(ON.relative_to(REPO_ROOT)), "sha256": sha256(ON)},
            "calibration": {"path": str(CALIBRATION.relative_to(REPO_ROOT)), "sha256": sha256(CALIBRATION)},
        },
        "decision": {"formal_three_seed_training_allowed": all(checks.values())},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
