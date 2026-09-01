#!/usr/bin/env python3
"""Qualify Candidate B partial Student initialization before any training run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils.torch_utils import intersect_dicts

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_WEIGHTS = EXPERIMENT_ROOT / "cache" / "yolo26n.pt"
DEFAULT_TARGET = REPO_ROOT / "ultralytics/cfg/models/26/yolo26-master-n.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results" / "d2_v3_student_init_audit.json"
SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
SOURCE_RELEASE = "ultralytics/assets v8.4.0"
SOURCE_LICENSE = "AGPL-3.0"
EXPECTED_BYTES = 5_544_453
EXPECTED_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"

# Frozen before Candidate B training. These are qualification thresholds, not efficacy thresholds.
MIN_TARGET_PARAMETER_COVERAGE = 0.40
MIN_SOURCE_PARAMETER_COVERAGE = 0.80
GROUPS = {
    "stem": tuple(f"model.{index}." for index in (0, 1, 2, 3)),
    "moe_backbone": tuple(f"model.{index}." for index in (4, 6, 8)),
    "shared_deep_backbone": tuple(f"model.{index}." for index in (5, 7, 9, 10)),
    "head": tuple(f"model.{index}." for index in range(11, 24)),
}


def sha256(path: Path) -> str:
    """Return one file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numel(values) -> int:
    """Count scalar parameters in an iterable of tensors."""
    return sum(value.numel() for value in values)


def main() -> None:
    """Measure same-name/same-shape transfer and fail closed on weak coverage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    weights = args.weights.resolve()
    target = args.target.resolve()
    output = args.output.resolve()

    source_model, _ = load_checkpoint(weights, device="cpu")
    target_model = DetectionModel(str(target), nc=80, verbose=False)
    source_state = source_model.float().state_dict()
    target_state = target_model.state_dict()
    matched_state = intersect_dicts(source_state, target_state)
    source_parameters = dict(source_model.named_parameters())
    target_parameters = dict(target_model.named_parameters())
    matched_parameter_keys = sorted(key for key in matched_state if key in target_parameters)
    matched_parameter_numel = numel(target_parameters[key] for key in matched_parameter_keys)
    source_parameter_numel = numel(source_parameters.values())
    target_parameter_numel = numel(target_parameters.values())

    group_results = {}
    for name, prefixes in GROUPS.items():
        keys = sorted(key for key in target_parameters if key.startswith(prefixes))
        matched = [key for key in keys if key in matched_state]
        target_group_numel = numel(target_parameters[key] for key in keys)
        matched_group_numel = numel(target_parameters[key] for key in matched)
        group_results[name] = {
            "matched_parameter_keys": len(matched),
            "target_parameter_keys": len(keys),
            "matched_parameter_numel": matched_group_numel,
            "target_parameter_numel": target_group_numel,
            "coverage": matched_group_numel / target_group_numel,
        }

    target_coverage = matched_parameter_numel / target_parameter_numel
    source_coverage = matched_parameter_numel / source_parameter_numel
    checks = {
        "asset_size_matches": weights.stat().st_size == EXPECTED_BYTES,
        "asset_sha256_matches": sha256(weights) == EXPECTED_SHA256,
        "target_parameter_coverage_at_least_0p40": target_coverage >= MIN_TARGET_PARAMETER_COVERAGE,
        "source_parameter_coverage_at_least_0p80": source_coverage >= MIN_SOURCE_PARAMETER_COVERAGE,
        "stem_fully_transferred": group_results["stem"]["coverage"] == 1.0,
        "shared_deep_backbone_fully_transferred": group_results["shared_deep_backbone"]["coverage"] == 1.0,
        "head_fully_transferred": group_results["head"]["coverage"] == 1.0,
    }
    result = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "claim": "candidate_b_initialization_compatibility_not_detection_or_kd_efficacy",
        "source": {
            "name": "Ultralytics YOLO26n",
            "release": SOURCE_RELEASE,
            "url": SOURCE_URL,
            "license": SOURCE_LICENSE,
            "bytes": weights.stat().st_size,
            "sha256": sha256(weights),
            "payload_committed": False,
        },
        "target": {
            "name": "YOLO-Master-N",
            "config": str(target.relative_to(REPO_ROOT)).replace("\\", "/"),
            "config_sha256": sha256(target),
            "architecture_replaced": False,
        },
        "pre_registered_qualification": {
            "minimum_target_parameter_coverage": MIN_TARGET_PARAMETER_COVERAGE,
            "minimum_source_parameter_coverage": MIN_SOURCE_PARAMETER_COVERAGE,
            "stem_must_be_fully_transferred": True,
            "shared_deep_backbone_must_be_fully_transferred": True,
            "head_must_be_fully_transferred": True,
        },
        "observations": {
            "matched_state_items": len(matched_state),
            "target_state_items": len(target_state),
            "matched_parameter_keys": len(matched_parameter_keys),
            "target_parameter_keys": len(target_parameters),
            "matched_parameter_numel": matched_parameter_numel,
            "source_parameter_numel": source_parameter_numel,
            "target_parameter_numel": target_parameter_numel,
            "source_parameter_coverage": source_coverage,
            "target_parameter_coverage": target_coverage,
            "matched_parameter_keyset_sha256": hashlib.sha256("\n".join(matched_parameter_keys).encode()).hexdigest(),
            "groups": group_results,
        },
        "checks": checks,
        "decision": {
            "candidate_b_training_allowed": all(checks.values()),
            "formal_p1_unlocked": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
