#!/usr/bin/env python3
"""Run the single pre-registered 0.15 DINOv3-S calibration extension."""

from __future__ import annotations

import json

from calibrate_v3_p1_weight import CONFIG, EXPERIMENT_ROOT, REPO_ROOT, git_state, run_candidate, sha256

PREVIOUS = EXPERIMENT_ROOT / "results/d2_v3_p1_weight_calibration.json"
OUTPUT = EXPERIMENT_ROOT / "results/d2_v3_p1_weight_calibration_extension.json"
WEIGHT = 0.15
TARGET_LOW = 0.03
TARGET_HIGH = 0.06
SEED = 20260824


def main() -> None:
    """Validate the prior stop, run one candidate, and fail closed if it misses the unchanged band."""
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    if previous["status"] != "no_candidate_in_band" or previous["selected_weight"] is not None:
        raise RuntimeError("extension requires the committed no-candidate first-stage result")
    if previous["selection_contract"]["uses_validation_metric"] is not False:
        raise RuntimeError("prior calibration must not use validation metrics")

    project = REPO_ROOT / "runs/rhino_d2/v3_p1_weight_calibration_extension"
    project.mkdir(parents=True, exist_ok=True)
    record = run_candidate(WEIGHT, SEED, project)
    eligible = TARGET_LOW <= record["foundation_task_ratio"] <= TARGET_HIGH and record["mechanism_identity"]["passed"]
    payload = {
        "schema_version": 1,
        "status": "selected" if eligible else "extension_failed",
        "claim": "train_only_single_candidate_protocol_correction_not_detection_or_kd_efficacy",
        "correction_contract": {
            "reason": "0.10 reached 0.0263252; one linear-extrapolation candidate is allowed",
            "uses_validation_metric": False,
            "candidate_weights": [WEIGHT],
            "target_foundation_task_ratio": [TARGET_LOW, TARGET_HIGH],
            "predicted_ratio_from_weight_0p10": 0.0263252 * (WEIGHT / 0.10),
            "no_further_candidates_if_failed": True,
        },
        "source_state": git_state(),
        "config": {"path": str(CONFIG.relative_to(REPO_ROOT)), "sha256": sha256(CONFIG)},
        "previous_result": {"path": str(PREVIOUS.relative_to(REPO_ROOT)), "sha256": sha256(PREVIOUS)},
        "record": record,
        "selected_weight": WEIGHT if eligible else None,
        "formal_p1_training_allowed": eligible,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
