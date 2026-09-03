"""Evidence contract for the synthetic-only P2-04 Response-Field gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
RESULT = EXPERIMENT_ROOT / "results/p2_response_field/smoke/d2_v3_p2_response_field_smoke.json"


def sha256(path: Path) -> str:
    """Return one file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_response_field_smoke_is_clean_synthetic_evidence_not_calibration():
    """The archived gate must bind clean inputs and make no efficacy claim."""
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "passed_synthetic_implementation_gate"
    assert result["authorization_boundary"] == "synthetic_only_no_alpha_calibration_no_formal_training"
    assert result["source_state"]["commit"] == "ead45e35aee4ac4b70a6258b7e620f25422d2907"
    assert result["source_state"]["experiment_inputs_dirty"] is False
    assert result["source_state"]["experiment_input_status"] == []
    for relative_path, expected_hash in result["source_state"]["input_sha256"].items():
        assert sha256(REPO_ROOT / relative_path) == expected_hash
    assert all(result["checks"].values())
    assert result["checks"]["no_calibration_executed"] is True
    assert result["checks"]["no_validation_or_formal_metrics_read"] is True


def test_response_field_smoke_records_expected_gradient_and_bn_boundaries():
    """Archived arm records must prove the intended null/supervised branch behavior."""
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["arms"]["A"]["perturbed_feature_gradient_norm"] == 0.0
    assert result["arms"]["A"]["projector_gradient_norm"] == 0.0
    for arm in ("B", "C"):
        assert result["arms"][arm]["perturbed_feature_gradient_norm"] > 0
        assert result["arms"][arm]["projector_gradient_norm"] > 0
    assert all(record["bn_buffers_restored_bitwise"] for record in result["arms"].values())
    assert all(record["bn_training_flags_match_and_true"] for record in result["arms"].values())
    assert result["optimization"]["first"] == pytest.approx(2.1094279289245605)
    assert result["optimization"]["last"] == pytest.approx(1.1397250890731812)
    assert result["optimization"]["last"] < result["optimization"]["first"]
