"""Evidence contract for the completed train64-only P2-04 alpha calibration."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
RESULTS = EXPERIMENT_ROOT / "results/p2_response_field/calibration"
RESULT = RESULTS / "calibration_result.json"
MANIFEST = RESULTS / "calibration_manifest.json"
SOURCE_COMMIT = "173f4604a8891adda1c5c84dae1558767cb50071"


def sha256(path: Path) -> str:
    """Return one file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_calibration_evidence_is_complete_failed_and_formal_stopped():
    """The archived decision preserves the frozen zero-candidate stop rule."""
    expected = {
        "calibration_raw.csv",
        "calibration_summary.csv",
        "calibration_result.json",
        "calibration_manifest.json",
        "calibration.log",
    }
    assert all((RESULTS / name).is_file() for name in expected)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["selected_alpha"] is None
    assert result["accepted_alpha_count"] == 0
    assert result["accepted_alphas"] == []
    assert result["formal_training_authorized"] is False
    assert result["failure_action"] == "stop_p2_04_no_formal_training"
    pooled = [row for row in result["summaries"] if row["scope"] == "pooled"]
    assert [row["alpha"] for row in pooled] == [0.25, 0.5, 1.0, 2.0, 4.0]
    assert not any(row["accepted"] for row in pooled)
    assert all(row["passing_seed_count"] == 0 for row in pooled)
    assert pooled[0]["r_loss"] == pytest.approx(0.8481793700729783)
    assert pooled[0]["r_grad"] == pytest.approx(4.259234064112473)


def test_calibration_raw_scope_and_machine_audit_are_exact():
    """All authorized observations exist and no update or forbidden evaluation occurred."""
    with (RESULTS / "calibration_raw.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 384
    assert Counter(int(row["seed"]) for row in rows) == {20260824: 128, 20260825: 128, 20260826: 128}
    assert all(count == 16 for count in Counter((row["seed"], row["condition_id"]) for row in rows).values())
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_state"]["commit"] == SOURCE_COMMIT
    assert manifest["source_state"]["calibration_scope_clean"] is True
    assert manifest["execution"]["raw_observations"] == 384
    assert manifest["execution"]["rolling_manifest_records"] == 1536
    assert manifest["execution"]["alpha_forward_reruns"] == 0
    assert manifest["execution"]["optimizer_steps"] == 0
    assert manifest["execution"]["no_validation_access"] is True
    assert manifest["execution"]["no_response128_access"] is True
    assert manifest["execution"]["formal_training_started"] is False
    for checkpoint in manifest["checkpoints"].values():
        assert checkpoint["checkpoint_sha256_before"] == checkpoint["checkpoint_sha256_after"]
        assert checkpoint["state_sha256_before"] == checkpoint["state_sha256_after"]
        assert checkpoint["changed_state_keys"] == []
        assert checkpoint["parameter_grads_remain_none"] is True
        assert checkpoint["optimizer_steps"] == 0


def test_calibration_artifact_hashes_and_forbidden_source_access_are_bound():
    """The manifest binds every result and the implementation contains no forbidden data path."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, record in manifest["artifacts"].items():
        assert sha256(RESULTS / name) == record["sha256"]
    assert (
        sha256(REPO_ROOT / "ultralytics/nn/foundation/response.py")
        == manifest["implementation"]["response.py"]["sha256"]
    )
    script = REPO_ROOT / "experiments/rhino_d2/scripts/calibrate_response_field_alpha.py"
    assert sha256(script) == manifest["implementation"]["calibration_script"]["sha256"]
    source = script.read_text(encoding="utf-8")
    assert "diagnostic_response128" not in source
    assert "DetectionValidator" not in source
    assert "optimizer.step" not in source
    assert "scheduler.step" not in source
