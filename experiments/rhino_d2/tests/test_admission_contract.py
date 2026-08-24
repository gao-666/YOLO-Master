"""Contracts for the Rhino-Bird D2 admission package."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_on_off_configs_have_no_uncontrolled_differences():
    validator = _load_script("validate_pair.py")
    off = yaml.safe_load((ROOT / "configs" / "d2_off.yaml").read_text(encoding="utf-8"))
    on = yaml.safe_load((ROOT / "configs" / "d2_on.yaml").read_text(encoding="utf-8"))
    result = validator.compare_configs(off, on)
    assert result["status"] == "passed"
    assert not result["unexpected_differences"]


def test_single_batch_projector_loss_decreases_offline():
    smoke = _load_script("d2_alignment_smoke.py")
    torch.manual_seed(20260824)
    student = torch.randn(1, 8, 4, 4)
    teacher = torch.randn(1, 12, 7, 7)
    projector, history = smoke.optimize_alignment(
        student,
        teacher,
        align_dim=8,
        steps=12,
        learning_rate=0.05,
    )
    assert history[-1] < history[0]
    assert projector.alignment["teacher_resized"] is True
    assert projector.teacher_projection_frozen


def test_real_p0_task_plus_kd_evidence_passes_every_redline():
    evidence = json.loads((ROOT / "results" / "d2_p0_train_smoke.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "passed"
    assert evidence["claim"] == "p0_real_yolo_loss_synthetic_target_fixed_batch_smoke_no_accuracy_claim"
    assert all(evidence["checks"].values())
    assert evidence["data_contract"]["target_source"] == "deterministic_synthetic"
    assert evidence["data_contract"]["purpose"] == "gradient_chain_smoke_not_accuracy_evaluation"
    assert evidence["teacher"]["metadata"]["preprocessing"]["contract"] == "dinov2_dense_spatial_preserving_v1"


def test_environment_manifest_tracks_commit_dirty_state_and_input_hashes():
    evidence = json.loads((ROOT / "env" / "environment.json").read_text(encoding="utf-8"))
    git = evidence["git"]
    assert evidence["schema_version"] == 2
    assert git["base_commit"]
    assert git["experiment_commit"]
    assert isinstance(git["repository_state"]["dirty"], bool)
    assert git["experiment_inputs_state"]["dirty"] is False
    assert "experiments\\rhino_d2\\scripts\\d2_p0_train_smoke.py" in evidence["scripts"]
    assert "ultralytics\\nn\\foundation\\teachers\\dinov2.py" in evidence["implementation"]


def test_loss_curve_artifacts_match_source_evidence():
    manifest = json.loads((ROOT / "results" / "d2_p0_loss_curves_manifest.json").read_text(encoding="utf-8"))
    p0 = json.loads((ROOT / "results" / "d2_p0_train_smoke.json").read_text(encoding="utf-8"))
    alignment = json.loads((ROOT / "results" / "d2_alignment_smoke.json").read_text(encoding="utf-8"))
    with (ROOT / "results" / "d2_p0_loss_curves.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert manifest["claim"] == "visualization_of_committed_p0_smoke_evidence_no_accuracy_claim"
    assert manifest["points"] == len(rows) == 36
    assert manifest["summary"]["single_stage"] == "p4"
    assert manifest["summary"]["integrated_kd_initial"] == p0["steps"][0]["foundation_loss"]
    assert manifest["summary"]["integrated_kd_final"] == p0["steps"][-1]["foundation_loss"]
    assert manifest["summary"]["alignment_final"] == alignment["optimization"]["history"][-1]
    for relative_path, expected in manifest["sources"].items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected
    for relative_path, expected in manifest["outputs"].items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected
    assert (ROOT / "results" / "d2_p0_loss_curves.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
