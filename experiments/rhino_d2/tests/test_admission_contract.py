"""Contracts for the Rhino-Bird D2 admission package."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
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


def test_p1_configs_preserve_epoch_checkpoints_and_isolate_relational_ablation():
    """The key ablation must change one mechanism while preserving the complete P1 budget."""
    off = yaml.safe_load((ROOT / "configs" / "d2_off.yaml").read_text(encoding="utf-8"))
    on = yaml.safe_load((ROOT / "configs" / "d2_on.yaml").read_text(encoding="utf-8"))
    cosine = yaml.safe_load((ROOT / "configs" / "d2_ablation_cosine_only.yaml").read_text(encoding="utf-8"))
    assert off["save_period"] == on["save_period"] == cosine["save_period"] == 1
    differences = {key for key in set(on) | set(cosine) if on.get(key) != cosine.get(key)}
    assert differences == {"name", "foundation_relation_weight"}
    assert on["foundation_relation_weight"] == 1.0
    assert cosine["foundation_relation_weight"] == 0.0


def test_p1_first_result_evidence_is_conservative_and_self_consistent():
    """Single-seed evidence must not be promoted to a formal go/no-go decision."""
    evidence = json.loads((ROOT / "results" / "d2_p1_first_results.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "completed_single_seed_first_pair_and_key_ablation"
    assert evidence["decision"]["status"] == "pending_more_seeds"
    assert evidence["decision"]["confidence_interval"] is None
    assert evidence["comparisons"]["absolute_delta_below_0_003"] is True
    assert evidence["arms"]["on"]["best_map50_95"] > evidence["arms"]["off"]["best_map50_95"]
    assert evidence["arms"]["on"]["best_map50_95"] > evidence["arms"]["cosine-only"]["best_map50_95"]
    assert all(arm["best_checkpoint"]["load_verified"] for arm in evidence["arms"].values())
    assert all(arm["healthy_checkpoint"]["load_verified"] for arm in evidence["arms"].values())
    for artifact in evidence["artifacts"]["csv"].values():
        path = ROOT.parents[1] / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    curve = evidence["artifacts"]["curve"]
    curve_path = ROOT.parents[1] / curve["path"]
    assert hashlib.sha256(curve_path.read_bytes()).hexdigest() == curve["sha256"]
    assert curve_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_p1_runner_builds_same_budget_seeded_commands(tmp_path):
    """The runner should apply the same seed/project override to both paired arms."""
    runner = _load_script("run_p1.py")
    off, off_name = runner.build_command("yolo", "off", 20260825, tmp_path)
    on, on_name = runner.build_command("yolo", "on", 20260825, tmp_path)
    assert off_name == "off-s20260825"
    assert on_name == "on-s20260825"
    assert off[2] == on[2]
    assert off[4:] == on[4:]
    assert off[1].endswith("d2_off.yaml")
    assert on[1].endswith("d2_on.yaml")


def test_p1_three_seed_decision_is_pre_registered_and_recomputable():
    """The formal no-go must follow from committed paired rows and the frozen rule."""
    evidence = json.loads((ROOT / "results" / "d2_p1_three_seed_results.json").read_text(encoding="utf-8"))
    deltas = [pair["on_best_map50_95"] - pair["off_best_map50_95"] for pair in evidence["pairs"]]
    stats = evidence["statistics"]
    assert evidence["status"] == "completed_three_seed_same_budget_pair"
    assert len(deltas) == stats["n"] == 3
    assert all(
        math.isclose(actual, recorded, abs_tol=1e-12) for actual, recorded in zip(deltas, stats["paired_deltas"])
    )
    assert math.isclose(sum(deltas) / len(deltas), stats["mean_delta"], abs_tol=1e-12)
    low, high = stats["confidence_interval_95"]
    assert low <= 0 <= high
    assert abs(stats["mean_delta"]) < evidence["decision"]["pre_registered_threshold"] == 0.003
    assert evidence["decision"]["status"] == "no_go"
    assert evidence["decision"]["rule_unchanged_after_observing_results"] is True
    assert all(
        arm["best_checkpoint"]["load_verified"] and arm["healthy_checkpoint"]["load_verified"]
        for paired_arms in evidence["arms"].values()
        for arm in paired_arms.values()
    )
    for paired_artifacts in evidence["artifacts"]["per_run_csv"].values():
        for artifact in paired_artifacts.values():
            path = ROOT.parents[1] / artifact["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    for key in ("aggregate_csv", "curve"):
        artifact = evidence["artifacts"][key]
        path = ROOT.parents[1] / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    runtime_records = evidence["artifacts"]["runtime_records"]
    assert len(runtime_records) == 4
    assert all(record["returncode"] == 0 and record["log_hash_verified"] for record in runtime_records)
    assert evidence["artifacts"]["runtime_record_coverage"]["complete_console_tee"] == [20260825, 20260826]


def test_p1_runner_records_commit_and_dirty_state():
    """Future manifests must bind results to both a commit and working-tree state."""
    runner = _load_script("run_p1.py")
    state = runner.git_state()
    assert len(state["commit"]) == 40
    assert isinstance(state["dirty"], bool)
    assert len(state["porcelain_sha256"]) == 64
    assert isinstance(state["experiment_inputs_dirty"], bool)
    assert len(state["experiment_inputs_porcelain_sha256"]) == 64


def test_p1_runner_replaces_unrepresentable_console_characters():
    """A narrow Windows console must not interrupt the complete UTF-8 log tee."""
    runner = _load_script("run_p1.py")

    class AsciiStream:
        encoding = "ascii"

        def __init__(self):
            self.output = ""
            self.flushed = False

        def write(self, value):
            self.output += value

        def flush(self):
            self.flushed = True

    stream = AsciiStream()
    runner.console_write("epoch 1 ╸\n", stream=stream)
    assert stream.output == "epoch 1 ?\n"
    assert stream.flushed is True


def test_post_run_args_audits_pass_for_original_and_corrected_pairs():
    """Pre-run config equality must be followed by actual resolved-args equality."""
    for name in ("d2_p1_resolved_args_audit.json", "d2_p1_corrected_resolved_args_audit.json"):
        evidence = json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        assert evidence["status"] == "passed"
        assert len(evidence["records"]) == 3
        for record in evidence["records"]:
            assert not record["unexpected_differences"]
            assert all(record["required_equalities"].values())
            for file in record["files"].values():
                path = ROOT.parents[1] / file["path"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == file["sha256"]


def test_weight_calibration_is_ap_blind_and_mechanically_identified():
    """The corrected weight must be chosen from training signal, not the validation metric."""
    evidence = json.loads((ROOT / "results" / "d2_p1_weight_calibration.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "selected"
    assert evidence["selection_contract"]["uses_validation_metric"] is False
    assert evidence["selection_contract"]["target_foundation_task_ratio"] == [0.03, 0.06]
    assert evidence["selected_weight"] == 0.1
    assert all(record["mechanism_identity"]["passed"] for record in evidence["records"])
    selected = next(record for record in evidence["records"] if record["weight"] == evidence["selected_weight"])
    assert 0.03 <= selected["foundation_task_ratio"] <= 0.06


def test_corrected_three_seed_result_recomputes_to_no_go():
    """A visible KD signal must still satisfy the frozen paired decision rule."""
    evidence = json.loads((ROOT / "results" / "d2_p1_corrected_results.json").read_text(encoding="utf-8"))
    deltas = [pair["on_best_map50_95"] - pair["off_best_map50_95"] for pair in evidence["pairs"]]
    assert evidence["correction"]["selection_used_validation_ap"] is False
    assert evidence["correction"]["selected_weight"] == 0.1
    assert all(0.03 <= pair["on_mean_foundation_task_ratio"] <= 0.06 for pair in evidence["pairs"])
    assert math.isclose(sum(deltas) / len(deltas), evidence["statistics"]["mean_delta"], abs_tol=1e-12)
    low, high = evidence["statistics"]["confidence_interval_95"]
    assert low <= 0 <= high
    assert abs(evidence["statistics"]["mean_delta"]) < evidence["decision"]["pre_registered_threshold"]
    assert evidence["decision"]["status"] == "no_go"
    assert all(
        record["returncode"] == 0 and record["log_hash_verified"] for record in evidence["artifacts"]["runtime_records"]
    )
