"""Contracts for the isolated DINOv3 retest stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]


def sha256(path: Path) -> str:
    """Return one file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text_lf(path: Path) -> str:
    """Hash UTF-8 text after cross-platform LF normalization."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode()).hexdigest()


def load_yaml(name: str) -> dict:
    """Load one D2 config."""
    value = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def load_result(name: str) -> dict:
    """Load one committed DINOv3 gate result."""
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def test_historical_dinov2_configs_keep_their_original_semantics():
    """The new protocol must never rewrite historical evidence inputs."""
    assert load_yaml("d2_off.yaml")["foundation_teacher"] == "none"
    assert load_yaml("d2_on.yaml")["foundation_teacher"] == "dinov2"
    assert load_yaml("d2_on_calibrated.yaml")["foundation_teacher"] == "dinov2"


def test_v3_manifest_matches_gate_evidence_without_claiming_apache_license():
    """Teacher identity and license must agree across the manifest and P0 evidence."""
    manifest = json.loads((ROOT / "env" / "dinov3_teacher_manifest.json").read_text(encoding="utf-8"))
    assert manifest["license"]["name"] == "DINOv3 License"
    assert manifest["license"]["model_card_license_id"] == "dinov3-license"
    assert manifest["license"]["model_card_license_type"] == "other"
    for key, result_name in (("vits16", "d2_v3_p0_vits16.json"), ("vitl16", "d2_v3_p0_vitl16.json")):
        result = load_result(result_name)
        assert result["teacher"]["license"] == manifest["license"]["name"]
        result_assets = {asset["name"]: asset for asset in result["teacher"]["assets"]}
        assert result_assets["model.safetensors"]["sha256"] == manifest[key]["files"]["model.safetensors"]["sha256"]
        assert result_assets["LICENSE.md"]["sha256"] == manifest[key]["files"]["LICENSE.md"]["sha256"]


def test_v3_teacher_dtype_gates_use_common_bf16_and_finite_features():
    """S/L capacity comparisons may proceed only with one common stable dtype."""
    for name, channels in (("d2_v3_teacher_vits16_bf16.json", 384), ("d2_v3_teacher_vitl16_bf16.json", 1024)):
        result = load_result(name)
        assert result["status"] == "passed"
        assert result["teacher"]["actual_dtype"] == "torch.bfloat16"
        assert result["teacher"]["feature_shape"] == [2, channels, 14, 14]
        assert all(result["checks"].values())


def test_v3_p0_gates_recompute_and_bind_current_script_and_configs():
    """Both Teachers must pass every frozen P0 redline without spatial resizing."""
    script = ROOT / "scripts" / "d2_v3_p0_train_smoke.py"
    for config_name, result_name, channels in (
        ("d2_v3_p0_vits16.yaml", "d2_v3_p0_vits16.json", 384),
        ("d2_v3_p0_vitl16.yaml", "d2_v3_p0_vitl16.json", 1024),
    ):
        config = ROOT / "configs" / config_name
        result = load_result(result_name)
        assert result["status"] == "passed"
        assert result["script_sha256"] == sha256(script)
        assert result["config"]["sha256"] == sha256(config)
        assert result["teacher"]["feature_shape"] == [2, channels, 14, 14]
        assert result["alignment"]["metadata"]["teacher_resized"] is False
        assert all(result["checks"].values())
        assert result["steps"][-1]["foundation_loss"] < result["steps"][0]["foundation_loss"]


def test_sanity_config_changes_only_budget_and_output_identity_from_historical_off():
    """The OFF-only sanity pilot must not smuggle in a second training change."""
    historical = load_yaml("d2_off.yaml")
    sanity = load_yaml("d2_v3_off_sanity.yaml")
    differences = {key for key in set(historical) | set(sanity) if historical.get(key) != sanity.get(key)}
    assert differences == {"epochs", "name", "project"}
    assert historical["epochs"] == 10 and sanity["epochs"] == 50


def test_new_scripts_never_mask_nonfinite_teacher_features():
    """The DINOv3 gate must reject non-finite features instead of rewriting them."""
    for name in ("d2_v3_teacher_smoke.py", "d2_v3_p0_train_smoke.py"):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "nan_to_num" not in source


def test_failed_v3_baseline_gate_blocks_formal_p1_and_archives_evidence():
    """A weak OFF baseline must fail closed before any DINOv3 efficacy comparison."""
    result = load_result("d2_v3_baseline_sanity.json")
    assert result["status"] == "failed"
    assert result["claim"] == "engineering_pipeline_sanity_gate_not_foundation_efficacy"
    assert result["checks"]["completed_50_epochs"] is True
    assert result["checks"]["late_median_map_at_least_0p01"] is False
    assert result["checks"]["final_detection_loss_down_at_least_10pct"] is False
    assert result["runtime"]["experiment_inputs_dirty"] is False
    assert result["runtime"]["returncode"] == 0
    assert result["decision"]["formal_p1_unlocked"] is False
    for artifact in result["artifacts"].values():
        path = REPO_ROOT / artifact["path"]
        assert path.is_file()
        assert sha256(path) == artifact["sha256"]


def test_formal_v3_p1_configs_do_not_exist_before_baseline_gate_passes():
    """The failed sanity gate must leave formal DINOv3 ON/OFF inputs uncreated."""
    assert not (ROOT / "configs" / "d2_v3_off.yaml").exists()
    assert not (ROOT / "configs" / "d2_v3_vits16_on.yaml").exists()


def test_recovery_a_changes_only_data_and_output_identity():
    """Candidate A must isolate dataset scale from every training hyperparameter."""
    sanity = load_yaml("d2_v3_off_sanity.yaml")
    recovery = load_yaml("d2_v3_baseline_recovery_a.yaml")
    differences = {key for key in set(sanity) | set(recovery) if sanity.get(key) != recovery.get(key)}
    assert differences == {"data", "name", "project"}
    assert recovery["pretrained"] is False
    assert recovery["foundation_enabled"] is False


def test_recovery_a_dataset_selection_and_payload_are_frozen():
    """The committed lists and downloaded payload must match the frozen manifest."""
    manifest_path = ROOT / "datasets/d2_coco_mini_2048_seed20260901/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["protocol_role"] == "baseline_recovery_candidate_a_data_only"
    generator = REPO_ROOT / manifest["generator"]["path"]
    assert sha256_text_lf(generator) == manifest["generator"]["sha256_lf_canonical"]
    assert manifest["selection"]["seed"] == 20260901
    assert manifest["payload"]["missing_images"] == 0
    assert manifest["payload"]["images"]["count"] == 2560
    for split, expected_size in (("train2017", 2048), ("val2017", 512)):
        record = manifest["selection"]["splits"][split]
        path = REPO_ROOT / record["list_path"]
        assert record["size"] == expected_size
        assert len(path.read_text(encoding="utf-8").splitlines()) == expected_size
        assert sha256_text_lf(path) == record["list_sha256_lf_canonical"]


def test_failed_recovery_a_gate_is_bound_to_complete_evidence_and_blocks_p1():
    """Candidate A must fail closed on mAP while preserving every archived artifact."""
    result = load_result("d2_v3_baseline_recovery_a.json")
    assert result["status"] == "failed"
    assert result["claim"] == "engineering_pipeline_sanity_gate_not_foundation_efficacy"
    assert result["checks"] == {
        "completed_50_epochs": True,
        "late_median_map_at_least_0p01": False,
        "late_median_precision_nonzero": True,
        "late_median_recall_nonzero": True,
        "final_detection_loss_down_at_least_10pct": True,
    }
    assert result["observations"]["late_median_map50_95"] == 0.00546
    assert result["runtime"]["source_commit"] == "3d40f3b3f775f1985d9ff58e3de643a9a14eb6d8"
    assert result["runtime"]["experiment_inputs_dirty"] is False
    assert result["runtime"]["returncode"] == 0
    assert result["decision"]["formal_p1_unlocked"] is False
    assert result["decision"]["next_action"] == "run_pretrained_candidate_b_on_same_dataset"
    for artifact in result["artifacts"].values():
        path = REPO_ROOT / artifact["path"]
        assert path.is_file()
        assert sha256(path) == artifact["sha256"]


def test_recovery_b_changes_only_student_initialization_and_output_identity():
    """Candidate B must retain Candidate A data, budget, model, and gate."""
    recovery_a = load_yaml("d2_v3_baseline_recovery_a.yaml")
    recovery_b = load_yaml("d2_v3_baseline_recovery_b.yaml")
    differences = {key for key in set(recovery_a) | set(recovery_b) if recovery_a.get(key) != recovery_b.get(key)}
    assert differences == {"pretrained", "name", "project"}
    assert recovery_b["model"] == "ultralytics/cfg/models/26/yolo26-master-n.yaml"
    assert recovery_b["pretrained"] == "experiments/rhino_d2/cache/yolo26n.pt"
    assert recovery_b["foundation_enabled"] is False


def test_candidate_b_initialization_audit_passes_without_replacing_student():
    """Partial transfer must cover all shared families before Candidate B training."""
    result = load_result("d2_v3_student_init_audit.json")
    assert result["status"] == "passed"
    assert result["source"]["sha256"] == "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
    assert result["target"]["architecture_replaced"] is False
    assert result["observations"]["target_parameter_coverage"] >= 0.40
    assert result["observations"]["source_parameter_coverage"] >= 0.80
    assert result["observations"]["groups"]["stem"]["coverage"] == 1.0
    assert result["observations"]["groups"]["shared_deep_backbone"]["coverage"] == 1.0
    assert result["observations"]["groups"]["head"]["coverage"] == 1.0
    assert all(result["checks"].values())
    assert result["decision"]["candidate_b_training_allowed"] is True
    assert result["decision"]["formal_p1_unlocked"] is False


def test_recovery_b_passes_the_frozen_gate_and_archives_complete_evidence():
    """A qualified pretrained baseline may unlock protocol freeze, not claim KD efficacy."""
    result = load_result("d2_v3_baseline_recovery_b.json")
    assert result["status"] == "passed"
    assert result["claim"] == "engineering_pipeline_sanity_gate_not_foundation_efficacy"
    assert all(result["checks"].values())
    assert result["observations"]["late_median_map50_95"] == 0.048125
    assert result["runtime"]["source_commit"] == "a967a830330169fbe63ec868527e31fe9888b2e6"
    assert result["runtime"]["experiment_inputs_dirty"] is False
    assert result["runtime"]["returncode"] == 0
    assert result["decision"]["formal_p1_unlocked"] is True
    assert result["decision"]["next_action"] == "freeze_formal_p1_protocol"
    for artifact in result["artifacts"].values():
        path = REPO_ROOT / artifact["path"]
        assert path.is_file()
        assert sha256(path) == artifact["sha256"]


def test_v3_p1_calibration_changes_only_treatment_budget_and_output_from_off():
    """Calibration must inherit the qualified baseline and disable validation."""
    off = load_yaml("d2_v3_p1_off.yaml")
    calibration = load_yaml("d2_v3_p1_calibration.yaml")
    differences = {key for key in set(off) | set(calibration) if off.get(key) != calibration.get(key)}
    assert differences == {
        "epochs",
        "foundation_enabled",
        "foundation_loss_weight",
        "name",
        "project",
        "save",
        "val",
    }
    assert off["pretrained"] == calibration["pretrained"] == "experiments/rhino_d2/cache/yolo26n.pt"
    assert off["data"] == calibration["data"]
    assert calibration["foundation_teacher"] == "dinov3"
    assert calibration["foundation_loss"] == "cosine"
    assert calibration["foundation_teacher_dtype"] == "bf16"
    assert calibration["val"] is False
