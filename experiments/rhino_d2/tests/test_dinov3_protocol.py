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
