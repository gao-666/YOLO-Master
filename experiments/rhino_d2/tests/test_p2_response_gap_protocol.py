"""Guard the frozen P2-03 diagnostic-only response-gap protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = EXPERIMENT_ROOT / "configs/d2_v3_p2_response_gap.yaml"
PROTOCOL = EXPERIMENT_ROOT / "DINOV3_P2_RESPONSE_GAP_PROTOCOL.md"
SUBSET = EXPERIMENT_ROOT / "results/p2_response_gap/diagnostic_response128.txt"
OLD_SUBSET = EXPERIMENT_ROOT / "results/p2_gradient_conflict/diagnostic_train64.txt"
EXPECTED_SUBSET_SHA256 = "c0e00ba5e15de01d21afb35ae50937681186905d717498d4549565a3613dd582"


def sha256(path: Path) -> str:
    """Return one file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_is_diagnostic_only_and_uses_frozen_p1_late_checkpoints() -> None:
    """P2-03 must not silently become a new training experiment."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["mode"] == "diagnostic_only_no_training"
    assert config["arms"] == ["off", "on64"]
    assert config["seeds"] == [20260824, 20260825, 20260826]
    assert config["checkpoint_epoch"] == 49
    assert config["checkpoint_weights"] == "ema"
    assert all("epoch49.pt" in pattern for pattern in config["checkpoint_patterns"].values())
    assert config["decision"]["inconclusive_does_not_authorize_training"] is True


def test_new_subset_is_frozen_unique_and_disjoint() -> None:
    """Lock the new 128-image panel before any response result exists."""
    entries = SUBSET.read_text(encoding="utf-8").splitlines()
    old_entries = set(OLD_SUBSET.read_text(encoding="utf-8").splitlines())
    assert len(entries) == len(set(entries)) == 128
    assert not old_entries.intersection(entries)
    assert sha256(SUBSET) == EXPECTED_SUBSET_SHA256


def test_perturbations_and_hypotheses_are_frozen() -> None:
    """Guard the eight conditions and the three-stage falsification tree."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    conditions = {key: config["perturbations"][key] for key in (
        "brightness",
        "contrast",
        "gaussian_blur_sigma",
        "gaussian_noise_std",
    )}
    assert conditions == {
        "brightness": [0.8, 0.6],
        "contrast": [0.75, 0.5],
        "gaussian_blur_sigma": [1.0, 2.0],
        "gaussian_noise_std": [0.03, 0.06],
    }
    text = PROTOCOL.read_text(encoding="utf-8")
    assert all(name in text for name in ("H1", "H2", "H3"))
    assert "本协议本身不授权训练" in text
    assert "Response-Field Distillation" in text


def test_numeric_and_smoke_boundaries_are_explicit() -> None:
    """Lock deterministic noise reuse, numeric dtypes, and no-peeking smoke tests."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    embedding = config["relation_embedding"]
    assert embedding["feature_compute_dtype"] == "float32"
    assert embedding["statistical_accumulator_dtype"] == "float64"
    assert config["perturbations"]["gaussian_noise_generator"].endswith("reused_across_all_models")
    assert config["smoke_test"] == {
        "inputs": "synthetic_only",
        "diagnostic_images_forbidden": True,
        "formal_metrics_forbidden": True,
    }
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "int.from_bytes" in text
    assert "三个 seed 和 Teacher 必须复用逐位完全相同" in text


def test_image_level_detection_loss_cannot_be_faked_from_a_batch_total() -> None:
    """Freeze the one-image raw-head loss path and state-integrity audit."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["feature_batch_size"] == 4
    assert config["detection_loss_batch_size"] == 1
    assert config["detection_loss_mode"] == "raw_head_student_train_all_batchnorm_eval_no_grad"
    assert config["state_integrity"] == "state_dict_sha256_must_match_before_after"
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "不得把 batch aggregate loss 拆分、广播或伪装成 image-level loss" in text
