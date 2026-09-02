"""Guard the frozen DINOv3-S P2-02 single-variable protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
P1_CONFIG = EXPERIMENT_ROOT / "configs/d2_v3_p1_on.yaml"
P2_CONFIG = EXPERIMENT_ROOT / "configs/d2_v3_p2_align128.yaml"


def load_module(name: str, path: Path):
    """Load one experiment script without modifying sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_alignment_dimension_and_output_identity_change() -> None:
    """Reject every accidental treatment change."""
    p1 = yaml.safe_load(P1_CONFIG.read_text(encoding="utf-8"))
    p2 = yaml.safe_load(P2_CONFIG.read_text(encoding="utf-8"))
    differences = {key for key in set(p1) | set(p2) if p1.get(key) != p2.get(key)}
    assert differences == {"foundation_align_dim", "name", "project"}
    assert p1["foundation_align_dim"] == 64
    assert p2["foundation_align_dim"] == 128
    assert p1["foundation_loss_weight"] == p2["foundation_loss_weight"] == 0.15


def test_runner_freezes_three_seeds_and_on128_identity() -> None:
    """The runner must add only the registered ON128 runs."""
    runner = load_module("run_p2_align_dim", EXPERIMENT_ROOT / "scripts/run_p2_align_dim.py")
    assert runner.SEEDS == (20260824, 20260825, 20260826)
    command, name = runner.build_command("yolo", 20260824, Path("runs/test"))
    assert name == "v3-p2-align128-s20260824"
    assert "seed=20260824" in command
    assert "exist_ok=False" in command


def test_decision_rule_is_exactly_pre_registered() -> None:
    """Lock the support, no-support, and inconclusive regions."""
    summary = load_module("summarize_p2_align_dim", EXPERIMENT_ROOT / "scripts/summarize_p2_align_dim.py")
    assert summary.decision(0.003, 0.0001, 0.006)[0] == "support"
    assert summary.decision(0.0029, -0.001, 0.006)[0] == "no_support"
    assert summary.decision(0.0031, -0.001, 0.007)[0] == "inconclusive"
    assert summary.decision(-0.004, -0.008, -0.001)[0] == "inconclusive"


def test_protocol_freezes_references_and_gradient_subset() -> None:
    """Keep P1/P2-01 conclusions and the P2-01 diagnostic sample immutable."""
    text = (EXPERIMENT_ROOT / "DINOV3_P2_ALIGN_DIM_PROTOCOL.md").read_text(encoding="utf-8")
    assert "P1" in text and "No-Go" in text
    assert "P2-01" in text
    assert "diagnostic_train64.txt" in text
    assert "1ad936698234fd07651993dbdefe7a98ffaf74861432a103b6e397bb45b9b676" in text
    assert "256/512" in text
