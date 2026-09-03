"""Unit contracts for the train64-only P2-04 alpha calibration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from ultralytics.nn.foundation import RESPONSE_FIELD_CONDITIONS

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = EXPERIMENT_ROOT / "scripts/calibrate_response_field_alpha.py"
CONFIG = EXPERIMENT_ROOT / "configs/d2_v3_p2_response_field_calibration.yaml"
SPEC = importlib.util.spec_from_file_location("calibrate_response_field_alpha", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def frozen_config() -> dict:
    """Return the selection-only fields needed by synthetic rule tests."""
    return {
        "checkpoints": {20260824: {}, 20260825: {}, 20260826: {}},
        "alpha_candidates": [0.25, 0.5, 1.0, 2.0, 4.0],
        "lambda": 0.15,
        "acceptance_band": [0.5, 2.0],
        "minimum_passing_seeds": 2,
    }


def raw_row(seed: int, *, response_scale: float = 1.0) -> dict:
    """Build one finite cached observation for deterministic rule tests."""
    return {
        "seed": seed,
        "task_loss": 1.0,
        "static_clean_loss": 1.0,
        "static_perturbed_loss": 1.0,
        "response_loss": response_scale,
        "task_grad_norm": 1.0,
        "static_clean_grad_sq": 1.0,
        "static_perturbed_grad_sq": 1.0,
        "response_clean_grad_sq": response_scale,
        "response_perturbed_grad_sq": response_scale,
        "static_response_clean_dot": 0.0,
    }


def test_calibration_config_freezes_authorized_candidates_conditions_and_stopline():
    """The executable config exactly matches the authorized calibration envelope."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["alpha_candidates"] == [0.25, 0.5, 1.0, 2.0, 4.0]
    assert [tuple(value) for value in config["conditions"]] == list(RESPONSE_FIELD_CONDITIONS)
    assert config["train_subset_sha256"] == "1ad936698234fd07651993dbdefe7a98ffaf74861432a103b6e397bb45b9b676"
    assert config["formal_training_authorized"] is False


def test_analytic_gradient_scaling_matches_explicit_concatenated_gradient():
    """Five alpha candidates can be evaluated without rerunning forward or autograd."""
    static_clean = torch.tensor([1.2, -0.4, 0.7])
    response_clean = torch.tensor([-0.2, 0.5, 1.1])
    response_perturbed = torch.tensor([0.3, -0.9])
    for alpha in (0.25, 0.5, 1.0, 2.0, 4.0):
        explicit = torch.linalg.vector_norm(
            torch.cat((static_clean + alpha * response_clean, alpha * response_perturbed))
        )
        analytic = MODULE.analytic_pair_gradient_norm(
            float(static_clean.square().sum()),
            float(response_clean.square().sum()),
            float(response_perturbed.square().sum()),
            float((static_clean * response_clean).sum()),
            alpha,
        )
        assert analytic == pytest.approx(float(explicit), rel=1e-6)


def test_selection_rule_chooses_signal_matched_alpha_one_with_frozen_ties():
    """The exact score and deterministic tie policy select from accepted candidates only."""
    rows = [raw_row(seed) for seed in frozen_config()["checkpoints"]]
    _, result = MODULE.summarize_calibration(rows, frozen_config())
    assert result["status"] == "passed"
    assert result["selected_alpha"] == 1.0
    assert result["selected_alpha_label"] == "signal-matched alpha"
    assert result["formal_training_authorized"] is False


def test_zero_accepted_candidates_fails_and_stops_formal_training():
    """The candidate grid and acceptance band cannot expand after calibration failure."""
    rows = [raw_row(seed, response_scale=100.0) for seed in frozen_config()["checkpoints"]]
    _, result = MODULE.summarize_calibration(rows, frozen_config())
    assert result["status"] == "failed"
    assert result["selected_alpha"] is None
    assert result["failure_action"] == "stop_p2_04_no_formal_training"
    assert result["formal_training_authorized"] is False
