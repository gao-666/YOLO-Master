from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_gradient_conflict.py"
SPEC = importlib.util.spec_from_file_location("probe_gradient_conflict", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_subset_selection_is_deterministic_and_unique():
    entries = [f"./images/train2017/{index:012d}.jpg" for index in range(100)]
    first = MODULE.select_subset(entries, 16, "fixed")
    second = MODULE.select_subset(list(reversed(entries)), 16, "fixed")
    assert first == second
    assert len(first) == len(set(first)) == 16


def test_gradient_metrics_distinguish_alignment_conflict_and_strength():
    task = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    kd = torch.tensor([[2.0, 0.0], [-0.5, 0.0]])
    rows = MODULE.gradient_metrics(task, kd, 1e-12)
    assert rows[0]["cosine"] == 1.0
    assert rows[0]["norm_ratio"] == 2.0
    assert rows[1]["cosine"] == -1.0
    assert rows[1]["norm_ratio"] == 0.5


def test_decision_rule_is_fail_closed():
    summaries = [
        {"epoch": 49, "cosine_ci_high": -0.1},
        {"epoch": 49, "cosine_ci_high": -0.2},
        {"epoch": 49, "cosine_ci_high": 0.1},
    ]
    assert (
        MODULE.decision_from_summaries(summaries, {"cosine_ci_low": -0.3, "cosine_ci_high": -0.05}, 49)["status"]
        == "supported"
    )
    assert (
        MODULE.decision_from_summaries(summaries, {"cosine_ci_low": 0.01, "cosine_ci_high": 0.2}, 49)["status"]
        == "not_supported"
    )
    assert (
        MODULE.decision_from_summaries(summaries, {"cosine_ci_low": -0.1, "cosine_ci_high": 0.1}, 49)["status"]
        == "inconclusive"
    )
