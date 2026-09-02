"""Tests for the ON128-minus-ON64 gradient follow-up."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_p2_align_gradient.py"
SPEC = importlib.util.spec_from_file_location("compare_p2_align_gradient", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_direction_label_requires_interval_separation() -> None:
    """A shift is resolved only when its full interval excludes zero."""
    assert MODULE.direction_label({"ci_low": 0.1, "ci_high": 0.2}) == "increase"
    assert MODULE.direction_label({"ci_low": -0.2, "ci_high": -0.1}) == "decrease"
    assert MODULE.direction_label({"ci_low": -0.1, "ci_high": 0.2}) == "no_detectable_shift"


def test_pairing_is_exact_and_signed_on128_minus_on64() -> None:
    """The comparison must retain image identity and the registered sign."""
    rows64 = []
    rows128 = []
    for seed in MODULE.SEEDS:
        for epoch in MODULE.EPOCHS:
            for index in range(64):
                base = {
                    "seed": str(seed),
                    "epoch": str(epoch),
                    "image": f"images/train2017/{index:012d}.jpg",
                    "valid": "True",
                }
                rows64.append({**base, "cosine": "0.1", "norm_ratio": "0.01"})
                rows128.append({**base, "cosine": "0.2", "norm_ratio": "0.03"})
    paired = MODULE.pair_records(rows64, rows128)
    assert len(paired) == 576
    assert all(abs(row["delta_cosine"] - 0.1) < 1e-12 for row in paired)
    assert all(abs(row["delta_norm_ratio"] - 0.02) < 1e-12 for row in paired)
