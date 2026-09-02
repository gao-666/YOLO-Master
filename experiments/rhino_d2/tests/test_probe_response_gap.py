"""Synthetic-only checks for the P2-03 response-gap probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_response_gap.py"
SPEC = importlib.util.spec_from_file_location("probe_response_gap", SCRIPT)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_gaussian_noise_is_bitwise_reproducible_and_path_bound() -> None:
    """The same condition is reused exactly while a different image path changes noise."""
    images = torch.full((2, 3, 16, 16), 0.5, dtype=torch.float32)
    paths = ["images/train2017/a.jpg", "images/train2017/b.jpg"]
    first = PROBE.perturb_images(images, paths, "gaussian_noise", 0.03, "gaussian_noise:0.03")
    second = PROBE.perturb_images(images, paths, "gaussian_noise", 0.03, "gaussian_noise:0.03")
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])


@pytest.mark.parametrize(
    ("family", "value", "condition"),
    [
        ("brightness", 0.8, "brightness:0.8"),
        ("contrast", 0.75, "contrast:0.75"),
        ("gaussian_blur", 1.0, "gaussian_blur:1.0"),
        ("gaussian_noise", 0.03, "gaussian_noise:0.03"),
    ],
)
def test_all_perturbation_families_are_finite_and_bounded(family: str, value: float, condition: str) -> None:
    """Synthetic images exercise every frozen perturbation implementation."""
    images = torch.rand(2, 3, 32, 32)
    output = PROBE.perturb_images(images, ["a", "b"], family, value, condition)
    assert output.shape == images.shape
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


def test_projection_free_embedding_contract_and_zero_norm_guard() -> None:
    """The exact 16x16 relation vector is FP32 and degenerate responses fail closed."""
    result = PROBE.spatial_relation_embedding(torch.randn(3, 11, 7, 9, dtype=torch.float16))
    assert result.shape == (3, 32640)
    assert result.dtype == torch.float32
    with pytest.raises(ValueError, match="fail-closed"):
        PROBE.cosine_gap(torch.zeros(3, 8), torch.ones(3, 8))


def test_spearman_ties_and_image_cluster_bootstrap() -> None:
    """Tie-aware ranks and the seed-by-image cluster shape stay deterministic."""
    rho = PROBE.spearman(np.array([1, 2, 2, 4]), np.array([4, 2, 2, 1]))
    assert rho == pytest.approx(-1.0)
    values = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    first = PROBE.bootstrap_mean_by_image(values, 100, 7, 0.95)
    second = PROBE.bootstrap_mean_by_image(values, 100, 7, 0.95)
    assert first == second
    assert first["estimate"] == pytest.approx(3.5)
