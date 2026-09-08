"""Tests that distinguish derivative correctness, branch accounting, and state isolation."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("p2_geometry", SCRIPTS / "probe_response_gradient_geometry.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SETTINGS = MODULE.base.load_yaml(MODULE.CONFIG)


def test_scale_invariance_and_inverse_gradient_with_token_mean():
    result = MODULE.synthetic_invariants(SETTINGS)
    assert result["status"] == "passed"
    norms = [r["gradient_norm"] for r in result["observations"]]
    assert norms == pytest.approx([norms[0] * k for k in (1, 2, 4, 8)])


def test_vector_check_rejects_missing_mean_factor_and_sign_error():
    x = torch.randn(2, 4, 3, 3, requires_grad=True)
    t = torch.randn_like(x)
    g = torch.autograd.grad(MODULE.strict_cosine_kd_loss(x, t), x)[0]
    pred = MODULE.cosine_geometry(x, t)["gradient"]
    MODULE.verify_vector(g, pred, SETTINGS)
    for bad in (-pred, pred * 18):
        with pytest.raises(RuntimeError, match="invariant"):
            MODULE.verify_vector(g, bad, SETTINGS)


def test_two_branch_and_projector_decomposition_with_known_linear_map():
    generator = torch.Generator().manual_seed(14)
    fc = torch.randn(2, 4, 3, 3, generator=generator, requires_grad=True)
    fp = torch.randn(2, 4, 3, 3, generator=generator, requires_grad=True)
    # A common isotropic projector scales all feature gradients equally: J must be one.
    zc, zp = 3 * fc, 3 * fp
    tc = torch.randn(2, 4, 3, 3, generator=generator, requires_grad=True)
    tp = torch.randn(2, 4, 3, 3, generator=generator, requires_grad=True)
    row = MODULE.measure_geometry(zc, zp, tc, tp, fc, fp, SETTINGS)
    assert row["branch_factor"] == pytest.approx(math.sqrt(2))
    assert row["projector_transfer_ratio"] == pytest.approx(1, rel=1e-5)
    assert row["A_P4"] == pytest.approx(math.sqrt(2) * row["G_norm"] * row["G_angle"], rel=1e-5)
    assert all(x.grad is None for x in (fc, fp, tc, tp))


def test_zero_response_fails_closed():
    s = torch.ones(2, 3, 2, 2)
    with pytest.raises(ValueError, match="zero-norm"):
        MODULE.cosine_geometry(s - s, s)


def test_real_projector_train_bn_allows_geometry_and_exact_buffer_rollback():
    from ultralytics.nn.foundation import P4AlignmentProjector, preserve_batchnorm_buffers

    torch.manual_seed(24)
    projector = P4AlignmentProjector(4, 6, 3).train()
    before = MODULE.base.state_digest(projector)
    snapshot = MODULE.base.buffer_snapshot(projector)
    fc = torch.randn(2, 4, 3, 3, requires_grad=True)
    fp = torch.randn_like(fc, requires_grad=True)
    zc, tc = projector(fc, torch.randn(2, 6, 3, 3))
    post_clean = MODULE.BatchNormBufferSnapshot({"projector": projector})
    with preserve_batchnorm_buffers({"projector": projector}):
        zp, tp = projector(fp, torch.randn(2, 6, 3, 3))
    assert post_clean.matches()
    row = MODULE.measure_geometry(zc, zp, tc, tp, fc, fp, SETTINGS)
    assert row["max_derivative_relative_error"] < SETTINGS["relative_l2_tolerance"]
    MODULE.base.restore_buffer_snapshot(projector, snapshot)
    assert MODULE.base.state_digest(projector) == before
    assert all(p.grad is None for p in projector.parameters())


def test_bootstrap_preserves_batch_blocks_and_is_reproducible():
    settings = {**SETTINGS, "bootstrap_replicates": 100}
    rows = [
        {
            "seed": seed,
            "family": family,
            "condition": family,
            "batch_index": batch,
            **{metric: float(batch + 1) for metric in MODULE.METRICS},
        }
        for batch in range(16)
        for seed in (24, 25, 26)
        for family in ("brightness", "contrast", "blur", "noise")
    ]
    summary = MODULE.summarize(rows, settings)
    assert summary == MODULE.summarize(rows, settings)
    assert all(row["blocks"] == 16 and row["median"] == 8.5 for row in summary)
    incomplete = [row for row in rows if row["batch_index"] != 15]
    with pytest.raises(RuntimeError, match="unbalanced"):
        MODULE.summarize(incomplete, settings)


def test_read_guard_rejects_prohibited_data_and_not_training_source():
    guard = MODULE.ReadGuard()
    guard("open", ("E:/2026YOLO/datasets/mini/images/train2017/a.jpg", "r", 0))
    for path in ("E:/datasets/val2017/a.jpg", "diagnostic_response128.txt", "E:/p2_response_gap/formal/result.json"):
        with pytest.raises(PermissionError):
            guard("open", (path, "r", 0))


def test_h1_decision_can_reject_or_remain_ambiguous():
    def row(scope, metric, low, high):
        return {"scope": scope, "metric": metric, "ci_low": low, "ci_high": high}

    data = [row("pooled", "dominance_margin", -2, -1), row("pooled", "A_Z_single", 2, 3)]
    data += [row("seed", "dominance_margin", -2, 1)] * 3
    data += [row("family", "dominance_margin", -2, 1)] * 4
    data += [row("leave_family_out", "A_Z_single", 2, 3)] * 4
    assert MODULE.decide(data)["status"] == "H1_no_support"
    data[0]["ci_high"] = 1
    assert MODULE.decide(data)["status"] == "ambiguous"
