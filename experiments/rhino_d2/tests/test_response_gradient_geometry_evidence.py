"""Offline integrity checks for P2-05, consuming no validation or formal metrics."""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results/p2_response_gradient_geometry"
PREFIX = "response_gradient_geometry"


def read_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_geometry_artifacts_match_execution_manifest():
    manifest = read_json(f"{PREFIX}_manifest.json")
    for name, expected in manifest["artifacts"].items():
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == expected
    assert manifest["protocol_commit"] == "aebf564db30d6329484f7e8fe5d66d82d5647746"
    assert manifest["source_state"]["commit"] == "8a990995add82afcef6d2d8b865b997eeb1a473b"


def test_all_observations_satisfy_independently_recomputed_geometry_identity():
    with (RESULTS / f"{PREFIX}_raw.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 384
    assert Counter(r["seed"] for r in rows) == {"20260824": 128, "20260825": 128, "20260826": 128}
    assert len({(r["seed"], r["batch_index"], r["condition"]) for r in rows}) == 384
    assert len({r["condition"] for r in rows}) == 8
    for r in rows:
        for key, value in r.items():
            if key not in ("condition", "family"):
                assert math.isfinite(float(value))
        az = float(r["response_grad_Z"]) / float(r["static_grad_Z"])
        af = float(r["response_grad_P4"]) / float(r["static_grad_P4"])
        gn, ga, j = (float(r[key]) for key in ("G_norm", "G_angle", "projector_transfer_ratio"))
        assert af == pytest.approx(az * j, rel=2e-4)
        assert az == pytest.approx(math.sqrt(2) * gn * ga, rel=2e-4)
        assert float(r["dominance_margin"]) == pytest.approx(
            math.log(gn) - max(math.log(ga), 0) - math.log(math.sqrt(2)) - max(math.log(j), 0)
        )
        assert float(r["max_derivative_relative_error"]) <= 2e-4


def test_checkpoint_and_access_audit_has_explicit_booleans():
    manifest = read_json(f"{PREFIX}_manifest.json")
    assert manifest["optimizer_steps"] == 0
    assert manifest["model_ema_updates"] == 0
    assert manifest["formal_training_started"] is False
    assert manifest["no_validation_access"] is True
    assert manifest["no_response128_access"] is True
    assert manifest["file_access_audit"]["denied"] == []
    assert len(manifest["data_inventory"]) == 128
    assert len(manifest["input_records"]) == 1536
    for entry in manifest["data_inventory"]:
        assert entry["relative_path"].startswith(("images/train2017/", "labels/train2017/"))
    for runtime in manifest["checkpoint_runtime"].values():
        for name in (
            "state_unchanged",
            "file_unchanged",
            "parameter_grads_none",
            "bn_post_clean_restored_all_observations",
            "teacher_projection_frozen",
            "teacher_aligned_features_detached",
        ):
            assert runtime[name] is True
        assert all(flag is True for flag in runtime["bn_training_flags"].values())
        assert runtime["state_before"] == runtime["state_after"]
        assert runtime["checkpoint_hash_before"] == runtime["checkpoint_hash_after"]
        assert runtime["optimizer_steps"] == 0


def test_same_input_tensors_as_frozen_p2_04_calibration():
    """Use only the old input manifest hash; do not read its metric values."""
    manifest = read_json(f"{PREFIX}_manifest.json")
    old_path = EXP / "results/p2_response_field/calibration/calibration_manifest.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        json.dumps(manifest["input_records"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == old["execution"]["rolling_manifest_sha256"]
    assert manifest["frozen_inputs"]["train_subset"]["sha256"] == (
        "1ad936698234fd07651993dbdefe7a98ffaf74861432a103b6e397bb45b9b676"
    )


def test_overlapping_signals_match_old_calibration_without_alpha_selection():
    with (RESULTS / f"{PREFIX}_raw.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (EXP / "results/p2_response_field/calibration/calibration_raw.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        previous = list(csv.DictReader(stream))
    assert len(previous) == len(rows)
    for r, old in zip(rows, previous):
        assert (r["seed"], r["batch_index"], r["condition"]) == (old["seed"], old["batch_index"], old["condition_id"])
        for key in ("static_clean_loss", "static_perturbed_loss", "response_loss", "static_response_clean_dot"):
            assert float(r[key]) == pytest.approx(float(old[key]), rel=2e-5, abs=2e-7)
        assert float(r["A_P4"]) == pytest.approx(
            float(old["second_term_grad_ratio_response_over_static_perturbed"]), rel=2e-5
        )


def test_pooled_bootstrap_and_frozen_decision_reproduce_from_raw():
    with (RESULTS / f"{PREFIX}_raw.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = read_json(f"{PREFIX}_result.json")
    rng = np.random.default_rng(20260908)
    batch_draws = rng.integers(0, 16, size=(10000, 16))
    for metric in ("dominance_margin", "A_Z_single"):
        blocks = np.array([[float(r[metric]) for r in rows if int(r["batch_index"]) == b] for b in range(16)])
        medians = np.median(blocks[batch_draws].reshape(10000, -1), axis=1)
        expected = np.quantile(medians, [0.025, 0.975])
        summary = next(r for r in result["pooled"] if r["metric"] == metric)
        assert [summary["ci_low"], summary["ci_high"]] == pytest.approx(expected)
    with (RESULTS / f"{PREFIX}_summary.csv").open(encoding="utf-8", newline="") as stream:
        summaries = list(csv.DictReader(stream))
    seeds = [r for r in summaries if r["scope"] == "seed" and r["metric"] == "dominance_margin"]
    families = [r for r in summaries if r["scope"] == "family" and r["metric"] == "dominance_margin"]
    leave = [r for r in summaries if r["scope"] == "leave_family_out" and r["metric"] == "A_Z_single"]
    assert len(seeds) == 3 and len(families) == 4 and len(leave) == 4
    assert all(float(r["ci_low"]) > 0 for r in seeds + families)
    assert all(float(r["ci_low"]) > 1 for r in leave)
    assert result["decision"]["status"] == "H1_support"
    assert result["decision"]["alpha_selection_authorized"] is False
    assert result["decision"]["formal_training_authorized"] is False
