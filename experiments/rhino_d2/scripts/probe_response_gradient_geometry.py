"""P2-05: locate response-gradient amplification without updating a model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import calibrate_response_field_alpha as base
import numpy as np
import torch

from ultralytics.nn.foundation import (
    BatchNormBufferSnapshot,
    StudentFeatureTap,
    apply_response_field_condition_batch,
    preserve_batchnorm_buffers,
    response_field_kd_loss,
    strict_cosine_kd_loss,
)

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/rhino_d2"
PROTOCOL = EXP / "DINOV3_P2_RESPONSE_GRADIENT_GEOMETRY_PROTOCOL.md"
PROTOCOL_COMMIT = "aebf564db30d6329484f7e8fe5d66d82d5647746"
CONFIG = EXP / "configs/d2_v3_p2_response_gradient_geometry.yaml"
OUTPUT = EXP / "results/p2_response_gradient_geometry"
PREFIX = "response_gradient_geometry"


def norm(tensor):
    """Accumulate diagnostic norms in float64, after FP32 autograd."""
    return float(torch.linalg.vector_norm(tensor.detach().double()))


def pair_norm(a, b):
    """Measure a concatenated gradient without summing different feature coordinates."""
    return math.hypot(norm(a), norm(b))


def cosine_geometry(student, teacher):
    """Derive the exact mean-token cosine gradient, independently of backward."""
    s, t = student.detach().double(), teacher.detach().double()
    ns, nt = s.norm(dim=1), t.norm(dim=1)
    if not torch.isfinite(s).all() or not torch.isfinite(t).all() or min(float(ns.min()), float(nt.min())) < 1e-6:
        raise ValueError("non-finite or zero-norm cosine geometry")
    u, v = s / ns[:, None], t / nt[:, None]
    cosine = (u * v).sum(dim=1)
    tangent = cosine[:, None] * u - v
    sine = tangent.norm(dim=1)
    n = ns.numel()
    return {"norm": ns, "teacher_norm": nt, "cos": cosine, "sin": sine, "gradient": tangent / (n * ns[:, None])}


def verify_vector(observed, predicted, settings):
    """Fail on an incorrect derivative, including its sign and token mean."""
    obs, pred = observed.detach().double(), predicted.detach().double()
    error = norm(obs - pred) / max(norm(pred), 1e-30)
    if error > settings["relative_l2_tolerance"] or not torch.allclose(
        obs, pred, rtol=settings["vector_rtol"], atol=settings["vector_atol"]
    ):
        raise RuntimeError(f"cosine derivative invariant failed: relative L2={error}")
    return error


def tensor_summary(prefix, tensor):
    """Keep token distributions rather than an ambiguous norm of a mean vector."""
    x = tensor.detach().double().cpu().numpy().reshape(-1)
    return {
        f"{prefix}_{key}": float(value)
        for key, value in {
            "mean": x.mean(),
            "std": x.std(ddof=1),
            "min": x.min(),
            "q25": np.quantile(x, 0.25),
            "median": np.median(x),
            "q75": np.quantile(x, 0.75),
            "max": x.max(),
        }.items()
    }


def measure_geometry(zc, zp, tc, tp, fc, fp, settings):
    """Separate angle, inverse norm, branch counting, and projector transfer."""
    sc = strict_cosine_kd_loss(zc, tc)
    sp = strict_cosine_kd_loss(zp, tp)
    response = response_field_kd_loss(zc, zp, tc, tp)
    gsc_z, gsc_f = torch.autograd.grad(sc, (zc, fc), retain_graph=True)
    gsp_z, gsp_f = torch.autograd.grad(sp, (zp, fp), retain_graph=True)
    grc_z, grp_z, grc_f, grp_f = torch.autograd.grad(response, (zc, zp, fc, fp))
    static = cosine_geometry(zp, tp)
    clean = cosine_geometry(zc, tc)
    geom = cosine_geometry(zp.float() - zc.float(), tp.detach().float() - tc.detach().float())
    errors = [
        verify_vector(gsp_z, static["gradient"], settings),
        verify_vector(grp_z, geom["gradient"], settings),
        verify_vector(grc_z, -geom["gradient"], settings),
    ]
    r, s = norm(geom["gradient"]), norm(static["gradient"])
    u = norm(geom["sin"] / static["norm"]) / static["norm"].numel()
    if min(r, s, u, norm(gsp_f)) <= 1e-12:
        raise RuntimeError("undefined geometry ratio; observation cannot be skipped")
    az = pair_norm(grc_z, grp_z) / norm(gsp_z)
    af = pair_norm(grc_f, grp_f) / norm(gsp_f)
    gn, ga, j = r / u, u / s, af / az
    branch = pair_norm(grc_z, grp_z) / norm(grp_z)
    if not math.isclose(branch, math.sqrt(2), rel_tol=2e-4):
        raise RuntimeError("paired-branch sqrt(2) invariant failed")
    factor_error = abs(math.sqrt(2) * gn * ga * j / af - 1)
    if factor_error > settings["relative_l2_tolerance"]:
        raise RuntimeError("multiplicative decomposition failed")
    gradients = (gsc_z, gsc_f, gsp_z, gsp_f, grc_z, grp_z, grc_f, grp_f)
    if not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("non-finite observed gradient")
    return {
        "token_count": static["norm"].numel(),
        "static_clean_loss": float(sc.detach()),
        "static_perturbed_loss": float(sp.detach()),
        "response_loss": float(response.detach()),
        "predicted_response_grad_Z_single": r,
        "observed_response_grad_Z_single": norm(grp_z),
        "static_grad_Z": norm(gsp_z),
        "response_grad_Z": pair_norm(grc_z, grp_z),
        "static_grad_P4": norm(gsp_f),
        "response_grad_P4": pair_norm(grc_f, grp_f),
        "static_clean_grad_P4": norm(gsc_f),
        "static_2v_grad_P4": pair_norm(gsc_f, gsp_f),
        "static_response_clean_dot": float((gsc_f.detach().double() * grc_f.detach().double()).sum()),
        "A_Z": az,
        "A_Z_single": az / math.sqrt(2),
        "A_P4": af,
        "projector_transfer_ratio": j,
        "G_norm": gn,
        "G_angle": ga,
        "branch_factor": branch,
        "dominance_margin": math.log(gn) - max(math.log(ga), 0) - math.log(math.sqrt(2)) - max(math.log(j), 0),
        "max_derivative_relative_error": max(errors),
        "factor_relative_error": factor_error,
        **tensor_summary("student_static_clean_norm", clean["norm"]),
        **tensor_summary("student_static_norm", static["norm"]),
        **tensor_summary("student_response_norm", geom["norm"]),
        **tensor_summary("teacher_response_norm", geom["teacher_norm"]),
        **tensor_summary("cos_static", static["cos"]),
        **tensor_summary("cos_response", geom["cos"]),
        **tensor_summary("sin_static", static["sin"]),
        **tensor_summary("sin_response", geom["sin"]),
        **tensor_summary("predicted_token_grad", geom["gradient"].norm(dim=1)),
        **tensor_summary("observed_token_grad", grp_z.detach().norm(dim=1)),
    }


def synthetic_invariants(settings):
    """Check scaling, token mean, two branches, and detach before GPU use."""
    rows = []
    for scale in (1.0, 0.5, 0.25, 0.125):
        s = torch.zeros(2, 2, 2, 2)
        s[:, 0] = scale
        s.requires_grad_(True)
        t = torch.ones_like(s, requires_grad=True)
        loss = strict_cosine_kd_loss(s, t)
        g = torch.autograd.grad(loss, s)[0]
        error = verify_vector(g, cosine_geometry(s, t)["gradient"], settings)
        assert t.grad is None
        expected = math.sqrt(0.5) / (scale * math.sqrt(8))
        assert math.isclose(norm(g), expected, rel_tol=2e-5)
        rows.append({"scale": scale, "loss": float(loss.detach()), "gradient_norm": norm(g), "relative_error": error})
    assert max(x["loss"] for x in rows) - min(x["loss"] for x in rows) < 1e-6
    return {"status": "passed", "device": "cpu", "observations": rows}


class ReadGuard:
    """Audit Python file opens; explicitly not an OS-level native I/O tracer."""

    def __init__(self):
        self.active = True
        self.reads = set()
        self.denied = []

    def __call__(self, event, args):
        if not self.active or event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.fsdecode(args[0]).replace("\\", "/").lower()
        prohibited = (
            "/val2017",
            "diagnostic_response128",
            "/p2_response_gap/",
            "/p2_response_field/formal/",
            "/experiments/study/",
        )
        if any(value in path for value in prohibited) or ("/v3_p1/" in path and path.endswith((".csv", ".json"))):
            self.denied.append(path)
            raise PermissionError(f"P2-05 forbidden input: {path}")
        mode = args[1]
        if (mode is None or "r" in str(mode)) and "2026yolo" in path:
            self.reads.add(path)


def build_train64(config, settings, output):
    """Reuse YOLODataset on isolated copies of only the authorized 64 image/label pairs."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset

    data = base.load_yaml(ROOT / config["data"])
    root = Path(settings["dataset_root"])
    entries = [
        line.strip().removeprefix("./")
        for line in (ROOT / config["train_subset"]).read_text().splitlines()
        if line.strip()
    ]
    if len(set(entries)) != 64:
        raise RuntimeError("train64 membership invalid")
    inventory, local_paths = [], []
    for relative in entries:
        label = relative.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
        for entry in (relative, label):
            source = (root / entry).resolve()
            if root.resolve() not in source.parents:
                raise RuntimeError("dataset path escape")
            target = output / "runtime/data" / entry
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = base.sha256(source)
            if base.sha256(target) != digest:
                raise RuntimeError("train64 staging digest mismatch")
            inventory.append({"relative_path": entry, "sha256": digest})
        local_paths.append((output / "runtime/data" / relative).resolve())
    listing = output / "runtime/train64.txt"
    listing.write_text("".join(str(p) + "\n" for p in local_paths), encoding="utf-8")
    args = get_cfg(overrides={"task": "detect", "imgsz": 256, "batch": 4, "workers": 0, "rect": False, "cache": False})
    dataset = build_yolo_dataset(
        args, str(listing), 4, {"names": data["names"], "nc": len(data["names"])}, mode="val", rect=False, stride=32
    )
    by_path = {Path(path).resolve(): index for index, path in enumerate(dataset.im_files)}
    batches = [dataset.collate_fn([dataset[by_path[p]] for p in local_paths[i : i + 4]]) for i in range(0, 64, 4)]
    return batches, [entries[i : i + 4] for i in range(0, 64, 4)], inventory


METRICS = (
    "A_Z",
    "A_Z_single",
    "A_P4",
    "projector_transfer_ratio",
    "G_norm",
    "G_angle",
    "dominance_margin",
    "student_static_norm_median",
    "student_response_norm_median",
    "teacher_response_norm_median",
    "cos_static_mean",
    "cos_response_mean",
    "response_grad_Z",
    "response_grad_P4",
)


def summarize(rows, settings):
    """Bootstrap whole fixed batch blocks with seeds and conditions retained together."""
    groups = [("pooled", "all", rows)]
    for field in ("seed", "family", "condition"):
        for value in sorted({r[field] for r in rows}):
            groups.append((field, str(value), [r for r in rows if r[field] == value]))
    for family in sorted({r["family"] for r in rows}):
        groups.append(("leave_family_out", family, [r for r in rows if r["family"] != family]))
    result = []
    for group_index, (scope, label, subset) in enumerate(groups):
        blocks = [[r for r in subset if r["batch_index"] == b] for b in range(16)]
        if len({len(block) for block in blocks}) != 1 or not blocks[0]:
            raise RuntimeError("unbalanced bootstrap blocks")
        array = np.asarray([[[r[m] for m in METRICS] for r in block] for block in blocks])
        rng = np.random.default_rng(settings["bootstrap_seed"] + group_index)
        draws = rng.integers(0, 16, (settings["bootstrap_replicates"], 16))
        for mi, metric in enumerate(METRICS):
            raw = array[:, :, mi].reshape(-1)
            bootstrap = np.median(array[:, :, mi][draws].reshape(len(draws), -1), axis=1)
            lo, hi = np.quantile(bootstrap, [0.025, 0.975])
            result.append(
                {
                    "scope": scope,
                    "group": label,
                    "metric": metric,
                    "n": len(raw),
                    "blocks": 16,
                    "mean": float(raw.mean()),
                    "std": float(raw.std(ddof=1)),
                    "median": float(np.median(raw)),
                    "q25": float(np.quantile(raw, 0.25)),
                    "q75": float(np.quantile(raw, 0.75)),
                    "positive_fraction": float(np.mean(raw > 0)),
                    "negative_fraction": float(np.mean(raw < 0)),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                }
            )
    return result


def decide(summary):
    """Apply the pre-data operational definition of widespread norm dominance."""

    def select(scope, metric):
        return [r for r in summary if r["scope"] == scope and r["metric"] == metric]

    d = select("pooled", "dominance_margin")[0]
    a = select("pooled", "A_Z_single")[0]
    seeds = sum(r["ci_low"] > 0 for r in select("seed", "dominance_margin"))
    families = sum(r["ci_low"] > 0 for r in select("family", "dominance_margin"))
    leave = all(r["ci_low"] > 1 for r in select("leave_family_out", "A_Z_single"))
    if d["ci_low"] > 0 and seeds >= 2 and families >= 3 and leave:
        status = "H1_support"
    elif d["ci_high"] < 0 or a["ci_high"] <= 1:
        status = "H1_no_support"
    else:
        status = "ambiguous"
    return {
        "status": status,
        "passing_seeds": seeds,
        "passing_families": families,
        "leave_family_out_amplification": leave,
        "pooled_dominance_ci": [d["ci_low"], d["ci_high"]],
        "formal_training_authorized": False,
        "alpha_selection_authorized": False,
    }


def plot_result(rows, path):
    """Draw the observed derivative identity and separate geometry from projector gain."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted({r["family"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    colors = ["#2369a4", "#c86b27", "#36936d", "#965fa3"]
    for family, color in zip(families, colors):
        group = [r for r in rows if r["family"] == family]
        axes[0].scatter(
            [r["predicted_response_grad_Z_single"] for r in group],
            [r["observed_response_grad_Z_single"] for r in group],
            s=11,
            alpha=0.55,
            label=family,
            color=color,
        )
        axes[1].scatter([r["A_Z"] for r in group], [r["A_P4"] for r in group], s=11, alpha=0.55, color=color)
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        low = min(ax.get_xlim()[0], ax.get_ylim()[0])
        high = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([low, high], [low, high], "k--", alpha=0.5)
        ax.grid(alpha=0.2)
    axes[0].set(
        xlabel="Predicted single-branch gradient norm",
        ylabel="Autograd norm",
        title="Derivative identity (sanity check)",
    )
    axes[1].set(
        xlabel="A_Z (paired response / perturbed static)", ylabel="A_P4", title="Projector transfer: diagonal means J=1"
    )
    keys = ["G_norm", "G_angle", "projector_transfer_ratio"]
    for i, (key, label) in enumerate(zip(keys, ["Inverse norm", "Angle", "Projector J"])):
        values = [np.median([math.log(r[key]) for r in rows if r["family"] == f]) for f in families]
        axes[2].bar(np.arange(4) + (i - 1) * 0.25, values, width=0.25, label=label)
    axes[2].set_xticks(np.arange(4), [f.replace("gaussian_", "") for f in families], rotation=20)
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[2].set(ylabel="Median log factor", title="Descriptive decomposition by family")
    axes[2].legend(fontsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle("P2-05 | frozen train64 geometry probe | no training / no efficacy evaluation")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(settings, config, output, log, start):
    """Execute exactly three frozen checkpoint probes and archive model-state checks."""
    random.seed(20260903)
    np.random.seed(20260903)
    torch.manual_seed(20260903)
    torch.cuda.manual_seed_all(20260903)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    frozen = base.validate_frozen_inputs(ROOT, ROOT / settings["base_config"], config)
    base.validate_p1_config(config, base.load_yaml(ROOT / config["p1_config"]))
    batches, paths, inventory = build_train64(config, settings, output)
    device = torch.device(config["device"])
    rows, runtime, records = [], {}, []
    for seed, spec in frozen["checkpoints"].items():
        log.write("teacher_cache_start", seed=seed)
        tc_cache, tp_cache, teacher_metadata = base.cache_teacher_features(config, batches, paths, seed, device)
        checkpoint = torch.load(spec["path"], weights_only=False, map_location="cpu")
        base.validate_checkpoint(checkpoint, config, seed)
        wrapper = checkpoint["ema"].float().to(device)
        student, projector = wrapper.student_model, wrapper.projector_for("p4")
        student.train()
        projector.train()
        before = base.state_digest(wrapper)
        tap = StudentFeatureTap(student, "p4")
        if tap.source_index != settings["expected_tap_index"] or not projector.teacher_projection_frozen:
            raise RuntimeError("tap/projector contract changed")
        roots = {"student": student, "projector": projector}
        bn_flags = {
            f"{root_name}.{name}": module.training
            for root_name, root in roots.items()
            for name, module in root.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        }
        if not bn_flags or any(value is not True for value in bn_flags.values()):
            raise RuntimeError("Student/projector BN must retain train semantics")
        try:
            for bi, cpu_batch in enumerate(batches):
                for ci, (family, value, condition) in enumerate(config["conditions"]):
                    if time.perf_counter() - start > settings["maximum_wall_seconds"]:
                        raise RuntimeError("frozen wall-clock budget exceeded")
                    snapshot = base.buffer_snapshot(wrapper)
                    try:
                        batch = base.batch_to_device(cpu_batch, device)
                        image = batch["img"].detach().requires_grad_(True)
                        tap.clear()
                        predictions = student(image)
                        fc = tap.feature
                        task, _ = student.loss(batch, predictions)
                        if not torch.isfinite(task).all():
                            raise RuntimeError("clean task loss is not finite")
                        zc, tc = projector(fc, tc_cache[bi].to(device))
                        post_clean = BatchNormBufferSnapshot(roots)
                        tp_feature, manifest = tp_cache[(bi, ci)]
                        perturbed, repeated = apply_response_field_condition_batch(
                            cpu_batch["img"].float() / 255,
                            paths[bi],
                            family=family,
                            value=value,
                            condition_id=condition,
                            seed=seed,
                            epoch_index=49,
                            batch_index_within_epoch=bi,
                            num_batches_per_epoch=16,
                        )
                        if manifest != repeated:
                            raise RuntimeError("teacher/student input tensor digest mismatch")
                        with preserve_batchnorm_buffers(roots):
                            tap.clear()
                            student(perturbed.to(device).requires_grad_(True))
                            fp = tap.feature
                            zp, tp = projector(fp, tp_feature.to(device))
                        if not post_clean.matches():
                            raise RuntimeError("BN post-clean state mismatch")
                        measured = measure_geometry(zc, zp, tc, tp, fc, fp, settings)
                        shapes = {"aligned_Z": list(zc.shape), "student_P4": list(fc.shape)}
                        if tc.requires_grad or tp.requires_grad:
                            raise RuntimeError("projected Teacher features must be detached")
                        row = {"seed": seed, "batch_index": bi, "condition": condition, "family": family, **measured}
                        rows.append(row)
                        records.extend({"seed": seed, **record} for record in manifest)
                        base.write_csv(output / f"{PREFIX}_raw.csv", rows)
                        del image, predictions, task, fc, fp, zc, zp, tc, tp
                    finally:
                        base.restore_buffer_snapshot(wrapper, snapshot)
                log.write(
                    "batch_complete",
                    seed=seed,
                    batch_index=bi,
                    observations=len(rows),
                    elapsed=round(time.perf_counter() - start, 2),
                )
        finally:
            tap.close()
        after = base.state_digest(wrapper)
        file_after = base.sha256(spec["path"])
        checks = {
            "state_unchanged": before == after,
            "file_unchanged": spec["sha256"] == file_after,
            "parameter_grads_none": all(p.grad is None for p in wrapper.parameters()),
        }
        if any(v is not True for v in checks.values()):
            raise RuntimeError(f"checkpoint audit failed: {checks}")
        runtime[str(seed)] = {
            **checks,
            "state_before": before,
            "state_after": after,
            "checkpoint_hash_before": spec["sha256"],
            "checkpoint_hash_after": file_after,
            "tap_index": tap.source_index,
            "dtype": str(next(student.parameters()).dtype),
            "device": str(next(student.parameters()).device),
            "bn_training_flags": bn_flags,
            "bn_post_clean_restored_all_observations": True,
            "teacher_projection_frozen": projector.teacher_projection_frozen,
            "teacher_aligned_features_detached": True,
            "feature_shapes": shapes,
            "optimizer_steps": 0,
            "student_forward_batches": 256,
            "teacher_forward_batches": 144,
            "autograd_grad_calls": 384,
        }
        log.write("seed_complete", seed=seed, **checks)
        del wrapper, student, projector, checkpoint, tc_cache, tp_cache
        torch.cuda.empty_cache()
    if len(rows) != 384 or len(records) != 1536:
        raise RuntimeError("incomplete observations")
    return rows, runtime, records, inventory, frozen, teacher_metadata


def main():
    """Expose only CPU sanity and the fixed real probe; no hyperparameter overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    settings = base.load_yaml(CONFIG)
    sanity = synthetic_invariants(settings)
    if args.synthetic_only:
        print(json.dumps(sanity, indent=2))
        return
    source = base.git_state(ROOT)
    for path in (PROTOCOL, CONFIG):
        expected = subprocess.check_output(
            ["git", "show", f"{PROTOCOL_COMMIT}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT
        )
        if path.read_bytes().replace(b"\r\n", b"\n") != expected.replace(b"\r\n", b"\n"):
            raise RuntimeError("preregistered protocol/config changed")
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite an existing P2-05 evidence directory")
    config_path = ROOT / settings["base_config"]
    if (
        base.sha256(config_path) != settings["base_config_sha256"]
        or base.sha256(Path(base.__file__)) != settings["base_script_sha256"]
    ):
        raise RuntimeError("frozen reused implementation/config changed")
    config = base.load_yaml(config_path)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "runtime").mkdir()
    (OUTPUT / "runtime/.gitignore").write_text("*\n", encoding="utf-8")
    base.configure_runtime(OUTPUT)
    # Import local plotting support before installing the experiment data-read guard.
    import matplotlib.pyplot  # noqa: F401

    guard = ReadGuard()
    sys.addaudithook(guard)
    log = base.AuditLog(OUTPUT / "probe.log")
    start = time.perf_counter()
    try:
        log.write("start", source_commit=source["commit"], protocol_commit=PROTOCOL_COMMIT, training_runs=0)
        rows, runtime, records, inventory, frozen, teacher_metadata = run(settings, config, OUTPUT, log, start)
        summary = summarize(rows, settings)
        decision = decide(summary)
        base.write_csv(OUTPUT / f"{PREFIX}_summary.csv", summary)
        result = {
            "protocol_id": settings["protocol_id"],
            "decision": decision,
            "source_commit": source["commit"],
            "protocol_commit": PROTOCOL_COMMIT,
            "observations": len(rows),
            "max_derivative_relative_error": max(r["max_derivative_relative_error"] for r in rows),
            "pooled": [r for r in summary if r["scope"] == "pooled"],
            "claim_boundary": "conditional train64 batch-level gradient decomposition; no detection efficacy or causal attribution",
        }
        (OUTPUT / f"{PREFIX}_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (OUTPUT / "synthetic_invariants.json").write_text(json.dumps(sanity, indent=2) + "\n", encoding="utf-8")
        plot_result(rows, OUTPUT / f"{PREFIX}.png")
        log.write("complete", **decision, elapsed_seconds=round(time.perf_counter() - start, 3))
        if guard.denied:
            raise RuntimeError("forbidden data access attempted")
        guard.active = False
        input_paths = [
            PROTOCOL,
            CONFIG,
            Path(__file__),
            Path(base.__file__),
            ROOT / "ultralytics/nn/foundation/response.py",
            ROOT / "ultralytics/nn/foundation/projectors.py",
            ROOT / "ultralytics/nn/foundation/taps.py",
            ROOT / "ultralytics/nn/foundation/teachers/dinov3.py",
            ROOT / "ultralytics/nn/foundation/preprocessing.py",
            ROOT / "ultralytics/cfg/models/26/yolo26-master-n.yaml",
        ]
        artifacts = {p.name: base.sha256(p) for p in OUTPUT.iterdir() if p.is_file()}
        manifest = {
            "protocol_commit": PROTOCOL_COMMIT,
            "source_state": source,
            "resolved_config": settings,
            "reused_config": config,
            "frozen_inputs": frozen["inputs"],
            "teacher_assets": frozen["teacher_assets"],
            "teacher_metadata": teacher_metadata,
            "checkpoint_runtime": runtime,
            "input_records": records,
            "data_inventory": inventory,
            "input_sha256": {p.relative_to(ROOT).as_posix(): base.sha256(p) for p in input_paths},
            "file_access_audit": {
                "python_open_reads": sorted(guard.reads),
                "denied": guard.denied,
                "coverage": "Python open events; native image reads bound by explicit staging inventory and tensor digests",
            },
            "optimizer_steps": 0,
            "model_ema_updates": 0,
            "task_ema_transient_writes_rolled_back": True,
            "formal_training_started": False,
            "no_validation_access": True,
            "no_response128_access": True,
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "wall_seconds": time.perf_counter() - start,
            },
            "artifacts": artifacts,
        }
        (OUTPUT / f"{PREFIX}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    except Exception:
        guard.active = False
        log.write("technical_invalid", traceback=traceback.format_exc(), alpha_selection=False, formal_training=False)
        raise


if __name__ == "__main__":
    main()
