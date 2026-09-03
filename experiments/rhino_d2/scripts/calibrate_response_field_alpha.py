#!/usr/bin/env python3
"""Run the authorized train64-only P2-04 signal calibration without model updates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs/d2_v3_p2_response_field_calibration.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/p2_response_field/calibration"


def sha256(path: Path) -> str:
    """Return one file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash a finite tensor without dtype conversions."""
    if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
        raise ValueError("tensor digest requires one finite tensor")
    value = tensor.detach().contiguous().cpu()
    header = f"{tuple(value.shape)}|{value.dtype}|".encode()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + raw).hexdigest()


def state_digest(module: nn.Module) -> str:
    """Hash a module state_dict with stable names, metadata, and raw tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value = tensor.detach().contiguous().cpu()
        digest.update(f"{name}\0{tuple(value.shape)}\0{value.dtype}\0".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_tensor_digests(module: nn.Module) -> dict[str, str]:
    """Return per-key state hashes for fail-closed mutation diagnostics."""
    return {name: tensor_digest(tensor) for name, tensor in module.state_dict().items()}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a YAML mapping: {path}")
    return value


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    """Resolve a repository-relative path without permitting escape."""
    path = (repo_root / value).resolve()
    if path != repo_root and repo_root not in path.parents:
        raise ValueError(f"path escapes repository: {value}")
    return path


class AuditLog:
    """Mirror timestamped progress to stdout and the complete calibration log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def write(self, event: str, **values: Any) -> None:
        record = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **values}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)


def configure_runtime(output_dir: Path) -> None:
    """Keep mutable library state inside the calibration evidence directory."""
    runtime = output_dir / "runtime"
    for name in ("yolo_config", "matplotlib", "huggingface"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime / "yolo_config")
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    os.environ["HF_HOME"] = str(runtime / "huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def validate_frozen_inputs(repo_root: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on every frozen calibration input before importing model code."""
    from ultralytics.nn.foundation import RESPONSE_FIELD_CONDITIONS

    expected_conditions = [tuple(item) for item in config["conditions"]]
    if expected_conditions != list(RESPONSE_FIELD_CONDITIONS):
        raise RuntimeError("calibration conditions differ from the eight frozen response-field conditions")
    if [float(value) for value in config["alpha_candidates"]] != [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise RuntimeError("alpha candidate set is not frozen")
    fixed = {
        "protocol": str(config["protocol_sha256"]),
        "amendment": str(config["amendment_sha256"]),
        "p1_config": str(config["p1_config_sha256"]),
        "data": str(config["data_sha256"]),
        "train_subset": str(config["train_subset_sha256"]),
    }
    inputs = {"config": {"path": str(config_path), "sha256": sha256(config_path)}}
    for name, expected in fixed.items():
        path = resolve_repo_path(repo_root, str(config[name]))
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name} SHA-256 mismatch: {actual} != {expected}")
        inputs[name] = {"path": str(path), "sha256": actual}

    teacher_root = Path(str(config["teacher_weights"])).resolve()
    teacher_assets = []
    for name, expected in config["teacher_asset_sha256"].items():
        path = teacher_root / name
        actual = sha256(path)
        if actual != str(expected):
            raise RuntimeError(f"Teacher asset SHA-256 mismatch for {name}: {actual} != {expected}")
        teacher_assets.append({"name": name, "path": str(path), "bytes": path.stat().st_size, "sha256": actual})

    checkpoints = {}
    for raw_seed, spec in config["checkpoints"].items():
        seed = int(raw_seed)
        path = resolve_repo_path(repo_root, str(spec["path"]))
        actual = sha256(path)
        if actual != str(spec["sha256"]):
            raise RuntimeError(f"checkpoint SHA-256 mismatch for seed {seed}: {actual} != {spec['sha256']}")
        checkpoints[seed] = {"path": path, "sha256": actual, "bytes": path.stat().st_size}
    if sorted(checkpoints) != [20260824, 20260825, 20260826]:
        raise RuntimeError("checkpoint seed set is not frozen")
    return {"inputs": inputs, "teacher_assets": teacher_assets, "checkpoints": checkpoints}


def git_state(repo_root: Path) -> dict[str, Any]:
    """Record the clean implementation commit and scoped working-tree state."""
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=repo_root, text=True).strip()
    branch = subprocess.check_output(("git", "branch", "--show-current"), cwd=repo_root, text=True).strip()
    status = subprocess.check_output(("git", "status", "--porcelain"), cwd=repo_root, text=True).splitlines()
    relevant = [line for line in status if "experiments/study/" not in line.replace("\\", "/")]
    if relevant:
        raise RuntimeError(f"calibration implementation/evidence scope is dirty: {relevant}")
    return {
        "commit": head,
        "branch": branch,
        "dirty": bool(status),
        "ignored_user_scope": [line for line in status if line not in relevant],
        "calibration_scope_clean": True,
    }


def validate_p1_config(config: dict[str, Any], p1: dict[str, Any]) -> None:
    """Verify the P1 Student/Teacher/calibration locks."""
    expected = {
        "foundation_enabled": True,
        "foundation_teacher": "dinov3",
        "foundation_model": config["teacher_model"],
        "foundation_weights": config["teacher_weights"],
        "foundation_target_levels": [config["student_level"]],
        "foundation_align_dim": int(config["align_dim"]),
        "foundation_loss": config["foundation_loss"],
        "foundation_loss_weight": float(config["lambda"]),
        "foundation_teacher_dtype": config["teacher_dtype"],
        "imgsz": int(config["imgsz"]),
        "batch": int(config["batch_size"]),
        "epochs": 50,
        "optimizer": "SGD",
    }
    mismatch = {key: [p1.get(key), value] for key, value in expected.items() if p1.get(key) != value}
    if mismatch:
        raise RuntimeError(f"P1 config lock mismatch: {json.dumps(mismatch)}")


def resolve_selected_paths(repo_root: Path, config: dict[str, Any]) -> tuple[list[str], list[Path], Path]:
    """Resolve exactly the authorized train64 list under the checked training root."""
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(resolve_repo_path(repo_root, str(config["data"]))))
    dataset_root = Path(data["path"]).resolve()
    subset = resolve_repo_path(repo_root, str(config["train_subset"]))
    entries = [
        line.strip().replace("\\", "/") for line in subset.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(entries) != int(config["subset_size"]) or len(entries) != len(set(entries)):
        raise RuntimeError("train64 subset count or uniqueness check failed")
    paths = []
    normalized = []
    for entry in entries:
        relative = entry.removeprefix("./")
        path = (dataset_root / relative).resolve()
        if dataset_root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"train64 image is missing or outside the dataset root: {path}")
        paths.append(path)
        normalized.append(relative)
    return normalized, paths, dataset_root


def build_batches(
    repo_root: Path, config: dict[str, Any], normalized: list[str], selected_paths: list[Path]
) -> tuple[list[dict[str, Any]], list[list[str]], dict[str, Any]]:
    """Build deterministic augmentation-free batches from the authorized training images."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(resolve_repo_path(repo_root, str(config["data"]))))
    args = get_cfg(
        overrides={
            "task": "detect",
            "imgsz": int(config["imgsz"]),
            "batch": int(config["batch_size"]),
            "workers": 0,
            "rect": False,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "fraction": 1.0,
        }
    )
    dataset = build_yolo_dataset(
        args, data["train"], int(config["batch_size"]), data, mode="val", rect=False, stride=32
    )
    by_path = {Path(path).resolve(): index for index, path in enumerate(dataset.im_files)}
    if any(path not in by_path for path in selected_paths):
        raise RuntimeError("one or more train64 images are absent from the frozen training split")
    indices = [by_path[path] for path in selected_paths]
    batches, path_batches = [], []
    batch_size = int(config["batch_size"])
    for start in range(0, len(indices), batch_size):
        stop = start + batch_size
        items = [dataset[index] for index in indices[start:stop]]
        batches.append(dataset.collate_fn(items))
        path_batches.append(normalized[start:stop])
    if len(batches) != int(config["num_batches_per_epoch"]):
        raise RuntimeError("calibration batch count differs from the frozen logical epoch size")
    return (
        batches,
        path_batches,
        {
            "split": "train",
            "augmentation": False,
            "image_count": len(indices),
            "batch_count": len(batches),
            "batch_size": batch_size,
            "dataset_root": str(Path(data["path"]).resolve()),
            "ordered_images_sha256": hashlib.sha256("".join(f"{x}\n" for x in normalized).encode()).hexdigest(),
        },
    )


def batch_to_device(cpu_batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move a collated detection batch and normalize images to [0, 1]."""
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in cpu_batch.items()}
    batch["img"] = batch["img"].float().div_(255.0)
    return batch


def validate_checkpoint(checkpoint: dict[str, Any], config: dict[str, Any], seed: int) -> None:
    """Fail closed on EMA checkpoint identity and Foundation locks."""
    epoch = int(config["checkpoint_epoch"])
    if int(checkpoint.get("epoch", -1)) != epoch or checkpoint.get("ema") is None:
        raise RuntimeError(f"seed {seed} is not the frozen epoch{epoch} EMA checkpoint")
    args = checkpoint.get("train_args") or {}
    foundation = checkpoint.get("foundation") or {}
    locks = {
        "seed": (int(args.get("seed", -1)), seed),
        "imgsz": (int(args.get("imgsz", -1)), int(config["imgsz"])),
        "batch": (int(args.get("batch", -1)), int(config["batch_size"])),
        "align_dim": (int(foundation.get("align_dim", -1)), int(config["align_dim"])),
        "loss": (foundation.get("loss"), config["foundation_loss"]),
        "loss_weight": (float(foundation.get("loss_weight", -1)), float(config["lambda"])),
        "teacher": (foundation.get("teacher"), "dinov3"),
        "teacher_dtype": (foundation.get("dtype"), config["teacher_dtype"]),
    }
    mismatch = {name: [actual, expected] for name, (actual, expected) in locks.items() if actual != expected}
    if mismatch:
        raise RuntimeError(f"checkpoint lock mismatch for seed {seed}: {json.dumps(mismatch)}")


def cache_teacher_features(
    config: dict[str, Any],
    batches: list[dict[str, Any]],
    path_batches: list[list[str]],
    seed: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], dict[tuple[int, int], tuple[torch.Tensor, list[dict[str, Any]]]], dict[str, Any]]:
    """Cache each clean batch once and every fixed perturbation once for one seed."""
    from ultralytics.nn.foundation import DINOv3Teacher, apply_response_field_condition_batch

    teacher = DINOv3Teacher(
        model_id=str(config["teacher_model"]),
        weights_path=Path(config["teacher_weights"]),
        dtype=str(config["teacher_dtype"]),
        device=device,
        local_files_only=True,
    )
    clean_cache: list[torch.Tensor] = []
    perturbed_cache: dict[tuple[int, int], tuple[torch.Tensor, list[dict[str, Any]]]] = {}
    teacher_metadata: dict[str, Any] = {}
    with torch.inference_mode():
        for batch_index, (cpu_batch, paths) in enumerate(zip(batches, path_batches)):
            clean_cpu = cpu_batch["img"].float().div(255.0)
            clean_output = teacher.encode(clean_cpu.to(device))
            clean_feature = clean_output.dense["p4"].detach().cpu()
            if not torch.isfinite(clean_feature).all():
                raise ValueError("Teacher clean P4 contains NaN or Inf")
            clean_cache.append(clean_feature)
            teacher_metadata = dict(clean_output.metadata)
            for condition_index, (family, value, condition_id) in enumerate(config["conditions"]):
                perturbed_cpu, manifest = apply_response_field_condition_batch(
                    clean_cpu,
                    paths,
                    family=str(family),
                    value=float(value),
                    condition_id=str(condition_id),
                    seed=seed,
                    epoch_index=int(config["calibration_epoch_index"]),
                    batch_index_within_epoch=batch_index,
                    num_batches_per_epoch=int(config["num_batches_per_epoch"]),
                )
                response = teacher.encode(perturbed_cpu.to(device))
                feature = response.dense["p4"].detach().cpu()
                if not torch.isfinite(feature).all():
                    raise ValueError("Teacher perturbed P4 contains NaN or Inf")
                perturbed_cache[(batch_index, condition_index)] = (feature, manifest)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return clean_cache, perturbed_cache, teacher_metadata


def analytic_pair_gradient_norm(
    static_clean_sq: float, response_clean_sq: float, response_perturbed_sq: float, cross: float, alpha: float
) -> float:
    """Return ||(g_static_clean + alpha*g_response_clean, alpha*g_response_perturbed)||."""
    squared = static_clean_sq + 2.0 * alpha * cross + alpha * alpha * (response_clean_sq + response_perturbed_sq)
    tolerance = 1e-10 * max(1.0, static_clean_sq, alpha * alpha * (response_clean_sq + response_perturbed_sq))
    if squared < -tolerance:
        raise ValueError(f"analytic gradient norm squared is negative: {squared}")
    return math.sqrt(max(0.0, squared))


def raw_to_candidate(row: dict[str, Any], alpha: float, weight: float) -> dict[str, float]:
    """Analytically scale one cached raw observation for one alpha."""
    task_loss = float(row["task_loss"])
    task_grad_norm = float(row["task_grad_norm"])
    if task_loss <= 0 or task_grad_norm <= 0:
        raise ValueError("task loss and task gradient norm must be positive")
    q_loss_b = weight * (float(row["static_clean_loss"]) + float(row["static_perturbed_loss"])) / task_loss
    q_loss_c = weight * (float(row["static_clean_loss"]) + alpha * float(row["response_loss"])) / task_loss
    b_norm = weight * math.sqrt(float(row["static_clean_grad_sq"]) + float(row["static_perturbed_grad_sq"]))
    c_norm = weight * analytic_pair_gradient_norm(
        float(row["static_clean_grad_sq"]),
        float(row["response_clean_grad_sq"]),
        float(row["response_perturbed_grad_sq"]),
        float(row["static_response_clean_dot"]),
        alpha,
    )
    return {
        "q_loss_B": q_loss_b,
        "q_loss_C": q_loss_c,
        "q_grad_B": b_norm / task_grad_norm,
        "q_grad_C": c_norm / task_grad_norm,
    }


def summarize_calibration(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen median, acceptance, score, and tie-break rules."""
    seeds = sorted(int(seed) for seed in config["checkpoints"])
    alphas = [float(alpha) for alpha in config["alpha_candidates"]]
    weight = float(config["lambda"])
    low, high = (float(value) for value in config["acceptance_band"])
    minimum = int(config["minimum_passing_seeds"])
    rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    for alpha in alphas:
        seed_rows = []
        for seed in seeds:
            values = [raw_to_candidate(row, alpha, weight) for row in records if int(row["seed"]) == seed]
            if not values:
                raise RuntimeError(f"no raw calibration observations for seed {seed}")
            medians = {key: float(np.median([value[key] for value in values])) for key in values[0]}
            r_loss = medians["q_loss_C"] / medians["q_loss_B"]
            r_grad = medians["q_grad_C"] / medians["q_grad_B"]
            accepted = low <= r_loss <= high and low <= r_grad <= high
            row = {
                "scope": "seed",
                "seed": seed,
                "alpha": alpha,
                "observations": len(values),
                **medians,
                "r_loss": r_loss,
                "r_grad": r_grad,
                "score": "",
                "accepted": accepted,
                "passing_seed_count": "",
                "reject_reason": "" if accepted else "seed_ratio_outside_frozen_band",
            }
            rows.append(row)
            seed_rows.append(row)
        pooled_values = [raw_to_candidate(row, alpha, weight) for row in records]
        medians = {key: float(np.median([value[key] for value in pooled_values])) for key in pooled_values[0]}
        r_loss = medians["q_loss_C"] / medians["q_loss_B"]
        r_grad = medians["q_grad_C"] / medians["q_grad_B"]
        passing = sum(bool(row["accepted"]) for row in seed_rows)
        pooled_band = low <= r_loss <= high and low <= r_grad <= high
        accepted = pooled_band and passing >= minimum
        reasons = []
        if not pooled_band:
            reasons.append("pooled_ratio_outside_frozen_band")
        if passing < minimum:
            reasons.append("fewer_than_two_seeds_in_frozen_band")
        pooled = {
            "scope": "pooled",
            "seed": "",
            "alpha": alpha,
            "observations": len(pooled_values),
            **medians,
            "r_loss": r_loss,
            "r_grad": r_grad,
            "score": abs(math.log(r_loss)) + abs(math.log(r_grad)),
            "accepted": accepted,
            "passing_seed_count": passing,
            "reject_reason": ";".join(reasons),
        }
        rows.append(pooled)
        pooled_rows.append(pooled)
    accepted = [row for row in pooled_rows if row["accepted"]]
    accepted.sort(key=lambda row: (float(row["score"]), abs(float(row["alpha"]) - 1.0), float(row["alpha"])))
    winner = accepted[0] if accepted else None
    result = {
        "status": "passed" if winner else "failed",
        "selected_alpha": None if winner is None else float(winner["alpha"]),
        "selected_alpha_label": None if winner is None else "signal-matched alpha",
        "formal_training_authorized": False,
        "accepted_alpha_count": len(accepted),
        "accepted_alphas": [float(row["alpha"]) for row in accepted],
        "selection_rule": "minimum_abs_log_r_loss_plus_abs_log_r_grad_then_closest_to_one_then_smaller_alpha",
        "failure_action": "stop_p2_04_no_formal_training" if winner is None else None,
    }
    return rows, result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a stable UTF-8 CSV."""
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure_seed(
    seed: int,
    checkpoint_spec: dict[str, Any],
    config: dict[str, Any],
    batches: list[dict[str, Any]],
    clean_teacher: list[torch.Tensor],
    perturbed_teacher: dict[tuple[int, int], tuple[torch.Tensor, list[dict[str, Any]]]],
    device: torch.device,
    log: AuditLog,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Measure cached raw loss/gradient components for one frozen EMA checkpoint."""
    from ultralytics.nn.foundation import (
        BatchNormBufferSnapshot,
        StudentFeatureTap,
        apply_response_field_condition_batch,
        preserve_batchnorm_buffers,
        response_field_kd_loss,
        strict_cosine_kd_loss,
    )

    checkpoint_path = Path(checkpoint_spec["path"])
    file_before = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, config, seed)
    wrapper = checkpoint["ema"].float().to(device)
    student = wrapper.student_model
    projector = wrapper.projector_for("p4")
    student.train()
    projector.train()
    roots = {"student": student, "projector": projector}
    initial_state = state_digest(wrapper)
    initial_tensor_states = state_tensor_digests(wrapper)
    if any(parameter.grad is not None for parameter in wrapper.parameters()):
        raise RuntimeError("checkpoint entered calibration with populated parameter gradients")
    tap = StudentFeatureTap(student, target="p4")
    rows: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    autograd_calls = 0
    try:
        for batch_index, cpu_batch in enumerate(batches):
            for condition_index, (_, _, condition_id) in enumerate(config["conditions"]):
                batch = batch_to_device(cpu_batch, device)
                clean = batch["img"].detach().requires_grad_(True)
                teacher_clean = clean_teacher[batch_index].to(device)
                teacher_perturbed, input_manifest = perturbed_teacher[(batch_index, condition_index)]
                teacher_perturbed = teacher_perturbed.to(device)
                initial_bn = BatchNormBufferSnapshot(roots)
                tap.clear()
                predictions = student(clean)
                clean_feature = tap.feature
                task_vector, task_items = student.loss(batch, predictions)
                task_loss = task_vector.sum() / int(clean.shape[0])
                clean_aligned, teacher_clean_aligned = projector(clean_feature, teacher_clean)
                post_clean_bn = BatchNormBufferSnapshot(roots)
                with preserve_batchnorm_buffers(roots):
                    tap.clear()
                    family, value, frozen_condition_id = config["conditions"][condition_index]
                    normalized_paths = [str(record["image_id"]) for record in input_manifest]
                    clean_cpu = cpu_batch["img"].float().div(255.0)
                    perturbed_cpu, regenerated_manifest = apply_response_field_condition_batch(
                        clean_cpu,
                        normalized_paths,
                        family=str(family),
                        value=float(value),
                        condition_id=str(frozen_condition_id),
                        seed=seed,
                        epoch_index=int(config["calibration_epoch_index"]),
                        batch_index_within_epoch=batch_index,
                        num_batches_per_epoch=int(config["num_batches_per_epoch"]),
                    )
                    if regenerated_manifest != input_manifest:
                        raise RuntimeError("cached perturbation manifest could not be reproduced exactly")
                    perturbed = perturbed_cpu.to(device).requires_grad_(True)
                    student(perturbed)
                    perturbed_feature = tap.feature
                    perturbed_aligned, teacher_perturbed_aligned = projector(perturbed_feature, teacher_perturbed)
                if not post_clean_bn.matches():
                    raise RuntimeError("perturbed forward was not restored to the post-clean BN state")
                static_clean = strict_cosine_kd_loss(
                    clean_aligned, teacher_clean_aligned, eps=float(config["cosine_epsilon"])
                )
                static_perturbed = strict_cosine_kd_loss(
                    perturbed_aligned, teacher_perturbed_aligned, eps=float(config["cosine_epsilon"])
                )
                response = response_field_kd_loss(
                    clean_aligned,
                    perturbed_aligned,
                    teacher_clean_aligned,
                    teacher_perturbed_aligned,
                    eps=float(config["cosine_epsilon"]),
                )
                if not all(torch.isfinite(value) for value in (task_loss, static_clean, static_perturbed, response)):
                    raise ValueError("calibration loss contains NaN or Inf")
                g_task = torch.autograd.grad(task_loss, clean_feature, retain_graph=True, allow_unused=False)[0]
                g_static_clean = torch.autograd.grad(
                    static_clean, clean_feature, retain_graph=True, allow_unused=False
                )[0]
                g_static_perturbed = torch.autograd.grad(
                    static_perturbed, perturbed_feature, retain_graph=True, allow_unused=False
                )[0]
                g_response_clean, g_response_perturbed = torch.autograd.grad(
                    response, (clean_feature, perturbed_feature), retain_graph=False, allow_unused=False
                )
                autograd_calls += 4
                gradients = (g_task, g_static_clean, g_static_perturbed, g_response_clean, g_response_perturbed)
                if not all(torch.isfinite(gradient).all() for gradient in gradients):
                    raise ValueError("calibration gradient contains NaN or Inf")
                norms_sq = [float(gradient.detach().float().square().sum()) for gradient in gradients]
                epsilon = float(config["gradient_norm_epsilon"])
                if min(norms_sq) <= epsilon * epsilon:
                    raise ValueError(f"zero-norm calibration gradient at seed={seed}, batch={batch_index}")
                cross = float((g_static_clean.detach().float() * g_response_clean.detach().float()).sum())
                row = {
                    "seed": seed,
                    "batch_index": batch_index,
                    "condition_index": condition_index,
                    "condition_id": str(condition_id),
                    "batch_images": len(input_manifest),
                    "task_loss": float(task_loss.detach()),
                    "static_clean_loss": float(static_clean.detach()),
                    "static_perturbed_loss": float(static_perturbed.detach()),
                    "response_loss": float(response.detach()),
                    "task_grad_norm": math.sqrt(norms_sq[0]),
                    "static_clean_grad_sq": norms_sq[1],
                    "static_perturbed_grad_sq": norms_sq[2],
                    "response_clean_grad_sq": norms_sq[3],
                    "response_perturbed_grad_sq": norms_sq[4],
                    "static_response_clean_dot": cross,
                    "second_term_loss_ratio_response_over_static_perturbed": float(response.detach())
                    / float(static_perturbed.detach()),
                    "second_term_grad_ratio_response_over_static_perturbed": math.sqrt(norms_sq[3] + norms_sq[4])
                    / math.sqrt(norms_sq[2]),
                    "teacher_clean_feature_sha256": tensor_digest(teacher_clean),
                    "teacher_perturbed_feature_sha256": tensor_digest(teacher_perturbed),
                    "input_manifest_sha256": hashlib.sha256(
                        json.dumps(input_manifest, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
                rows.append(row)
                manifest_records.extend({"seed": seed, **record} for record in input_manifest)
                initial_bn.restore()
                if not initial_bn.matches():
                    raise RuntimeError("calibration observation did not restore its initial BN state")
                del (
                    batch,
                    clean,
                    predictions,
                    clean_feature,
                    task_vector,
                    task_items,
                    task_loss,
                    clean_aligned,
                    teacher_clean_aligned,
                    perturbed,
                    perturbed_feature,
                    perturbed_aligned,
                    teacher_perturbed_aligned,
                    static_clean,
                    static_perturbed,
                    response,
                    g_task,
                    g_static_clean,
                    g_static_perturbed,
                    g_response_clean,
                    g_response_perturbed,
                )
            log.write("seed_batch_complete", seed=seed, batch_index=batch_index, observations=len(rows))
    finally:
        tap.close()
    final_state = state_digest(wrapper)
    final_tensor_states = state_tensor_digests(wrapper)
    changed_state_keys = [
        name for name, digest in initial_tensor_states.items() if final_tensor_states.get(name) != digest
    ]
    changed_state_keys.extend(name for name in final_tensor_states if name not in initial_tensor_states)
    file_after = sha256(checkpoint_path)
    no_parameter_grads = all(parameter.grad is None for parameter in wrapper.parameters())
    runtime = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256_before": file_before,
        "checkpoint_sha256_after": file_after,
        "checkpoint_file_unchanged": file_before == file_after == str(checkpoint_spec["sha256"]),
        "state_sha256_before": initial_state,
        "state_sha256_after": final_state,
        "checkpoint_state_unchanged": initial_state == final_state,
        "changed_state_keys": changed_state_keys,
        "parameter_grads_remain_none": no_parameter_grads,
        "autograd_grad_calls": autograd_calls,
        "optimizer_steps": 0,
        "student_forward_batches": len(batches) * len(config["conditions"]) * 2,
        "teacher_forward_batches": len(batches) * (1 + len(config["conditions"])),
    }
    if not all(
        runtime[key]
        for key in ("checkpoint_file_unchanged", "checkpoint_state_unchanged", "parameter_grads_remain_none")
    ):
        raise RuntimeError(f"checkpoint mutation audit failed for seed {seed}: {runtime}")
    del wrapper, student, projector, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, runtime, manifest_records


def main() -> None:
    """Execute input audit or the authorized calibration and archive evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = load_yaml(config_path)
    frozen = validate_frozen_inputs(repo_root, config_path, config)
    p1 = load_yaml(resolve_repo_path(repo_root, str(config["p1_config"])))
    validate_p1_config(config, p1)
    if args.audit_only:
        print(json.dumps({"status": "audit_passed", "frozen": frozen}, indent=2, default=str))
        return

    source = git_state(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime(output_dir)
    log_path = output_dir / "calibration.log"
    log = AuditLog(log_path)
    log.write("calibration_started", source_commit=source["commit"], authorization="train64_only")
    random.seed(20260903)
    np.random.seed(20260903)
    torch.manual_seed(20260903)
    torch.cuda.manual_seed_all(20260903)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA calibration was requested but CUDA is unavailable")
    normalized, selected_paths, dataset_root = resolve_selected_paths(repo_root, config)
    batches, path_batches, dataset = build_batches(repo_root, config, normalized, selected_paths)
    dataset["dataset_root"] = str(dataset_root)
    all_records: list[dict[str, Any]] = []
    checkpoint_runtime: dict[str, Any] = {}
    rolling_manifest: list[dict[str, Any]] = []
    teacher_metadata: dict[str, Any] = {}
    for seed in sorted(frozen["checkpoints"]):
        log.write("teacher_cache_started", seed=seed)
        clean_teacher, perturbed_teacher, teacher_metadata = cache_teacher_features(
            config, batches, path_batches, seed, device
        )
        log.write(
            "teacher_cache_complete",
            seed=seed,
            clean_batches=len(clean_teacher),
            perturbed_batches=len(perturbed_teacher),
        )
        rows, runtime, manifests = measure_seed(
            seed,
            frozen["checkpoints"][seed],
            config,
            batches,
            clean_teacher,
            perturbed_teacher,
            device,
            log,
        )
        all_records.extend(rows)
        checkpoint_runtime[str(seed)] = runtime
        rolling_manifest.extend(manifests)
        log.write("seed_complete", seed=seed, observations=len(rows), state_unchanged=True)
        del clean_teacher, perturbed_teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    expected = len(config["checkpoints"]) * len(batches) * len(config["conditions"])
    if len(all_records) != expected:
        raise RuntimeError(f"raw observation count mismatch: {len(all_records)} != {expected}")
    summaries, decision = summarize_calibration(all_records, config)
    raw_path = output_dir / "calibration_raw.csv"
    summary_path = output_dir / "calibration_summary.csv"
    result_path = output_dir / "calibration_result.json"
    manifest_path = output_dir / "calibration_manifest.json"
    write_csv(raw_path, all_records)
    write_csv(summary_path, summaries)
    checks = {
        "train_split_only": dataset["split"] == "train",
        "train64_sha256_exact": sha256(resolve_repo_path(repo_root, str(config["train_subset"])))
        == str(config["train_subset_sha256"]),
        "all_64_images_x_8_conditions_x_3_seeds": len(rolling_manifest)
        == int(config["subset_size"]) * len(config["conditions"]) * len(config["checkpoints"]),
        "alpha_uses_cached_raw_values_only": True,
        "no_validation_access": True,
        "no_response128_access": True,
        "optimizer_steps": 0,
        "checkpoint_state_unchanged": all(value["checkpoint_state_unchanged"] for value in checkpoint_runtime.values()),
        "checkpoint_files_unchanged": all(value["checkpoint_file_unchanged"] for value in checkpoint_runtime.values()),
        "parameter_grads_remain_none": all(
            value["parameter_grads_remain_none"] for value in checkpoint_runtime.values()
        ),
        "formal_training_started": False,
    }
    if not all(value is True or value == 0 for value in checks.values()):
        raise RuntimeError(f"calibration audit checks failed: {checks}")
    result = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **decision,
        "claim_boundary": "train64-only signal calibration; selected alpha is signal-matched, not optimal",
        "lambda": float(config["lambda"]),
        "alpha_candidates": [float(value) for value in config["alpha_candidates"]],
        "acceptance_band": [float(value) for value in config["acceptance_band"]],
        "minimum_passing_seeds": int(config["minimum_passing_seeds"]),
        "summaries": summaries,
        "checks": checks,
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.write(
        "calibration_complete",
        status=decision["status"],
        selected_alpha=decision["selected_alpha"],
        formal_training_authorized=False,
    )
    script_path = Path(__file__).resolve()
    response_path = repo_root / "ultralytics/nn/foundation/response.py"
    manifest = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_state": source,
        "frozen_inputs": frozen["inputs"],
        "implementation": {
            "response.py": {"path": str(response_path), "sha256": sha256(response_path)},
            "calibration_script": {"path": str(script_path), "sha256": sha256(script_path)},
            "calibration_config": {"path": str(config_path), "sha256": sha256(config_path)},
        },
        "dataset": dataset,
        "teacher": {
            "model": config["teacher_model"],
            "dtype": config["teacher_dtype"],
            "metadata": teacher_metadata,
            "assets": frozen["teacher_assets"],
        },
        "checkpoints": checkpoint_runtime,
        "execution": {
            "observation_unit": config["observation_unit"],
            "raw_observations": len(all_records),
            "rolling_manifest_records": len(rolling_manifest),
            "rolling_manifest_sha256": hashlib.sha256(
                json.dumps(rolling_manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "optimizer_steps": 0,
            "alpha_forward_reruns": 0,
            "no_validation_access": True,
            "no_response128_access": True,
            "formal_training_started": False,
        },
        "decision": decision,
        "checks": checks,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "artifacts": {
            "calibration_raw.csv": {"sha256": sha256(raw_path)},
            "calibration_summary.csv": {"sha256": sha256(summary_path)},
            "calibration_result.json": {"sha256": sha256(result_path)},
            "calibration.log": {"sha256": sha256(log_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], "selected_alpha": decision["selected_alpha"]}, indent=2))


if __name__ == "__main__":
    main()
