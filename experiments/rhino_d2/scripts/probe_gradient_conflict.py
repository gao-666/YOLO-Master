#!/usr/bin/env python3
"""Probe P4 detection/KD gradient conflict on frozen DINOv3-S P1 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_lines_sha256(lines: list[str]) -> str:
    """Hash non-empty lines using a stable LF-terminated representation."""
    payload = "".join(f"{line.strip()}\n" for line in lines if line.strip()).encode()
    return hashlib.sha256(payload).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a YAML mapping: {path}")
    return value


def configure_runtime(output_dir: Path) -> None:
    """Keep mutable library caches inside the experiment output tree."""
    runtime = output_dir / "runtime"
    yolo_config = runtime / "yolo_config"
    matplotlib_config = runtime / "matplotlib"
    yolo_config.mkdir(parents=True, exist_ok=True)
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def git_state(repo_root: Path, expected_commit: str) -> dict[str, Any]:
    """Verify the frozen P1 commit and report local state without changing it."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected_commit:
        raise RuntimeError(f"P1 commit mismatch: expected {expected_commit}, found {commit}")
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    protected = [
        "experiments/rhino_d2/DINOV3_P1_GO_NO_GO.md",
        "experiments/rhino_d2/configs/d2_v3_p1_on.yaml",
        "ultralytics/nn/foundation",
        "ultralytics/nn/foundation_distill_model.py",
    ]
    protected_diff = subprocess.run(
        ["git", "diff", "--name-only", expected_commit, "--", *protected],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if protected_diff.strip():
        raise RuntimeError(f"protected P1 inputs differ from the frozen commit:\n{protected_diff}")
    return {
        "commit": commit,
        "dirty": bool(porcelain.strip()),
        "porcelain_sha256": hashlib.sha256(porcelain.encode()).hexdigest(),
        "protected_inputs_clean": True,
    }


def validate_p1_config(config: dict[str, Any], p1: dict[str, Any]) -> None:
    """Fail closed if the requested probe no longer matches frozen P1 conditions."""
    expected = {
        "foundation_enabled": True,
        "foundation_teacher": "dinov3",
        "foundation_model": config["teacher_model"],
        "foundation_weights": config["teacher_weights"],
        "foundation_target_levels": [config["student_level"]],
        "foundation_align_dim": int(config["align_dim"]),
        "foundation_loss": config["foundation_loss"],
        "foundation_loss_weight": float(config["foundation_loss_weight"]),
        "foundation_teacher_dtype": config["teacher_dtype"],
        "imgsz": int(config["imgsz"]),
        "batch": int(config["batch_size"]),
        "epochs": 50,
        "optimizer": "SGD",
    }
    mismatches = {
        key: {"expected": value, "actual": p1.get(key)} for key, value in expected.items() if p1.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"P1 lock mismatch: {json.dumps(mismatches, ensure_ascii=False)}")


def select_subset(entries: list[str], size: int, selection_seed: str) -> list[str]:
    """Select a deterministic SHA-256-ranked subset without replacement."""
    normalized = [entry.strip().replace("\\", "/") for entry in entries if entry.strip()]
    if len(normalized) != len(set(normalized)):
        raise ValueError("training list contains duplicate paths")
    if size <= 0 or size > len(normalized):
        raise ValueError(f"subset size must be in [1, {len(normalized)}], got {size}")
    return sorted(
        normalized,
        key=lambda value: (
            hashlib.sha256(f"{selection_seed}\0{value}".encode()).hexdigest(),
            value,
        ),
    )[:size]


def resolve_dataset_entries(data: dict[str, Any], selected: list[str]) -> list[Path]:
    """Resolve selected dataset-list entries beneath the checked dataset root."""
    dataset_root = Path(data["path"]).resolve()
    resolved = []
    for entry in selected:
        relative = entry.removeprefix("./")
        path = (dataset_root / relative).resolve()
        if dataset_root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"selected training image is missing or outside the dataset root: {path}")
        resolved.append(path)
    return resolved


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move one collated detection batch to the probe device and normalize images."""
    moved = {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    moved["img"] = moved["img"].float().div_(255.0)
    return moved


def build_batches(
    repo_root: Path,
    config: dict[str, Any],
    selected_paths: list[Path],
) -> tuple[list[dict], dict]:
    """Build deterministic, augmentation-free batches from the frozen train subset."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data_path = (repo_root / config["data"]).resolve()
    data = check_det_dataset(str(data_path))
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
        args,
        data["train"],
        int(config["batch_size"]),
        data,
        mode="val",
        rect=False,
        stride=32,
    )
    index_by_path = {Path(path).resolve(): index for index, path in enumerate(dataset.im_files)}
    missing = [path for path in selected_paths if path.resolve() not in index_by_path]
    if missing:
        raise RuntimeError(f"selected images disappeared from the frozen training dataset: {missing[:3]}")
    selected_indices = [index_by_path[path.resolve()] for path in selected_paths]
    batches = []
    batch_size = int(config["batch_size"])
    for start in range(0, len(selected_indices), batch_size):
        items = [dataset[index] for index in selected_indices[start : start + batch_size]]
        batches.append(dataset.collate_fn(items))
    metadata = {
        "dataset_root": str(Path(data["path"]).resolve()),
        "dataset_order": [path.relative_to(Path(data["path"]).resolve()).as_posix() for path in selected_paths],
        "batch_count": len(batches),
        "full_dataset_count": len(dataset),
    }
    return batches, metadata


def teacher_inventory(weights: Path) -> list[dict[str, Any]]:
    """Hash version-sensitive local teacher assets."""
    names = (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "LICENSE.md",
        "README.md",
    )
    inventory = []
    for name in names:
        path = weights / name
        if not path.is_file():
            raise FileNotFoundError(f"missing teacher asset: {path}")
        inventory.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return inventory


def cache_teacher_features(
    config: dict[str, Any], batches: list[dict], device: torch.device
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Run the frozen teacher once per fixed batch and cache raw P4 features on CPU."""
    from ultralytics.nn.foundation import DINOv3Teacher

    teacher = DINOv3Teacher(
        model_id=str(config["teacher_model"]),
        weights_path=Path(config["teacher_weights"]),
        dtype=str(config["teacher_dtype"]),
        device=device,
        local_files_only=True,
    )
    cached = []
    metadata = None
    with torch.inference_mode():
        for batch in batches:
            image = batch["img"].to(device).float().div(255.0)
            response = teacher.encode(image)
            feature = response.dense["p4"]
            if not torch.isfinite(feature).all():
                raise ValueError("teacher P4 feature contains NaN or Inf")
            cached.append(feature.detach().cpu())
            metadata = response.metadata
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cached, dict(metadata or {})


def freeze_batchnorm(module: nn.Module) -> int:
    """Set BatchNorm modules to eval while leaving the detection head in training mode."""
    count = 0
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()
            count += 1
    return count


def gradient_metrics(g_task: torch.Tensor, g_kd: torch.Tensor, epsilon: float) -> list[dict[str, float | bool]]:
    """Return per-image gradient direction and strength statistics."""
    if g_task.shape != g_kd.shape or g_task.ndim < 2:
        raise ValueError(f"gradient shapes must match and include a batch dimension: {g_task.shape}, {g_kd.shape}")
    task = g_task.detach().float().reshape(g_task.shape[0], -1)
    kd = g_kd.detach().float().reshape(g_kd.shape[0], -1)
    dot = (task * kd).sum(dim=1)
    task_norm = torch.linalg.vector_norm(task, dim=1)
    kd_norm = torch.linalg.vector_norm(kd, dim=1)
    denominator = task_norm * kd_norm
    valid = denominator > float(epsilon)
    cosine = torch.full_like(denominator, float("nan"))
    ratio = torch.full_like(denominator, float("nan"))
    cosine[valid] = dot[valid] / denominator[valid]
    ratio[valid] = kd_norm[valid] / task_norm[valid]
    return [
        {
            "valid": bool(valid[index]),
            "cosine": float(cosine[index]),
            "norm_ratio": float(ratio[index]),
            "task_norm": float(task_norm[index]),
            "kd_norm": float(kd_norm[index]),
            "dot": float(dot[index]),
        }
        for index in range(g_task.shape[0])
    ]


def checkpoint_path(repo_root: Path, config: dict[str, Any], seed: int, epoch: int) -> Path:
    """Resolve one checkpoint from the frozen path template."""
    relative = str(config["checkpoint_pattern"]).format(seed=seed, epoch=epoch)
    path = (repo_root / relative).resolve()
    if repo_root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return path


def validate_checkpoint(checkpoint: dict[str, Any], config: dict[str, Any], seed: int, epoch: int) -> None:
    """Validate checkpoint identity and all P1 Foundation locks."""
    if int(checkpoint.get("epoch", -1)) != epoch:
        raise RuntimeError(f"checkpoint epoch mismatch for seed {seed}: {checkpoint.get('epoch')} != {epoch}")
    args = checkpoint.get("train_args") or {}
    foundation = checkpoint.get("foundation") or {}
    locks = {
        "seed": (int(args.get("seed", -1)), seed),
        "imgsz": (int(args.get("imgsz", -1)), int(config["imgsz"])),
        "batch": (int(args.get("batch", -1)), int(config["batch_size"])),
        "align_dim": (int(foundation.get("align_dim", -1)), int(config["align_dim"])),
        "loss": (foundation.get("loss"), config["foundation_loss"]),
        "loss_weight": (
            float(foundation.get("loss_weight", -1)),
            float(config["foundation_loss_weight"]),
        ),
        "teacher": (foundation.get("teacher"), "dinov3"),
        "teacher_dtype": (foundation.get("dtype"), config["teacher_dtype"]),
    }
    mismatches = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in locks.items()
        if actual != expected
    }
    if mismatches:
        raise RuntimeError(f"checkpoint P1 lock mismatch: {json.dumps(mismatches)}")
    if checkpoint.get("ema") is None:
        raise RuntimeError("checkpoint has no EMA model")


def probe_checkpoint(
    checkpoint_file: Path,
    config: dict[str, Any],
    seed: int,
    epoch_label: str,
    epoch: int,
    batches: list[dict],
    teacher_features: list[torch.Tensor],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure task/KD P4 gradients for one fixed EMA checkpoint."""
    from ultralytics.nn.foundation import StudentFeatureTap, cosine_kd_loss

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, config, seed, epoch)
    wrapper = checkpoint["ema"].float().to(device)
    student = wrapper.student_model
    projector = wrapper.projector_for("p4")
    student.train()
    projector.eval()
    trainable_parameter_count = 0
    for parameter in student.parameters():
        parameter.requires_grad_(True)
        trainable_parameter_count += parameter.numel()
    batchnorm_count = freeze_batchnorm(student) + freeze_batchnorm(projector)
    tap = StudentFeatureTap(student, target="p4")
    rows = []
    try:
        for batch_index, (cpu_batch, cpu_teacher) in enumerate(zip(batches, teacher_features)):
            batch = batch_to_device(cpu_batch, device)
            teacher_feature = cpu_teacher.to(device)
            tap.clear()
            predictions = student(batch["img"])
            feature = tap.feature
            task_loss, task_items = student.loss(batch, predictions)
            task_scalar = task_loss.sum()
            student_aligned, teacher_aligned = projector(feature, teacher_feature)
            kd_raw = cosine_kd_loss(student_aligned, teacher_aligned)
            kd_weighted = kd_raw * float(config["foundation_loss_weight"]) * int(batch["img"].shape[0])
            if not torch.isfinite(task_scalar) or not torch.isfinite(kd_weighted):
                raise ValueError(f"non-finite loss at seed={seed}, epoch={epoch}, batch={batch_index}")
            g_task = torch.autograd.grad(task_scalar, feature, retain_graph=True, allow_unused=False)[0]
            g_kd = torch.autograd.grad(kd_weighted, feature, retain_graph=False, allow_unused=False)[0]
            metrics = gradient_metrics(g_task, g_kd, float(config["gradient"]["norm_epsilon"]))
            image_files = list(batch["im_file"])
            if len(metrics) != len(image_files):
                raise RuntimeError("per-image gradient metric count does not match the batch")
            item_values = [float(value) for value in task_items.detach().float().reshape(-1).cpu()]
            for sample_index, (image_file, values) in enumerate(zip(image_files, metrics)):
                rows.append(
                    {
                        "seed": seed,
                        "epoch_label": epoch_label,
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "sample_index": sample_index,
                        "image": Path(image_file).as_posix(),
                        **values,
                        "task_loss_batch": float(task_scalar.detach()),
                        "kd_raw_batch": float(kd_raw.detach()),
                        "kd_weighted_batch": float(kd_weighted.detach()),
                        "task_items": json.dumps(item_values, separators=(",", ":")),
                    }
                )
            del (
                predictions,
                feature,
                task_loss,
                task_items,
                task_scalar,
                student_aligned,
                teacher_aligned,
            )
            del kd_raw, kd_weighted, g_task, g_kd, batch, teacher_feature
    finally:
        tap.close()
        del wrapper, student, projector, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows, {
        "batchnorm_modules_frozen": batchnorm_count,
        "student_parameters_marked_for_autograd": trainable_parameter_count,
        "optimizer_steps": 0,
    }


def bootstrap_median_ci(values: np.ndarray, replicates: int, seed: int, confidence: float) -> tuple[float, float]:
    """Return a percentile bootstrap interval for the median."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.empty(replicates, dtype=np.float64)
    chunk = 1000
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        samples = rng.choice(values, size=(count, len(values)), replace=True)
        medians[start : start + count] = np.median(samples, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(medians, alpha)), float(np.quantile(medians, 1.0 - alpha))


def summarize_values(values: list[float], replicates: int, seed: int, confidence: float) -> dict[str, float | int]:
    """Summarize one finite vector and bootstrap its median."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    low, high = bootstrap_median_ci(array, replicates, seed, confidence)
    return {
        "n": len(array),
        "mean": float(np.mean(array)) if len(array) else float("nan"),
        "median": float(np.median(array)) if len(array) else float("nan"),
        "ci_low": low,
        "ci_high": high,
        "q25": float(np.quantile(array, 0.25)) if len(array) else float("nan"),
        "q75": float(np.quantile(array, 0.75)) if len(array) else float("nan"),
    }


def summarize_records(records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-seed/per-checkpoint summaries from raw image records."""
    groups: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(int(row["seed"]), str(row["epoch_label"]), int(row["epoch"]))].append(row)
    stats = config["statistics"]
    summaries = []
    for group_index, ((seed, label, epoch), rows) in enumerate(sorted(groups.items())):
        valid = [row for row in rows if row["valid"]]
        cosine = summarize_values(
            [row["cosine"] for row in valid],
            int(stats["bootstrap_replicates"]),
            int(stats["bootstrap_seed"]) + group_index,
            float(stats["confidence_level"]),
        )
        ratio = summarize_values(
            [row["norm_ratio"] for row in valid],
            int(stats["bootstrap_replicates"]),
            int(stats["bootstrap_seed"]) + 1000 + group_index,
            float(stats["confidence_level"]),
        )
        summaries.append(
            {
                "seed": seed,
                "epoch_label": label,
                "epoch": epoch,
                "n_total": len(rows),
                "n_valid": len(valid),
                "negative_fraction": float(np.mean([row["cosine"] < 0 for row in valid])) if valid else float("nan"),
                **{f"cosine_{key}": value for key, value in cosine.items()},
                **{f"norm_ratio_{key}": value for key, value in ratio.items()},
            }
        )
    return summaries


def pooled_late_summary(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Cluster late records by image before computing the pooled three-seed summary."""
    late_epoch = int(config["checkpoints"]["late"])
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if int(row["epoch"]) == late_epoch and row["valid"]:
            by_image[str(row["image"])].append(row)
    expected = len(config["seeds"])
    if any(len(rows) != expected for rows in by_image.values()):
        raise RuntimeError("pooled late summary is missing one or more seed/image observations")
    image_cosines = [float(np.mean([row["cosine"] for row in rows])) for rows in by_image.values()]
    image_ratios = [float(np.mean([row["norm_ratio"] for row in rows])) for rows in by_image.values()]
    stats = config["statistics"]
    cosine = summarize_values(
        image_cosines,
        int(stats["bootstrap_replicates"]),
        int(stats["bootstrap_seed"]) + 2000,
        float(stats["confidence_level"]),
    )
    ratio = summarize_values(
        image_ratios,
        int(stats["bootstrap_replicates"]),
        int(stats["bootstrap_seed"]) + 3000,
        float(stats["confidence_level"]),
    )
    return {
        "cluster_unit": "image",
        "seed_observations_per_image": expected,
        **{f"cosine_{key}": value for key, value in cosine.items()},
        **{f"norm_ratio_{key}": value for key, value in ratio.items()},
        "negative_fraction": float(np.mean(np.asarray(image_cosines) < 0)),
    }


def average_ranks(values: list[float]) -> np.ndarray:
    """Return one-based average ranks with deterministic tie handling."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(values_x: list[float], values_y: list[float]) -> float:
    """Return Spearman rank correlation without inferential claims."""
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return float("nan")
    ranks_x = average_ranks(values_x)
    ranks_y = average_ranks(values_y)
    if np.std(ranks_x) == 0 or np.std(ranks_y) == 0:
        return float("nan")
    return float(np.corrcoef(ranks_x, ranks_y)[0, 1])


def decision_from_summaries(summaries: list[dict[str, Any]], pooled: dict[str, Any], late_epoch: int) -> dict[str, Any]:
    """Apply the protocol's frozen support/refutation rule."""
    late = [row for row in summaries if int(row["epoch"]) == int(late_epoch)]
    seed_negative_cis = sum(float(row["cosine_ci_high"]) < 0 for row in late)
    if (
        float(pooled["cosine_ci_upper"] if "cosine_ci_upper" in pooled else pooled["cosine_ci_high"]) < 0
        and seed_negative_cis >= 2
    ):
        status = "supported"
    elif float(pooled["cosine_ci_low"]) >= 0:
        status = "not_supported"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "seed_late_ci_uppers_below_zero": seed_negative_cis,
        "seed_count": len(late),
        "rule": "pooled_late_ci_upper_below_zero_and_at_least_two_seed_late_ci_uppers_below_zero",
        "refutation_rule": "pooled_late_ci_lower_at_or_above_zero",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to a stable UTF-8 CSV."""
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(result: dict[str, Any]) -> str:
    """Render a concise local evidence report from machine-readable results."""
    lines = [
        "# DINOv3-S P2-01 梯度冲突诊断结果",
        "",
        f"判定：**{result['decision']['status']}**。该判定只属于 P2-01，不修改 P1 No-Go。",
        "",
        "| seed | checkpoint | median cosine | 95% CI | negative | median norm ratio |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summaries"]:
        lines.append(
            f"| {row['seed']} | {row['epoch_label']} (epoch{row['epoch']}) | {row['cosine_median']:.6f} | "
            f"[{row['cosine_ci_low']:.6f}, {row['cosine_ci_high']:.6f}] | {row['negative_fraction']:.1%} | "
            f"{row['norm_ratio_median']:.6f} |"
        )
    pooled = result["pooled_late"]
    lines.extend(
        [
            "",
            "## Late pooled（按图像聚类）",
            "",
            f"- median cosine: `{pooled['cosine_median']:.6f}`",
            f"- 95% bootstrap CI: `[{pooled['cosine_ci_low']:.6f}, {pooled['cosine_ci_high']:.6f}]`",
            f"- negative fraction: `{pooled['negative_fraction']:.1%}`",
            f"- median weighted norm ratio: `{pooled['norm_ratio_median']:.6f}`",
            "",
            "## 与 P1 检测差值的描述性对应",
            "",
            (
                f"三个 seed 的 late median cosine 与 P1 ON-OFF 的 Spearman rho 为 "
                f"`{result['p1_descriptive_association']['spearman_rho']:.6f}`。n=3，不作显著性或因果声明。"
            ),
            "",
            "## 边界",
            "",
            "本结果来自固定 EMA checkpoint、固定训练子集和 P4 局部一阶梯度；它不是新训练的性能结果，也不是全训练轨迹的因果证明。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the frozen P2-01 diagnostic and write local evidence without Git operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[3]
    experiment_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--config",
        type=Path,
        default=experiment_root / "configs" / "d2_v3_p2_gradient_conflict.yaml",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=experiment_root / "DINOV3_P2_GRADIENT_CONFLICT_PROTOCOL.md",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_root / "results" / "p2_gradient_conflict",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    config_path = args.config.resolve()
    protocol_path = args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime(output_dir)
    config = load_yaml(config_path)
    source_state = git_state(repo_root, str(config["p1_commit"]))
    p1_path = (repo_root / config["p1_config"]).resolve()
    p1_config = load_yaml(p1_path)
    validate_p1_config(config, p1_config)

    from ultralytics.data.utils import check_det_dataset

    data_path = (repo_root / config["data"]).resolve()
    data = check_det_dataset(str(data_path))
    train_list = Path(data["train"]).resolve()
    train_entries = [line.strip() for line in train_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = select_subset(
        train_entries,
        int(config["subset"]["size"]),
        str(config["subset"]["selection_seed"]),
    )
    selected_paths = resolve_dataset_entries(data, selected)
    checkpoint_files = {
        f"{seed}:{label}": checkpoint_path(repo_root, config, int(seed), int(epoch))
        for seed in config["seeds"]
        for label, epoch in config["checkpoints"].items()
    }
    subset_path = output_dir / "diagnostic_train64.txt"
    subset_path.write_text("".join(f"{entry}\n" for entry in selected), encoding="utf-8")
    preparation = {
        "protocol_id": config["protocol_id"],
        "source_state": source_state,
        "train_list": {
            "path": str(train_list),
            "count": len(train_entries),
            "sha256": sha256(train_list),
        },
        "selection": {
            **config["subset"],
            "selected_list": str(subset_path),
            "selected_list_sha256": sha256(subset_path),
            "selected_count": len(selected),
        },
        "checkpoints": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for key, path in checkpoint_files.items()
        },
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "p1_config": {"path": str(p1_path), "sha256": sha256(p1_path)},
            "data_yaml": {"path": str(data_path), "sha256": sha256(data_path)},
        },
    }
    preparation_path = output_dir / "preparation_manifest.json"
    preparation_path.write_text(json.dumps(preparation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "prepared", **preparation}, indent=2, ensure_ascii=False))
        return

    seed_value = int(config["statistics"]["bootstrap_seed"])
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    batches, dataset_metadata = build_batches(repo_root, config, selected_paths)
    # The dataset may impose a deterministic lexical order. Freeze the actual order used and verify membership.
    if dataset_metadata["dataset_order"] != selected and sorted(dataset_metadata["dataset_order"]) != sorted(
        entry.removeprefix("./") for entry in selected
    ):
        raise RuntimeError("dataset membership changed after building the diagnostic batches")
    teacher_features, teacher_metadata = cache_teacher_features(config, batches, device)
    records = []
    checkpoint_runtime = {}
    for seed in config["seeds"]:
        for label, epoch in config["checkpoints"].items():
            key = f"{seed}:{label}"
            rows, runtime = probe_checkpoint(
                checkpoint_files[key],
                config,
                int(seed),
                str(label),
                int(epoch),
                batches,
                teacher_features,
                device,
            )
            records.extend(rows)
            checkpoint_runtime[key] = runtime
            print(
                json.dumps(
                    {
                        "checkpoint": key,
                        "images": len(rows),
                        "median_cosine": float(np.nanmedian([row["cosine"] for row in rows])),
                        "median_norm_ratio": float(np.nanmedian([row["norm_ratio"] for row in rows])),
                    }
                ),
                flush=True,
            )

    expected_records = len(config["seeds"]) * len(config["checkpoints"]) * int(config["subset"]["size"])
    if len(records) != expected_records:
        raise RuntimeError(f"raw record count mismatch: {len(records)} != {expected_records}")
    if not all(row["valid"] for row in records):
        raise RuntimeError("one or more images produced zero/non-finite task or KD gradient norms")
    summaries = summarize_records(records, config)
    pooled = pooled_late_summary(records, config)
    decision = decision_from_summaries(summaries, pooled, int(config["checkpoints"]["late"]))
    late_by_seed = {
        int(row["seed"]): float(row["cosine_median"])
        for row in summaries
        if int(row["epoch"]) == int(config["checkpoints"]["late"])
    }
    ordered_seeds = [int(seed) for seed in config["seeds"]]
    effect = {int(seed): float(value) for seed, value in config["p1_paired_delta"].items()}
    association = {
        "n": len(ordered_seeds),
        "spearman_rho": spearman(
            [late_by_seed[seed] for seed in ordered_seeds],
            [effect[seed] for seed in ordered_seeds],
        ),
        "inferential_claim": False,
        "seed_values": [
            {
                "seed": seed,
                "late_median_cosine": late_by_seed[seed],
                "p1_on_minus_off": effect[seed],
            }
            for seed in ordered_seeds
        ],
    }
    result = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "completed",
        "claim_boundary": "local_p4_gradient_diagnostic_no_training_no_accuracy_or_causal_claim",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_state": source_state,
        "preparation_manifest": {
            "path": str(preparation_path),
            "sha256": sha256(preparation_path),
        },
        "dataset": dataset_metadata,
        "teacher": {
            "model": config["teacher_model"],
            "weights": config["teacher_weights"],
            "dtype": config["teacher_dtype"],
            "metadata": teacher_metadata,
            "assets": teacher_inventory(Path(config["teacher_weights"])),
        },
        "gradient_definition": config["gradient"],
        "summaries": summaries,
        "pooled_late": pooled,
        "p1_descriptive_association": association,
        "decision": decision,
        "checks": {
            "p1_commit_exact": source_state["commit"] == config["p1_commit"],
            "protected_p1_inputs_clean": source_state["protected_inputs_clean"],
            "uses_training_split_only": config["subset"]["split"] == "train",
            "selected_images": len(selected),
            "expected_raw_records": expected_records,
            "actual_raw_records": len(records),
            "all_gradients_valid": all(row["valid"] for row in records),
            "no_optimizer_steps": True,
            "batchnorm_running_stats_frozen": all(
                value["batchnorm_modules_frozen"] > 0 for value in checkpoint_runtime.values()
            ),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "checkpoint_details": checkpoint_runtime,
        },
    }
    raw_path = output_dir / "gradient_conflict_raw.csv"
    summary_path = output_dir / "gradient_conflict_summary.csv"
    report_path = output_dir / "DINOv3_P2_GRADIENT_CONFLICT_RESULT.md"
    result_path = output_dir / "gradient_conflict_result.json"
    write_csv(raw_path, records)
    write_csv(summary_path, summaries)
    report_path.write_text(markdown_report(result), encoding="utf-8")
    result["artifacts"] = {
        "raw_csv": {"path": str(raw_path), "sha256": sha256(raw_path)},
        "summary_csv": {"path": str(summary_path), "sha256": sha256(summary_path)},
        "report": {"path": str(report_path), "sha256": sha256(report_path)},
        "subset": {"path": str(subset_path), "sha256": sha256(subset_path)},
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": "completed", "decision": decision, "result": str(result_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
