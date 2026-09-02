#!/usr/bin/env python3
"""Run the pre-registered DINOv3 P2-03 static-response-gap diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs/d2_v3_p2_response_gap.yaml"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/p2_response_gap/formal"
CONDITION_KEYS = (
    "brightness:0.8",
    "brightness:0.6",
    "contrast:0.75",
    "contrast:0.5",
    "gaussian_blur:1.0",
    "gaussian_blur:2.0",
    "gaussian_noise:0.03",
    "gaussian_noise:0.06",
)


def sha256(path: Path) -> str:
    """Return one file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected YAML mapping: {path}")
    return value


def configure_runtime(output_dir: Path) -> None:
    """Keep mutable third-party caches beneath the requested output root."""
    runtime = output_dir / "runtime"
    for name in ("yolo_config", "matplotlib", "torch"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime / "yolo_config"))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime / "matplotlib"))
    os.environ.setdefault("TORCH_HOME", str(runtime / "torch"))
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def normalized_condition_specs(config: dict[str, Any]) -> list[tuple[str, str, float]]:
    """Return the eight conditions in their frozen order."""
    perturbations = config["perturbations"]
    specs = [
        *(('brightness', float(v), f"brightness:{float(v):g}") for v in perturbations["brightness"]),
        *(('contrast', float(v), f"contrast:{float(v):g}") for v in perturbations["contrast"]),
        *(("gaussian_blur", float(v), f"gaussian_blur:{float(v):.1f}") for v in perturbations["gaussian_blur_sigma"]),
        *(("gaussian_noise", float(v), f"gaussian_noise:{float(v):.2f}") for v in perturbations["gaussian_noise_std"]),
    ]
    result = [(family, value, condition) for family, value, condition in specs]
    if tuple(condition for _, _, condition in result) != CONDITION_KEYS:
        raise RuntimeError(f"perturbation order or IDs differ from the preregistration: {result}")
    return result


def gaussian_noise_seed(normalized_path: str, perturbation_id: str) -> int:
    """Derive the frozen per-image Gaussian-noise seed."""
    payload = f"dinov3-p2-response-gap-v1\0{normalized_path}\0{perturbation_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def gaussian_kernel1d(sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build the frozen radius=ceil(3*sigma) normalized Gaussian kernel."""
    radius = math.ceil(3.0 * float(sigma))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coordinates.square()) / (2.0 * float(sigma) ** 2))
    return kernel / kernel.sum()


def perturb_images(
    images: torch.Tensor,
    normalized_paths: list[str],
    family: str,
    value: float,
    perturbation_id: str,
) -> torch.Tensor:
    """Apply one deterministic pre-registered perturbation to an FP32 BCHW tensor."""
    if images.device.type != "cpu" or images.dtype != torch.float32 or images.ndim != 4:
        raise ValueError("formal perturbations require a CPU FP32 BCHW tensor")
    if len(normalized_paths) != images.shape[0]:
        raise ValueError("image/path count mismatch")
    if family == "brightness":
        output = images * value
    elif family == "contrast":
        mean = images.mean(dim=(-2, -1), keepdim=True)
        output = mean + value * (images - mean)
    elif family == "gaussian_blur":
        kernel = gaussian_kernel1d(value, device=images.device, dtype=images.dtype)
        radius = kernel.numel() // 2
        channels = images.shape[1]
        horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        output = F.conv2d(F.pad(images, (radius, radius, 0, 0), mode="reflect"), horizontal, groups=channels)
        output = F.conv2d(F.pad(output, (0, 0, radius, radius), mode="reflect"), vertical, groups=channels)
    elif family == "gaussian_noise":
        noise = []
        for path in normalized_paths:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(gaussian_noise_seed(path, perturbation_id))
            noise.append(torch.randn(images.shape[1:], dtype=torch.float32, generator=generator))
        output = images + value * torch.stack(noise)
    else:
        raise ValueError(f"unknown perturbation family: {family}")
    output = output.clamp(0.0, 1.0)
    if not torch.isfinite(output).all():
        raise ValueError(f"non-finite perturbed image: {perturbation_id}")
    return output


def spatial_relation_embedding(feature: torch.Tensor, spatial_size: tuple[int, int] = (16, 16)) -> torch.Tensor:
    """Compute the frozen projection-free strict-upper-triangle spatial-relation embedding."""
    if feature.ndim != 4:
        raise ValueError(f"feature must be BCHW, got {tuple(feature.shape)}")
    feature = feature.float()
    if not torch.isfinite(feature).all():
        raise ValueError("feature contains NaN or Inf")
    feature = F.adaptive_avg_pool2d(feature, spatial_size)
    tokens = feature.flatten(2).transpose(1, 2)
    tokens = tokens - tokens.mean(dim=1, keepdim=True)
    tokens = F.normalize(tokens, p=2, dim=2, eps=1e-12)
    gram = torch.bmm(tokens, tokens.transpose(1, 2))
    indices = torch.triu_indices(gram.shape[1], gram.shape[2], offset=1, device=gram.device)
    embedding = gram[:, indices[0], indices[1]].float()
    if not torch.isfinite(embedding).all():
        raise ValueError("spatial-relation embedding contains NaN or Inf")
    return embedding


def cosine_gap(left: torch.Tensor, right: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Return per-row cosine distance and fail closed on a zero response norm."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(f"cosine inputs must be matching matrices, got {left.shape} and {right.shape}")
    left, right = left.float(), right.float()
    left_norm = torch.linalg.vector_norm(left, dim=1)
    right_norm = torch.linalg.vector_norm(right, dim=1)
    if (left_norm < epsilon).any() or (right_norm < epsilon).any():
        raise ValueError("cosine input norm is below the preregistered fail-closed threshold")
    result = 1.0 - (left * right).sum(dim=1) / (left_norm * right_norm)
    if not torch.isfinite(result).all():
        raise ValueError("cosine gap contains NaN or Inf")
    return result


def rankdata(values: np.ndarray) -> np.ndarray:
    """Assign average ranks with stable tie handling."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("rankdata requires one finite vector")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Compute one finite Spearman correlation without an optional SciPy dependency."""
    left_rank, right_rank = rankdata(left), rankdata(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator <= 0:
        raise ValueError("Spearman correlation is undefined for a constant vector")
    return float(np.dot(left_centered, right_centered) / denominator)


def percentile_ci(values: np.ndarray, confidence: float) -> list[float]:
    """Return a two-sided percentile interval."""
    alpha = (1.0 - confidence) / 2.0
    return [float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))]


def bootstrap_mean_by_image(
    values: np.ndarray, replicates: int, seed: int, confidence: float
) -> dict[str, Any]:
    """Bootstrap a seed-by-image array by resampling image clusters only."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("cluster bootstrap expects a finite seed-by-image matrix")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        image_indices = rng.integers(0, values.shape[1], size=values.shape[1])
        samples[index] = values[:, image_indices].mean()
    return {"estimate": float(values.mean()), "ci95": percentile_ci(samples, confidence)}


def fisher_z(rho: float) -> float:
    """Apply a numerically guarded Fisher transform."""
    return float(np.arctanh(np.clip(rho, -1.0 + 1e-12, 1.0 - 1e-12)))


def h3_statistics(
    static_gap: np.ndarray,
    response_gap: np.ndarray,
    degradation: np.ndarray,
    image_indices: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Return pooled response association, pooled advantage, and six seed-arm advantages."""
    if image_indices is None:
        image_indices = np.arange(static_gap.shape[-1])
    response_z, static_z = [], []
    seed_arm_delta = np.empty((static_gap.shape[0], static_gap.shape[1]), dtype=np.float64)
    for seed in range(static_gap.shape[0]):
        for arm in range(static_gap.shape[1]):
            local_response, local_static = [], []
            for condition in range(response_gap.shape[2]):
                loss = degradation[seed, arm, condition, image_indices]
                local_response.append(fisher_z(spearman(response_gap[seed, arm, condition, image_indices], loss)))
                local_static.append(fisher_z(spearman(static_gap[seed, arm, image_indices], loss)))
            response_z.extend(local_response)
            static_z.extend(local_static)
            seed_arm_delta[seed, arm] = np.mean(local_response) - np.mean(local_static)
    return float(np.mean(response_z)), float(np.mean(response_z) - np.mean(static_z)), seed_arm_delta


def bootstrap_h3(
    static_gap: np.ndarray,
    response_gap: np.ndarray,
    degradation: np.ndarray,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    """Apply the frozen image-cluster bootstrap to all 48 correlation cells."""
    point_response, point_delta, seed_arm_delta = h3_statistics(static_gap, response_gap, degradation)
    rng = np.random.default_rng(seed)
    boot_response = np.empty(replicates, dtype=np.float64)
    boot_delta = np.empty(replicates, dtype=np.float64)
    image_count = static_gap.shape[-1]
    for index in range(replicates):
        image_indices = rng.integers(0, image_count, size=image_count)
        boot_response[index], boot_delta[index], _ = h3_statistics(
            static_gap, response_gap, degradation, image_indices
        )
        if (index + 1) % 1000 == 0:
            print(f"H3 bootstrap {index + 1}/{replicates}", flush=True)
    return {
        "mean_z_response": point_response,
        "mean_z_response_ci95": percentile_ci(boot_response, confidence),
        "delta_rho_z": point_delta,
        "delta_rho_z_ci95": percentile_ci(boot_delta, confidence),
        "seed_arm_delta_rho_z": seed_arm_delta.tolist(),
        "positive_seed_arm_count": int((seed_arm_delta > 0).sum()),
    }


def freeze_batchnorm(module: nn.Module) -> int:
    """Keep every BatchNorm module in eval while raw detection outputs are requested."""
    count = 0
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()
            count += 1
    return count


def state_dict_sha256(module: nn.Module) -> str:
    """Hash model parameters and buffers in a deterministic serialized representation."""
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def batch_to_device(batch: dict[str, Any], images: torch.Tensor, device: torch.device) -> dict[str, Any]:
    """Move a frozen-label batch and one already-normalized image tensor to the device."""
    moved = {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    moved["img"] = images.to(device, non_blocking=True)
    return moved


def single_image_loss_batch(batch: dict[str, Any], index: int) -> dict[str, Any]:
    """Slice exactly one image and its labels for an honest image-level detection loss."""
    mask = batch["batch_idx"].reshape(-1).long() == index
    return {
        "img": batch["img"][index : index + 1],
        "batch_idx": torch.zeros(int(mask.sum()), device=batch["img"].device, dtype=batch["batch_idx"].dtype),
        "cls": batch["cls"][mask],
        "bboxes": batch["bboxes"][mask],
    }


def image_detection_losses(student: nn.Module, batch: dict[str, Any]) -> list[float]:
    """Compute the frozen batch-size-one raw-head loss without updating any state."""
    losses = []
    with torch.no_grad():
        for index in range(batch["img"].shape[0]):
            single = single_image_loss_batch(batch, index)
            student.train()
            freeze_batchnorm(student)
            predictions = student(single["img"])
            # Raw head tensors have already been produced. Return the owner to eval before loss composition so
            # training-only auxiliary bookkeeping cannot mutate persistent diagnostic state.
            student.eval()
            loss, _ = student.loss(single, predictions)
            scalar = float(loss.detach().float().sum().cpu())
            if not math.isfinite(scalar):
                raise ValueError("non-finite per-image detection loss")
            losses.append(scalar)
    return losses


def top10_confidence(outputs: list[dict[str, torch.Tensor]]) -> list[float]:
    """Return the frozen zero-padded post-NMS top-10 confidence mean."""
    result = []
    for output in outputs:
        confidence = output["conf"].float()
        top = torch.topk(confidence, min(10, confidence.numel())).values if confidence.numel() else confidence
        result.append(float(top.sum().cpu()) / 10.0)
    return result


def build_metric_validator(student: nn.Module, device: torch.device, save_dir: Path) -> Any:
    """Create one plot-free diagnostic mAP collector with frozen NMS settings."""
    from ultralytics.models.yolo.detect import DetectionValidator

    validator = DetectionValidator(
        args={
            "task": "detect",
            "mode": "val",
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "multi_scale": False,
            "single_cls": False,
            "agnostic_nms": False,
            "plots": False,
            "visualize": False,
            "save_json": False,
            "save_txt": False,
            "save_conf": False,
            "val": False,
            "split": "val",
        },
        save_dir=save_dir,
    )
    validator.device = device
    validator.data = {"val": ""}
    validator.init_metrics(student)
    return validator


def build_batches(repo_root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Build deterministic augmentation-free batches in the preregistered list order."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    list_path = (repo_root / config["diagnostic_list"]).resolve()
    if sha256(list_path) != config["diagnostic_list_sha256"]:
        raise RuntimeError("diagnostic list SHA-256 mismatch")
    normalized = [line.strip().replace("\\", "/") for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(normalized) != config["subset_size"] or len(normalized) != len(set(normalized)):
        raise RuntimeError("diagnostic list count or uniqueness mismatch")
    excluded_path = (repo_root / config["excluded_list"]).resolve()
    excluded = {line.strip().replace("\\", "/") for line in excluded_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if excluded.intersection(normalized):
        raise RuntimeError("diagnostic list overlaps the P2-01 subset")
    data = check_det_dataset(str((repo_root / config["data"]).resolve()))
    args = get_cfg(overrides={
        "task": "detect",
        "imgsz": int(config["imgsz"]),
        "batch": int(config["feature_batch_size"]),
        "workers": 0,
        "rect": False,
        "cache": False,
        "single_cls": False,
        "classes": None,
        "fraction": 1.0,
    })
    dataset = build_yolo_dataset(args, data["train"], int(config["feature_batch_size"]), data, mode="val", rect=False, stride=32)
    root = Path(data["path"]).resolve()
    selected_paths = [(root / value.removeprefix("./")).resolve() for value in normalized]
    index_by_path = {Path(path).resolve(): index for index, path in enumerate(dataset.im_files)}
    missing = [path for path in selected_paths if path not in index_by_path]
    if missing:
        raise FileNotFoundError(f"diagnostic images missing from dataset: {missing[:3]}")
    indices = [index_by_path[path] for path in selected_paths]
    batches = []
    batch_size = int(config["feature_batch_size"])
    for start in range(0, len(indices), batch_size):
        batches.append(dataset.collate_fn([dataset[index] for index in indices[start : start + batch_size]]))
    return batches, normalized, {"dataset_root": str(root), "full_dataset_count": len(dataset), "batch_count": len(batches)}


def batch_paths(normalized_paths: list[str], batch_index: int, batch_size: int, actual: int) -> list[str]:
    """Return normalized diagnostic-list paths for one order-preserving batch."""
    start = batch_index * batch_size
    return normalized_paths[start : start + actual]


def condition_images(
    cpu_batch: dict[str, Any], paths: list[str], condition: tuple[str, float, str] | None
) -> torch.Tensor:
    """Return clean or perturbed normalized images for one batch."""
    clean = cpu_batch["img"].float().div(255.0)
    if condition is None:
        return clean
    family, value, condition_id = condition
    return perturb_images(clean, paths, family, value, condition_id)


def cache_teacher(
    config: dict[str, Any], batches: list[dict[str, Any]], paths: list[str], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]], dict[str, Any]]:
    """Cache teacher embeddings and raw P4 batches once for all six Student checkpoints."""
    from ultralytics.nn.foundation import DINOv3Teacher

    teacher = DINOv3Teacher(
        model_id=str(config["teacher_model"]),
        weights_path=Path(config["teacher_weights"]),
        dtype=str(config["teacher_dtype"]),
        device=device,
        local_files_only=True,
    )
    conditions: list[tuple[str, float, str] | None] = [None, *normalized_condition_specs(config)]
    embeddings: dict[str, list[torch.Tensor]] = defaultdict(list)
    raw_features: dict[str, list[torch.Tensor]] = defaultdict(list)
    metadata: dict[str, Any] = {}
    with torch.inference_mode():
        for condition in conditions:
            key = "clean" if condition is None else condition[2]
            for batch_index, cpu_batch in enumerate(batches):
                local_paths = batch_paths(paths, batch_index, int(config["feature_batch_size"]), cpu_batch["img"].shape[0])
                images = condition_images(cpu_batch, local_paths, condition).to(device)
                response = teacher.encode(images)
                feature = response.dense["p4"]
                embeddings[key].append(spatial_relation_embedding(feature).cpu())
                raw_features[key].append(feature.detach().to("cpu", dtype=torch.bfloat16))
                metadata = dict(response.metadata)
            print(f"teacher cached: {key}", flush=True)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {key: torch.cat(value) for key, value in embeddings.items()}, dict(raw_features), metadata


def validate_checkpoint(checkpoint: dict[str, Any], arm: str, seed: int, config: dict[str, Any]) -> None:
    """Fail closed if an EMA checkpoint differs from the frozen P1 identity."""
    if int(checkpoint.get("epoch", -1)) != int(config["checkpoint_epoch"]):
        raise RuntimeError("checkpoint epoch mismatch")
    args = checkpoint.get("train_args") or {}
    locks = {
        "seed": (int(args.get("seed", -1)), seed),
        "imgsz": (int(args.get("imgsz", -1)), int(config["imgsz"])),
        "batch": (int(args.get("batch", -1)), 4),
        "epochs": (int(args.get("epochs", -1)), 50),
        "optimizer": (args.get("optimizer"), "SGD"),
    }
    if arm == "on64":
        foundation = checkpoint.get("foundation") or {}
        locks.update({
            "align_dim": (int(foundation.get("align_dim", -1)), 64),
            "teacher": (foundation.get("teacher"), "dinov3"),
            "loss": (foundation.get("loss"), "cosine"),
            "loss_weight": (float(foundation.get("loss_weight", -1)), 0.15),
            "target_levels": (foundation.get("target_levels"), ["p4"]),
        })
    elif checkpoint.get("foundation"):
        raise RuntimeError("OFF checkpoint unexpectedly contains Foundation configuration")
    mismatches = {name: {"actual": actual, "expected": expected} for name, (actual, expected) in locks.items() if actual != expected}
    if mismatches:
        raise RuntimeError(f"checkpoint lock mismatch: {json.dumps(mismatches)}")
    if checkpoint.get("ema") is None:
        raise RuntimeError("checkpoint has no EMA model")


def checkpoint_path(repo_root: Path, config: dict[str, Any], arm: str, seed: int) -> Path:
    """Resolve one of the six frozen P1 checkpoint paths."""
    path = (repo_root / str(config["checkpoint_patterns"][arm]).format(seed=seed)).resolve()
    if repo_root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return path


def projector_gap(projector: nn.Module, student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
    """Return the per-image P1 cosine objective-space gap for the ON64 audit only."""
    student_aligned, teacher_aligned = projector(student_feature, teacher_feature)
    student_tokens = student_aligned.float().flatten(2)
    teacher_tokens = teacher_aligned.float().flatten(2)
    similarity = F.cosine_similarity(student_tokens, teacher_tokens, dim=1, eps=1e-6)
    gap = 1.0 - similarity.mean(dim=1)
    if not torch.isfinite(gap).all():
        raise ValueError("non-finite projector objective-space gap")
    return gap


def evaluate_student(
    repo_root: Path,
    config: dict[str, Any],
    arm: str,
    seed: int,
    batches: list[dict[str, Any]],
    normalized_paths: list[str],
    teacher_embeddings: dict[str, torch.Tensor],
    teacher_raw: dict[str, list[torch.Tensor]],
    device: torch.device,
    metric_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate static/response gaps and degradation for one frozen Student checkpoint."""
    from ultralytics.nn.foundation import StudentFeatureTap

    path = checkpoint_path(repo_root, config, arm, seed)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, arm, seed, config)
    ema = checkpoint["ema"].float().to(device)
    student = ema.student_model if arm == "on64" else ema
    projector = ema.projector_for("p4").eval() if arm == "on64" else None
    student.eval()
    before_hash = state_dict_sha256(ema)
    tap = StudentFeatureTap(student, target="p4")
    conditions: list[tuple[str, float, str] | None] = [None, *normalized_condition_specs(config)]
    embeddings: dict[str, list[torch.Tensor]] = defaultdict(list)
    losses: dict[str, list[float]] = defaultdict(list)
    confidence: dict[str, list[float]] = defaultdict(list)
    objective_gap: dict[str, list[float]] = defaultdict(list)
    map_results: dict[str, dict[str, float]] = {}
    try:
        for condition in conditions:
            key = "clean" if condition is None else condition[2]
            validator = build_metric_validator(student, device, metric_root / f"{arm}-s{seed}-{key.replace(':', '_')}")
            for batch_index, cpu_batch in enumerate(batches):
                local_paths = batch_paths(
                    normalized_paths, batch_index, int(config["feature_batch_size"]), cpu_batch["img"].shape[0]
                )
                images = condition_images(cpu_batch, local_paths, condition)
                device_batch = batch_to_device(cpu_batch, images, device)
                student.eval()
                tap.clear()
                with torch.inference_mode():
                    predictions = student(device_batch["img"])
                    feature = tap.feature
                    embeddings[key].append(spatial_relation_embedding(feature).cpu())
                    processed = validator.postprocess(predictions)
                    confidence[key].extend(top10_confidence(processed))
                    validator.update_metrics(processed, device_batch)
                    if projector is not None:
                        teacher_feature = teacher_raw[key][batch_index].to(device=device, dtype=torch.float32)
                        objective_gap[key].extend(projector_gap(projector, feature, teacher_feature).cpu().tolist())
                losses[key].extend(image_detection_losses(student, device_batch))
            map_results[key] = {name: float(value) for name, value in validator.get_stats().items()}
            print(f"student {arm} seed {seed}: {key}", flush=True)
    finally:
        tap.close()
    student_embeddings = {key: torch.cat(value) for key, value in embeddings.items()}
    static = cosine_gap(student_embeddings["clean"], teacher_embeddings["clean"]).cpu().numpy()
    rows = []
    for condition in CONDITION_KEYS:
        response = cosine_gap(
            student_embeddings[condition] - student_embeddings["clean"],
            teacher_embeddings[condition] - teacher_embeddings["clean"],
            epsilon=float(config["response_norm_epsilon"]),
        ).cpu().numpy()
        for image_index, image in enumerate(normalized_paths):
            rows.append({
                "seed": seed,
                "arm": arm,
                "image_index": image_index,
                "image": image,
                "condition": condition,
                "static_gap": float(static[image_index]),
                "response_gap": float(response[image_index]),
                "detection_loss_clean": losses["clean"][image_index],
                "detection_loss_perturbed": losses[condition][image_index],
                "detection_loss_increase": losses[condition][image_index] - losses["clean"][image_index],
                "confidence_clean": confidence["clean"][image_index],
                "confidence_perturbed": confidence[condition][image_index],
                "confidence_drop": confidence["clean"][image_index] - confidence[condition][image_index],
                "projector_static_gap": objective_gap["clean"][image_index] if projector is not None else "",
            })
    after_hash = state_dict_sha256(ema)
    if before_hash != after_hash:
        raise RuntimeError(f"model state changed during diagnostic: {arm} seed {seed}")
    metadata = {
        "checkpoint": path.relative_to(repo_root).as_posix(),
        "checkpoint_sha256": sha256(path),
        "state_dict_sha256_before": before_hash,
        "state_dict_sha256_after": after_hash,
        "state_unchanged": True,
        "optimizer_steps": 0,
        "rows": len(rows),
        "diagnostic_metrics": map_results,
        "diagnostic_map50_95_drop": {
            condition: map_results["clean"]["metrics/mAP50-95(B)"]
            - map_results[condition]["metrics/mAP50-95(B)"]
            for condition in CONDITION_KEYS
        },
    }
    del checkpoint, ema, student, projector, embeddings, student_embeddings
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, metadata


def rows_to_arrays(rows: list[dict[str, Any]], seeds: list[int], image_count: int) -> tuple[np.ndarray, ...]:
    """Convert paired raw rows into the exact arrays used by H1/H2/H3."""
    seed_index = {seed: index for index, seed in enumerate(seeds)}
    arm_index = {"off": 0, "on64": 1}
    condition_index = {condition: index for index, condition in enumerate(CONDITION_KEYS)}
    static = np.full((len(seeds), 2, image_count), np.nan, dtype=np.float64)
    response = np.full((len(seeds), 2, len(CONDITION_KEYS), image_count), np.nan, dtype=np.float64)
    degradation = np.full_like(response, np.nan)
    for row in rows:
        s, a, c, i = (
            seed_index[int(row["seed"])],
            arm_index[str(row["arm"])],
            condition_index[str(row["condition"])],
            int(row["image_index"]),
        )
        static[s, a, i] = float(row["static_gap"])
        response[s, a, c, i] = float(row["response_gap"])
        degradation[s, a, c, i] = float(row["detection_loss_increase"])
    if not np.isfinite(static).all() or not np.isfinite(response).all() or not np.isfinite(degradation).all():
        raise RuntimeError("raw result matrix is incomplete or non-finite")
    return static, response, degradation


def formal_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Apply only the preregistered H1/H2/H3 estimands and stopping rule."""
    seeds = [int(value) for value in config["seeds"]]
    static, response, degradation = rows_to_arrays(rows, seeds, int(config["subset_size"]))
    statistics = config["statistics"]
    replicates = int(statistics["bootstrap_replicates"])
    bootstrap_seed = int(statistics["bootstrap_seed"])
    confidence = float(statistics["confidence_level"])
    static_delta = static[:, 1] - static[:, 0]
    h1_pooled = bootstrap_mean_by_image(static_delta, replicates, bootstrap_seed, confidence)
    h1_seeds = [
        bootstrap_mean_by_image(static_delta[index : index + 1], replicates, bootstrap_seed + 10 + index, confidence)
        for index in range(len(seeds))
    ]
    h1_positive = sum(result["ci95"][1] < 0 for result in h1_seeds)
    if h1_pooled["ci95"][1] < 0 and h1_positive >= 2:
        h1_status = "support"
    elif h1_pooled["ci95"][0] >= 0:
        h1_status = "not_supported"
    else:
        h1_status = "inconclusive"
    response_delta = (response[:, 1] - response[:, 0]).mean(axis=1)
    c_gap = response_delta - static_delta
    h2_pooled = bootstrap_mean_by_image(c_gap, replicates, bootstrap_seed + 100, confidence)
    h2_seeds = [
        bootstrap_mean_by_image(c_gap[index : index + 1], replicates, bootstrap_seed + 110 + index, confidence)
        for index in range(len(seeds))
    ]
    h2_positive = sum(result["ci95"][0] > 0 for result in h2_seeds)
    if h1_status != "support":
        h2_status = "not_evaluable"
    elif h2_pooled["ci95"][0] > 0 and h2_positive >= 2:
        h2_status = "support"
    elif h2_pooled["ci95"][1] <= 0:
        h2_status = "not_supported"
    else:
        h2_status = "inconclusive"
    h3 = bootstrap_h3(
        static, response, degradation, replicates, bootstrap_seed + 200, confidence
    )
    if h3["mean_z_response_ci95"][0] > 0 and h3["delta_rho_z_ci95"][0] > 0 and h3["positive_seed_arm_count"] >= 4:
        h3_status = "support"
    elif h3["delta_rho_z_ci95"][1] <= 0:
        h3_status = "not_supported"
    else:
        h3_status = "inconclusive"
    gate = h1_status == "support" and (h2_status == "support" or h3_status == "support")
    return {
        "h1": {"status": h1_status, "estimand": "mean(ON64_static_gap-OFF_static_gap)", "pooled": h1_pooled, "by_seed": h1_seeds},
        "h2": {"status": h2_status, "estimand": "mean((ON64_response-OFF_response)-(ON64_static-OFF_static))", "pooled": h2_pooled, "by_seed": h2_seeds},
        "h3": {"status": h3_status, "estimand": "48-cell equal-weight Fisher-z response association and advantage", **h3},
        "response_field_training_gate": {
            "passed": gate,
            "rule": "H1_support_and_at_least_one_of_H2_H3_support",
            "training_authorized_by_this_probe": False,
        },
    }


def git_and_protocol_audit(repo_root: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Record the current implementation commit and verify frozen source inputs."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    protected = [
        "experiments/rhino_d2/DINOV3_P1_GO_NO_GO.md",
        "experiments/rhino_d2/configs/d2_v3_p1_off.yaml",
        "experiments/rhino_d2/configs/d2_v3_p1_on.yaml",
        "ultralytics/nn/foundation",
        "ultralytics/nn/foundation_distill_model.py",
    ]
    difference = subprocess.run(
        ["git", "diff", "--name-only", str(config["source_commit"]), "--", *protected],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if difference:
        raise RuntimeError(f"protected P1 inputs differ from source_commit:\n{difference}")
    protocol = EXPERIMENT_ROOT / "DINOV3_P2_RESPONSE_GAP_PROTOCOL.md"
    diagnostic = repo_root / config["diagnostic_list"]
    return {
        "head": head,
        "source_commit": config["source_commit"],
        "protected_inputs_unchanged": True,
        "protocol_sha256": sha256(protocol),
        "config_sha256": sha256(config_path),
        "diagnostic_list_sha256": sha256(diagnostic),
        "runner_sha256": sha256(Path(__file__)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write raw paired observations with stable columns."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smoke_test() -> None:
    """Exercise only synthetic tensors; never load a diagnostic image or checkpoint."""
    images = torch.linspace(0, 1, 2 * 3 * 32 * 32, dtype=torch.float32).reshape(2, 3, 32, 32)
    paths = ["images/train2017/a.jpg", "images/train2017/b.jpg"]
    first = perturb_images(images, paths, "gaussian_noise", 0.03, "gaussian_noise:0.03")
    second = perturb_images(images, paths, "gaussian_noise", 0.03, "gaussian_noise:0.03")
    if not torch.equal(first, second):
        raise AssertionError("Gaussian perturbation is not bitwise deterministic")
    embedding = spatial_relation_embedding(torch.randn(2, 7, 9, 11))
    if embedding.shape != (2, 32640) or embedding.dtype != torch.float32:
        raise AssertionError(f"unexpected embedding contract: {embedding.shape}, {embedding.dtype}")
    try:
        cosine_gap(torch.zeros(2, 8), torch.ones(2, 8))
    except ValueError:
        pass
    else:
        raise AssertionError("zero response did not fail closed")
    rho = spearman(np.array([1, 2, 2, 4], dtype=float), np.array([4, 2, 2, 1], dtype=float))
    if not math.isfinite(rho) or abs(rho) > 1.0 + 1e-12:
        raise AssertionError("synthetic Spearman check failed")
    print(json.dumps({"status": "synthetic_smoke_passed", "formal_metrics": False}, indent=2))


def run(config_path: Path, output_dir: Path, device: torch.device) -> dict[str, Any]:
    """Run the official probe after all preregistration and implementation checks pass."""
    config = load_yaml(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime(output_dir)
    audit = git_and_protocol_audit(REPO_ROOT, config_path, config)
    batches, normalized_paths, dataset_metadata = build_batches(REPO_ROOT, config)
    teacher_embeddings, teacher_raw, teacher_metadata = cache_teacher(config, batches, normalized_paths, device)
    rows, checkpoints = [], []
    for seed in config["seeds"]:
        for arm in config["arms"]:
            local_rows, metadata = evaluate_student(
                REPO_ROOT,
                config,
                str(arm),
                int(seed),
                batches,
                normalized_paths,
                teacher_embeddings,
                teacher_raw,
                device,
                output_dir / "runtime/metrics",
            )
            rows.extend(local_rows)
            checkpoints.append({"seed": int(seed), "arm": str(arm), **metadata})
    expected_rows = len(config["seeds"]) * len(config["arms"]) * int(config["subset_size"]) * len(CONDITION_KEYS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} paired rows, found {len(rows)}")
    raw_path = output_dir / "d2_v3_p2_response_gap_raw.csv"
    write_csv(raw_path, rows)
    decisions = formal_summary(rows, config)
    payload = {
        "schema_version": 1,
        "status": "completed_p2_03_response_gap_probe",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "boundaries": {
            "diagnostic_only": True,
            "new_training_runs": 0,
            "optimizer_steps": 0,
            "p1_no_go_unchanged": True,
            "p2_01_inconclusive_unchanged": True,
            "p2_02_no_support_unchanged": True,
            "correlation_is_not_causation": True,
            "perturbation_breakdowns_secondary_only": True,
        },
        "audit": audit,
        "dataset": dataset_metadata,
        "teacher_metadata": teacher_metadata,
        "checkpoints": checkpoints,
        "decisions": decisions,
        "raw": {"path": raw_path.relative_to(REPO_ROOT).as_posix(), "rows": len(rows), "sha256": sha256(raw_path)},
    }
    summary_path = output_dir / "d2_v3_p2_response_gap_result.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = output_dir / "d2_v3_p2_response_gap_manifest.json"
    manifest = {
        "summary": {"path": summary_path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(summary_path)},
        "raw": payload["raw"],
        "protocol": audit,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["decisions"], indent=2, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    """Parse the diagnostic-only command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="run synthetic-only checks and exit")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
        return
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for the formal P2-03 probe")
    run(args.config.resolve(), args.output_dir.resolve(), torch.device(args.device))


if __name__ == "__main__":
    main()
