#!/usr/bin/env python3
"""Run fail-closed YOLO task loss plus local DINOv3 P4 distillation P0 smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.foundation import DINOv3Teacher
from ultralytics.nn.foundation_distill_model import FoundationDistillationModel

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
MODEL_CACHE_ROOT = REPO_ROOT.parent / "model_cache"


def sha256(path: Path) -> str:
    """Return one file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict:
    """Record the source commit and dirty-state digest without embedding file names."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(porcelain.strip()),
        "porcelain_sha256": hashlib.sha256(porcelain.encode()).hexdigest(),
    }


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate one DINOv3 P0 YAML mapping."""
    path = path.resolve()
    if EXPERIMENT_ROOT not in path.parents:
        raise ValueError("--config must stay inside the D2 experiment directory")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("DINOv3 P0 config must be a YAML mapping")
    return value


def resolve_repo_file(value: str) -> Path:
    """Resolve a repository-relative input through a strict allowlist."""
    path = (REPO_ROOT / value).resolve()
    if REPO_ROOT not in path.parents or not path.is_file():
        raise ValueError(f"repository input is missing or outside the repository: {value}")
    return path


def resolve_teacher_dir(value: str) -> Path:
    """Resolve an existing local Teacher directory under the shared model cache."""
    path = Path(value).resolve()
    if MODEL_CACHE_ROOT.resolve() not in path.parents or not path.is_dir():
        raise ValueError(f"teacher_weights must be an existing directory under {MODEL_CACHE_ROOT}")
    for required in ("config.json", "preprocessor_config.json", "model.safetensors", "LICENSE.md", "README.md"):
        if not (path / required).is_file():
            raise FileNotFoundError(f"missing DINOv3 asset: {path / required}")
    return path


def teacher_assets(directory: Path) -> list[dict[str, Any]]:
    """Inventory only the version-sensitive model, preprocessing, card, and license files."""
    names = ("config.json", "preprocessor_config.json", "model.safetensors", "LICENSE.md", "README.md")
    return [
        {"name": name, "bytes": (directory / name).stat().st_size, "sha256": sha256(directory / name)} for name in names
    ]


def load_image(path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    """Load one RGB image as a normalized BCHW tensor."""
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return F.interpolate(tensor, size=(imgsz, imgsz), mode="bilinear", align_corners=False)


def build_batch(
    image: torch.Tensor, batch_size: int, *, target_class: int, target_bbox_xywh: list[float]
) -> dict[str, torch.Tensor]:
    """Build a deterministic synthetic detection target for gradient-chain verification only."""
    if len(target_bbox_xywh) != 4:
        raise ValueError("target_bbox_xywh must contain four normalized xywh values")
    return {
        "img": image.repeat(batch_size, 1, 1, 1),
        "cls": torch.full((batch_size, 1), float(target_class), device=image.device),
        "bboxes": torch.tensor([target_bbox_xywh], device=image.device).repeat(batch_size, 1),
        "batch_idx": torch.arange(batch_size, device=image.device, dtype=torch.float32),
    }


def grad_norm(parameters) -> float:
    """Return the L2 norm of finite gradients, failing closed on non-finite values."""
    squares = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not torch.isfinite(gradient).all():
            raise ValueError("P0 gradient contains NaN or Inf")
        squares.append(gradient.square().sum())
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def wrapper_config(config: dict[str, Any]) -> SimpleNamespace:
    """Map the isolated P0 config to the production Foundation wrapper contract."""
    return SimpleNamespace(
        imgsz=int(config["imgsz"]),
        foundation_enabled=True,
        foundation_teacher="dinov3",
        foundation_target_levels=[str(config["student_level"])],
        foundation_multiscale=False,
        foundation_align_dim=int(config["align_dim"]),
        foundation_loss=str(config["foundation_loss"]),
        foundation_loss_weight=float(config["foundation_loss_weight"]),
        foundation_cosine_weight=1.0,
        foundation_relation_weight=0.0,
        foundation_foreground_weighting=False,
        foundation_router_distill=False,
        foundation_router_loss_weight=0.0,
        foundation_semantic_distill=False,
        foundation_semantic_loss_weight=0.0,
        foundation_multitask=False,
        foundation_weight_schedule="constant",
    )


def main() -> None:
    """Execute the DINOv3 P0 gate and write complete machine-readable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if bool(config.get("deterministic", False)):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    teacher_dir = resolve_teacher_dir(str(config["teacher_weights"]))
    teacher = DINOv3Teacher(
        model_id=str(config["teacher_model"]),
        weights_path=teacher_dir,
        dtype=str(config["teacher_dtype"]),
        device=device,
        local_files_only=True,
    )
    student_path = resolve_repo_file(str(config["student_model"]))
    image_path = resolve_repo_file(str(config["image"]))
    image = load_image(image_path, int(config["imgsz"]), device)
    teacher_probe = teacher.encode(image)
    teacher_probe_feature = teacher_probe.dense["p4"]
    if not torch.isfinite(teacher_probe_feature).all():
        raise ValueError("DINOv3 probe feature contains NaN or Inf")

    student = YOLO(str(student_path)).model.to(device).train()
    student.args = get_cfg(overrides={"imgsz": int(config["imgsz"])})
    wrapper = FoundationDistillationModel(student, teacher, wrapper_config(config)).to(device).train()
    trainable = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    teacher_parameter_ids = {id(parameter) for parameter in teacher.parameters()}
    optimizer_parameter_ids = {id(parameter) for parameter in trainable}
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]), weight_decay=0.0)
    batch = build_batch(
        image,
        int(config["batch_size"]),
        target_class=int(config["target_class"]),
        target_bbox_xywh=[float(value) for value in config["target_bbox_xywh"]],
    )

    records = []
    for step in range(int(config["steps"])):
        optimizer.zero_grad(set_to_none=True)
        loss_components, loss_items = wrapper(batch)
        total_loss = loss_components.sum()
        if not torch.isfinite(total_loss):
            raise ValueError("P0 total loss contains NaN or Inf")
        total_loss.backward()
        records.append(
            {
                "step": step,
                "total_loss": float(total_loss.detach()),
                "task_loss": float(loss_components[:-1].detach().sum()),
                "foundation_loss": float(loss_components[-1].detach()),
                "loss_items": [float(value) for value in loss_items.detach().reshape(-1)],
                "student_grad_norm": grad_norm(wrapper.student_model.parameters()),
                "projector_grad_norm": grad_norm(wrapper.projector.student_proj.parameters()),
                "foundation_metrics": wrapper.foundation_metrics(),
            }
        )
        optimizer.step()

    student_feature = wrapper.tap.feature.detach()
    teacher_batch = teacher.encode(batch["img"]).dense["p4"].detach()
    with torch.no_grad():
        student_aligned, teacher_aligned = wrapper.projector(student_feature, teacher_batch)
    teacher_frozen = all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
    checks = {
        "teacher_output_finite": bool(torch.isfinite(teacher_batch).all()),
        "student_p4_finite": bool(torch.isfinite(student_feature).all()),
        "student_shape_matches": list(student_feature.shape)
        == [int(config["batch_size"]), 128, int(config["imgsz"]) // 16, int(config["imgsz"]) // 16],
        "teacher_shape_matches": list(teacher_batch.shape)
        == [
            int(config["batch_size"]),
            int(config["expected_hidden_size"]),
            int(config["imgsz"]) // int(config["expected_patch_size"]),
            int(config["imgsz"]) // int(config["expected_patch_size"]),
        ],
        "aligned_shapes_equal": student_aligned.shape == teacher_aligned.shape,
        "aligned_features_finite": bool(
            torch.isfinite(student_aligned).all() and torch.isfinite(teacher_aligned).all()
        ),
        "teacher_frozen": teacher_frozen,
        "teacher_not_in_optimizer": not bool(teacher_parameter_ids & optimizer_parameter_ids),
        "student_has_gradient": all(record["student_grad_norm"] > 0 for record in records),
        "projector_has_gradient": all(record["projector_grad_norm"] > 0 for record in records),
        "finite_losses": all(
            np.isfinite(record["total_loss"])
            and np.isfinite(record["task_loss"])
            and np.isfinite(record["foundation_loss"])
            for record in records
        ),
        "foundation_loss_nonzero": all(record["foundation_loss"] > 0 for record in records),
        "foundation_in_total_loss": all(
            abs(record["total_loss"] - record["task_loss"] - record["foundation_loss"]) < 1e-4 for record in records
        ),
        "fixed_batch_foundation_loss_decreased": records[-1]["foundation_loss"] < records[0]["foundation_loss"],
    }
    observations = {
        # Detection assignment, BatchNorm, and routing can make total task loss fluctuate over five fixed-batch steps.
        # Preserve this observation without silently turning it into a gate that the frozen protocol did not require.
        "fixed_batch_total_loss_saw_decrease": min(record["total_loss"] for record in records[1:])
        < records[0]["total_loss"],
    }
    payload = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "claim": "p0_real_yolo_loss_synthetic_target_fixed_batch_smoke_no_accuracy_claim",
        "experiment_id": config["experiment_id"],
        "source_state": git_state(),
        "script_sha256": sha256(Path(__file__)),
        "config": {"path": str(config_path.relative_to(REPO_ROOT)), "sha256": sha256(config_path), **config},
        "student": {
            "model": str(student_path.relative_to(REPO_ROOT)),
            "model_yaml_sha256": sha256(student_path),
            "level": config["student_level"],
            "source_indices": list(wrapper.tap.source_indices),
            "feature_shape": list(student_feature.shape),
        },
        "teacher": {
            "model_id": config["teacher_model"],
            "source": config["teacher_source"],
            "weights_path": teacher_dir.as_posix(),
            "license": config["teacher_license"],
            "dtype": str(teacher.dtype),
            "feature_shape": list(teacher_batch.shape),
            "metadata": teacher_probe.metadata,
            "assets": teacher_assets(teacher_dir),
        },
        "alignment": {
            "student_aligned_shape": list(student_aligned.shape),
            "teacher_aligned_shape": list(teacher_aligned.shape),
            "metadata": wrapper.projector.alignment,
            "teacher_projection_frozen": wrapper.projector.teacher_projection_frozen,
        },
        "data_contract": {
            "image": str(image_path.relative_to(REPO_ROOT)),
            "image_repeated": int(config["batch_size"]),
            "target_source": config["target_source"],
            "target_class": int(config["target_class"]),
            "target_bbox_format": "normalized_xywh",
            "target_bbox_xywh": [float(value) for value in config["target_bbox_xywh"]],
            "purpose": "gradient_chain_smoke_not_accuracy_evaluation",
        },
        "checks": checks,
        "non_blocking_observations": observations,
        "steps": records,
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "seed": seed,
        },
    }
    output = EXPERIMENT_ROOT / "results" / str(config["output_file"])
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
