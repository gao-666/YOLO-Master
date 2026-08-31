#!/usr/bin/env python3
"""Fail-closed DINOv3 local-asset dtype and finite-feature smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from ultralytics.nn.foundation import DINOv3Teacher

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
MODEL_CACHE_ROOT = REPO_ROOT.parent / "model_cache"


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> dict:
    """Load one DINOv3 P0 config from the experiment directory."""
    path = path.resolve()
    if EXPERIMENT_ROOT not in path.parents:
        raise ValueError("--config must stay inside the D2 experiment directory")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("DINOv3 P0 config must be a YAML mapping")
    return value


def resolve_teacher_dir(value: str) -> Path:
    """Resolve a mentor-provided model directory through a strict cache allowlist."""
    path = Path(value).resolve()
    if MODEL_CACHE_ROOT.resolve() not in path.parents or not path.is_dir():
        raise ValueError(f"teacher_weights must be an existing directory under {MODEL_CACHE_ROOT}")
    for required in ("config.json", "model.safetensors", "LICENSE.md", "README.md"):
        if not (path / required).is_file():
            raise FileNotFoundError(f"missing DINOv3 asset: {path / required}")
    return path


def load_image(path: Path, imgsz: int, device: torch.device, batch_size: int) -> torch.Tensor:
    """Load one repository image as a deterministic repeated batch in [0, 1]."""
    path = path.resolve()
    if REPO_ROOT not in path.parents or not path.is_file():
        raise ValueError("image must be an existing file inside the repository")
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = F.interpolate(tensor, size=(imgsz, imgsz), mode="bilinear", align_corners=False)
    return tensor.repeat(batch_size, 1, 1, 1)


def main() -> None:
    """Run one offline teacher forward and persist dtype, geometry, and finite checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    teacher_dir = resolve_teacher_dir(str(config["teacher_weights"]))
    image_path = (REPO_ROOT / str(config["image"])).resolve()
    images = load_image(image_path, int(config["imgsz"]), device, int(config["batch_size"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    teacher = DINOv3Teacher(
        model_id=str(config["teacher_model"]),
        weights_path=teacher_dir,
        dtype=str(config["teacher_dtype"]),
        device=device,
        local_files_only=True,
    )
    features = teacher.encode(images)
    dense = features.dense["p4"]
    checks = {
        "dense_finite": bool(torch.isfinite(dense).all()),
        "pooled_finite": bool(features.pooled is not None and torch.isfinite(features.pooled).all()),
        "hidden_size_matches": dense.shape[1] == int(config["expected_hidden_size"]),
        "patch_size_matches": teacher.patch_size == int(config["expected_patch_size"]),
        "grid_matches": tuple(dense.shape[-2:])
        == (int(config["imgsz"]) // teacher.patch_size, int(config["imgsz"]) // teacher.patch_size),
        "dtype_matches": dense.dtype == teacher.dtype,
        "teacher_frozen": all(
            not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters()
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "claim": "teacher_dtype_and_finite_gate_only_no_accuracy_claim",
        "config": {"path": str(config_path.relative_to(REPO_ROOT)), "sha256": sha256(config_path)},
        "teacher": {
            "model_id": config["teacher_model"],
            "weights_path": teacher_dir.as_posix(),
            "weights_sha256": sha256(teacher_dir / "model.safetensors"),
            "license": config["teacher_license"],
            "license_sha256": sha256(teacher_dir / "LICENSE.md"),
            "requested_dtype": config["teacher_dtype"],
            "actual_dtype": str(teacher.dtype),
            "feature_shape": list(dense.shape),
            "metadata": features.metadata,
        },
        "checks": checks,
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
        },
    }
    output = EXPERIMENT_ROOT / "results" / str(config["teacher_smoke_output_file"])
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
