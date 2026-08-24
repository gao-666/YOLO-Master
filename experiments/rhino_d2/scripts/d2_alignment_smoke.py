#!/usr/bin/env python3
"""Run a real YOLO-Master P4 to public DINOv2 single-batch alignment smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn

# 将实验产物与模型缓存固定在仓库内，便于复现实验并避免污染用户级配置。
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
RUN_CONFIG_ROOT = REPO_ROOT / "runs" / "rhino_d2" / "config"
RUN_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUN_CONFIG_ROOT))
os.environ.setdefault("HF_HOME", str(EXPERIMENT_ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(EXPERIMENT_ROOT / "cache" / "huggingface" / "hub"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import snapshot_download
from transformers import Dinov2Model

from ultralytics import YOLO
from ultralytics.nn.foundation import (
    FoundationFeatures,
    P4AlignmentProjector,
    StudentFeatureTap,
    cosine_kd_loss,
)
from ultralytics.nn.foundation.preprocessing import (
    DINOV3_IMAGE_MEAN,
    DINOV3_IMAGE_STD,
    prepare_image_tensor,
)


class DINOv2Teacher(nn.Module):
    """将本地 DINOv2 适配为 YOLO-Master 的基础模型教师接口。

    该适配器把 DINOv2 的 token 输出转换为统一的 ``FoundationFeatures``，
    让学生模型可以读取教师的密集 P4 特征。教师参数和推理模式始终冻结，
    避免教师与学生同时变化。它不会直接提高 mAP，但为后续蒸馏提供稳定、
    可复现的监督目标，并检查特征形状与冻结状态是否满足实验契约。
    """

    name = "dinov2"

    def __init__(self, snapshot: Path, device: torch.device, model_id: str) -> None:
        super().__init__()
        # 教师只负责提供目标特征，整个模型在 smoke 过程中保持冻结状态。
        self.snapshot = snapshot
        self.model_id = model_id
        self.model = Dinov2Model.from_pretrained(snapshot, local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.patch_size = int(self.model.config.patch_size)
        self.hidden_size = int(self.model.config.hidden_size)
        self.device = device

    def freeze(self) -> None:
        """固定教师的评估行为并关闭所有参数梯度。

        评估模式固定 dropout 和 batch normalization 等行为，关闭梯度则
        防止教师被优化器更新。稳定的教师目标能让 loss 变化归因于学生
        学习，而不是教师漂移；这是比较蒸馏前后检测指标的基础。
        """
        self.model.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        """覆盖默认的 ``train`` 行为，防止外层封装意外解冻教师。

        蒸馏封装器可能递归调用子模块的 ``train`` 方法，因此这里强制教师
        保持 eval 状态，保证 smoke 测试和正式训练拥有一致的冻结语义。
        """
        super().train(False)
        self.freeze()
        return self

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """按教师的 patch 约束预处理图像。

        学生和教师必须使用同一输入，但输入尺寸、patch 倍数和归一化规范
        可能不同。本函数统一这些条件，避免预处理差异制造虚假的蒸馏损失，
        使 loss 下降更能反映学生对教师表征的学习。
        """
        return prepare_image_tensor(
            images.to(self.device),
            patch_size=self.patch_size,
            mean=DINOV3_IMAGE_MEAN,
            std=DINOV3_IMAGE_STD,
        )

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> FoundationFeatures:
        """提取 DINOv2 的密集 patch 特征和全局 pooled 特征。

        DINOv2 输出序列 token，而检测蒸馏需要保留空间布局的 NCHW 特征图。
        本函数去除 CLS 等前缀 token，并将 patch token 重排为 ``p4`` 特征，
        为学生 backbone 提供逐位置监督。空间监督通常比单个全局向量更适合
        目标定位；本 smoke 只验证接口、形状和数值链路，不代表真实精度提升。
        """
        pixel_values = self.preprocess(images)
        output = self.model(pixel_values=pixel_values)
        tokens = output.last_hidden_state
        # 去掉 CLS 等前缀 token，仅将 patch token 还原为空间特征图。
        grid_h = pixel_values.shape[-2] // self.patch_size
        grid_w = pixel_values.shape[-1] // self.patch_size
        prefix = tokens.shape[1] - grid_h * grid_w
        if prefix < 0:
            raise ValueError("DINOv2 token count is smaller than the expected patch grid")
        patches = tokens[:, prefix:, :]
        feature = patches.reshape(tokens.shape[0], grid_h, grid_w, tokens.shape[-1]).permute(0, 3, 1, 2)
        return FoundationFeatures(
            dense={"p4": feature.contiguous()},
            pooled=tokens[:, 0, :],
            metadata={
                "model_id": self.model_id,
                "patch_size": self.patch_size,
                "prefix_tokens": prefix,
                "grid_size": [grid_h, grid_w],
            },
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    """读取并验证 smoke 配置，确保实验参数来自明确的 YAML mapping。

    配置驱动可以固定教师版本、学生层级、输入尺寸和优化参数，减少命令行
    临时参数导致的实验漂移。提前检查根节点类型，可在模型启动前暴露配置
    错误，避免无效运行污染后续指标分析。
    """
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("smoke config must contain a YAML mapping")
    return value


def _sha256(path: Path) -> str:
    """计算文件 SHA-256，记录实验输入的精确身份。

    哈希不参与训练，也不会直接改善精度；它能证明结果使用了哪一份配置、
    模型 YAML 或教师权重，防止不同版本的结果被错误比较。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inside_repo(value: str) -> Path:
    """解析仓库内的实验文件并拒绝越界或缺失路径。

    统一输入来源可确保 smoke 使用仓库中已审计的模型和图片，而不是用户
    目录中的同名文件。可复现的输入是判断蒸馏是否真正改善检测指标的前提。
    """
    path = (REPO_ROOT / value).resolve()
    if REPO_ROOT not in path.parents:
        raise ValueError(f"path must stay inside repository: {value}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_image(path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    """加载固定 RGB 图片并转换为模型输入张量。

    单张固定图片让 smoke 输入完全可控，统一方形尺寸则保证学生和教师的
    空间特征可以对齐。它不代表真实数据集精度，但能低成本排除通道顺序、
    数值范围和特征尺寸错误，避免工程问题污染正式指标。
    """
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return F.interpolate(tensor, size=(imgsz, imgsz), mode="bilinear", align_corners=False)


def optimize_alignment(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    *,
    align_dim: int,
    steps: int,
    learning_rate: float,
) -> tuple[P4AlignmentProjector, list[float]]:
    """在固定 batch 上只优化学生投影层，并返回完整的损失轨迹。

    学生 backbone 和教师特征都保持不更新，只有学生侧 projector 接收梯度。
    如果 cosine KD loss 能下降，说明特征抽取、通道映射、损失计算和反向
    传播构成了有效链路；否则应先修复链路，再讨论蒸馏是否提升 mAP、召回率
    或小目标指标。由于这里只使用一张图片和少量 step，结果不代表泛化精度。
    """
    projector = P4AlignmentProjector(
        student_channels=int(student_feature.shape[1]),
        teacher_channels=int(teacher_feature.shape[1]),
        align_dim=align_dim,
        use_norm=False,
    ).to(student_feature.device)
    # 只更新学生侧投影层，验证冻结教师特征后对齐损失仍能下降。
    optimizer = torch.optim.AdamW(projector.student_proj.parameters(), lr=learning_rate, weight_decay=0.0)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        student_aligned, teacher_aligned = projector(student_feature, teacher_feature)
        loss = cosine_kd_loss(student_aligned, teacher_aligned)
        if not torch.isfinite(loss):
            raise ValueError("non-finite D2 smoke loss")
        history.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        student_aligned, teacher_aligned = projector(student_feature, teacher_feature)
        history.append(float(cosine_kd_loss(student_aligned, teacher_aligned)))
    return projector, history


def _teacher_assets(snapshot: Path) -> list[dict[str, Any]]:
    """记录教师快照关键权重文件的大小和哈希。

    结果只保存资产身份，不复制整个模型。这样既保持结果轻量，又能确认
    不同运行使用相同教师权重，减少教师版本差异造成的指标波动。
    """
    assets = []
    for path in sorted(snapshot.glob("*")):
        if path.is_file() and path.name in {"config.json", "model.safetensors", "pytorch_model.bin"}:
            assets.append({"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)})
    return assets


def parse_args() -> argparse.Namespace:
    """解析对齐 smoke 的配置路径和离线运行开关。

    ``--offline`` 强制使用已缓存的教师快照，避免网络下载到不同 revision，
    从而保证不同运行的 loss 和后续检测指标具备可比性。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs" / "d2_smoke.yaml")
    parser.add_argument("--offline", action="store_true", help="Require an already cached teacher snapshot")
    return parser.parse_args()


def main() -> None:
    """执行 DINOv2 P4 对齐 smoke 并写出结构化结果。

    流程依次固定配置与随机种子、加载教师快照、截取学生特征、执行单批次
    投影优化并归档结果。``passed`` 仅表示对齐损失下降，证明 D2 蒸馏管线
    可以工作，不等同于检测精度提升；完整训练和验证集评估仍是指标结论的
    必要步骤。结果中的形状、冻结状态、版本哈希和 loss history 用于审计。
    """
    args = parse_args()
    config_path = args.config.resolve()
    if EXPERIMENT_ROOT not in config_path.parents:
        raise ValueError("--config must stay inside the D2 experiment directory")
    config = _load_yaml(config_path)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(str(config["device"]))
    # 提前失败，避免在 CPU 上误跑本应使用 CUDA 的实验配置。
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    snapshot = Path(
        snapshot_download(
            repo_id=str(config["teacher_model"]),
            revision=str(config["teacher_revision"]),
            allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin"],
            local_files_only=args.offline,
        )
    ).resolve()
    teacher = DINOv2Teacher(snapshot, device, str(config["teacher_model"]))
    student_path = _resolve_inside_repo(str(config["student_model"]))
    image_path = _resolve_inside_repo(str(config["image"]))
    image = _load_image(image_path, int(config["imgsz"]), device)
    student = YOLO(str(student_path)).model.to(device).eval()
    tap = StudentFeatureTap(student, str(config["student_level"]))
    try:
        # 前向一次学生和教师，截取同一输入对应的 P4 特征进行对齐。
        with torch.no_grad():
            student(image)
            student_feature = tap.feature.detach()
            teacher_output = teacher.encode(image)
            teacher_feature = teacher_output.dense["p4"].detach()
    finally:
        tap.close()

    projector, history = optimize_alignment(
        student_feature,
        teacher_feature,
        align_dim=int(config["align_dim"]),
        steps=int(config["steps"]),
        learning_rate=float(config["learning_rate"]),
    )
    with torch.no_grad():
        student_aligned, teacher_aligned = projector(student_feature, teacher_feature)
    decreased = history[-1] < history[0]
    snapshot_revision = snapshot.name
    payload = {
        "schema_version": 1,
        "status": "passed" if decreased else "failed",
        "claim": "pipeline_smoke_only_no_accuracy_claim",
        "experiment_id": config["experiment_id"],
        "config": {"path": str(config_path.relative_to(REPO_ROOT)), "sha256": _sha256(config_path), **config},
        "student": {
            "model": str(student_path.relative_to(REPO_ROOT)),
            "model_yaml_sha256": _sha256(student_path),
            "level": config["student_level"],
            "source_indices": list(tap.source_indices),
            "feature_shape": list(student_feature.shape),
        },
        "teacher": {
            "model": config["teacher_model"],
            "requested_revision": config["teacher_revision"],
            "resolved_revision": snapshot_revision,
            "license": config["teacher_license"],
            "feature_shape": list(teacher_feature.shape),
            "metadata": teacher_output.metadata,
            "assets": _teacher_assets(snapshot),
            "frozen": all(not parameter.requires_grad for parameter in teacher.parameters()),
        },
        "alignment": {
            "student_aligned_shape": list(student_aligned.shape),
            "teacher_aligned_shape": list(teacher_aligned.shape),
            "metadata": projector.alignment,
            "teacher_projection_frozen": projector.teacher_projection_frozen,
        },
        "optimization": {
            "loss": config["loss"],
            "steps": config["steps"],
            "learning_rate": config["learning_rate"],
            "history": history,
            "initial_loss": history[0],
            "final_loss": history[-1],
            "relative_drop": (history[0] - history[-1]) / max(abs(history[0]), 1e-12),
            "decreased": decreased,
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "seed": seed,
        },
    }
    output = EXPERIMENT_ROOT / "results" / "d2_alignment_smoke.json"
    # JSON 同时保存配置摘要、模型信息和损失曲线，作为可审计的 smoke 结果。
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not decreased:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
