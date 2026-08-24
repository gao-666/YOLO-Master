#!/usr/bin/env python3
"""Run actual YOLO task loss plus DINOv2 P4 distillation for a few optimizer steps."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from d2_alignment_smoke import (
    EXPERIMENT_ROOT,
    REPO_ROOT,
    DINOv2Teacher,
    _load_image,
    _load_yaml,
    _resolve_inside_repo,
    _sha256,
    _teacher_assets,
)
from huggingface_hub import snapshot_download

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.foundation_distill_model import FoundationDistillationModel


def _batch(image: torch.Tensor, batch_size: int) -> dict[str, torch.Tensor]:
    """构造确定性的检测 batch，并为每张图片放置一个归一化框。

    该 batch 不用于衡量真实数据集精度，而是为检测损失提供最小、稳定的
    监督信号。固定图片、类别和边界框可以快速验证任务损失与 foundation
    蒸馏损失能否同时计算并反向传播，避免正式训练数小时后才发现标签格式
    或 batch 组装错误。它不直接提升 mAP，但能提高实验迭代效率。
    """
    images = image.repeat(batch_size, 1, 1, 1)
    return {
        "img": images,
        "cls": torch.zeros(batch_size, 1, device=image.device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.35, 0.55]], device=image.device).repeat(batch_size, 1),
        "batch_idx": torch.arange(batch_size, device=image.device, dtype=torch.float32),
    }


def _grad_norm(parameters) -> float:
    """计算参数集合中有限梯度的整体 L2 范数。

    学生梯度和 projector 梯度是 P0 smoke 的关键健康指标：非零梯度说明
    loss 确实连接到了可训练模块，有限值说明没有 NaN/Inf。该检查不能证明
    模型一定获得更高检测指标，但能区分“蒸馏方法无效”和“计算图未连通”。
    """
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            if torch.isfinite(gradient).all():
                squares.append(gradient.square().sum())
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def _wrapper_config(config: dict[str, Any]) -> SimpleNamespace:
    """把精简实验 YAML 映射为蒸馏封装器所需的配置契约。

    P0 只关注单尺度 P4 cosine 蒸馏，因此显式关闭关系蒸馏、前景加权、路由
    蒸馏和多任务分支。完整填充字段可以模拟正式 Trainer 的配置对象，避免
    测试因缺省字段而产生假阳性；同时便于确认 foundation loss 真正进入
    总损失，为后续比较 mAP、召回率等指标建立可信训练基础。
    """
    return SimpleNamespace(
        imgsz=int(config["imgsz"]),
        foundation_enabled=True,
        foundation_teacher="dinov2",
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


def parse_args() -> argparse.Namespace:
    """解析 P0 训练 smoke 的配置路径和离线教师选项。

    配置参数支持复用不同消融设置，离线选项则固定教师版本，避免网络下载
    结果变化给跨运行的损失和指标比较引入额外变量。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs" / "d2_p0.yaml")
    parser.add_argument("--offline", action="store_true", help="Require the locked local DINOv2 snapshot")
    return parser.parse_args()


def main() -> None:
    """执行少量真实任务损失加 DINOv2 蒸馏的 P0 smoke。

    流程覆盖学生前向、检测任务损失、foundation loss、总损失反向传播、
    优化器更新和教师冻结检查，比单独的特征对齐测试更接近正式训练。输出
    记录每一步的损失、梯度、教师状态和运行环境，帮助确认蒸馏信号确实
    影响学生参数。

    ``passed`` 仅表示训练链路满足有限性、梯度连通性和固定 batch 改善等
    工程条件，不代表 mAP 一定提升。真实指标仍需完整训练、独立验证集和
    D2 on/off 配对实验确认。
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
    # 配置声明 CUDA 时必须确保当前环境真的可用。
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
    student = YOLO(str(student_path)).model.to(device).train()
    # DetectionModel constructed directly from YAML keeps ``args`` as a plain dict. The formal Trainer replaces it
    # with the validated default namespace before the criterion is built; mirror that one required integration step.
    student.args = get_cfg(overrides={"imgsz": int(config["imgsz"])})
    wrapper = FoundationDistillationModel(student, teacher, _wrapper_config(config)).to(device).train()

    trainable = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    teacher_parameter_ids = {id(parameter) for parameter in teacher.parameters()}
    optimizer_parameter_ids = {id(parameter) for parameter in trainable}
    teacher_not_in_optimizer = not bool(teacher_parameter_ids & optimizer_parameter_ids)
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]), weight_decay=0.0)
    batch = _batch(image, int(config["batch_size"]))

    # 每一步都记录任务项、蒸馏项、梯度和指标，便于定位集成问题。
    records = []
    for step in range(int(config["steps"])):
        optimizer.zero_grad(set_to_none=True)
        loss_components, loss_items = wrapper(batch)
        total_loss = loss_components.sum()
        if not torch.isfinite(total_loss):
            raise ValueError("P0 total loss is NaN or Inf")
        total_loss.backward()
        student_grad = _grad_norm(wrapper.student_model.parameters())
        projector_grad = _grad_norm(wrapper.projector.student_proj.parameters())
        metrics = wrapper.foundation_metrics()
        records.append(
            {
                "step": step,
                "total_loss": float(total_loss.detach()),
                "task_loss": float(loss_components[:-1].detach().sum()),
                "foundation_loss": float(loss_components[-1].detach()),
                "loss_items": [float(value) for value in loss_items.detach().reshape(-1)],
                "student_grad_norm": student_grad,
                "projector_grad_norm": projector_grad,
                "foundation_metrics": metrics,
            }
        )
        optimizer.step()

    teacher_frozen = all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
    # 总损失允许非单调，但所有关键组成部分和梯度约束必须满足。
    checks = {
        "finite_losses": all(np.isfinite(record["total_loss"]) for record in records),
        "foundation_loss_nonzero": all(record["foundation_loss"] > 0 for record in records),
        "foundation_in_total_loss": all(
            abs(record["total_loss"] - record["task_loss"] - record["foundation_loss"]) < 1e-4 for record in records
        ),
        "student_has_gradient": all(record["student_grad_norm"] > 0 for record in records),
        "projector_has_gradient": all(record["projector_grad_norm"] > 0 for record in records),
        "teacher_frozen": teacher_frozen,
        "teacher_not_in_optimizer": teacher_not_in_optimizer,
        "fixed_batch_foundation_loss_decreased": records[-1]["foundation_loss"] < records[0]["foundation_loss"],
        # Detection assignment, BatchNorm, and routing can make the total task loss non-monotonic even on one batch.
        # Require at least one post-update improvement instead of manufacturing a monotonic-total-loss claim.
        "fixed_batch_total_loss_saw_decrease": min(record["total_loss"] for record in records[1:])
        < records[0]["total_loss"],
    }
    status = "passed" if all(checks.values()) else "failed"
    payload = {
        "schema_version": 1,
        "status": status,
        "claim": "p0_task_plus_foundation_single_batch_training_only_no_accuracy_claim",
        "experiment_id": config["experiment_id"],
        "config": {"path": str(config_path.relative_to(REPO_ROOT)), "sha256": _sha256(config_path), **config},
        "student": {
            "model": str(student_path.relative_to(REPO_ROOT)),
            "model_yaml_sha256": _sha256(student_path),
            "p4_source_indices": list(wrapper.tap.source_indices),
        },
        "teacher": {
            "model": config["teacher_model"],
            "requested_revision": config["teacher_revision"],
            "resolved_revision": snapshot.name,
            "license": config["teacher_license"],
            "assets": _teacher_assets(snapshot),
        },
        "checks": checks,
        "steps": records,
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "seed": seed,
        },
    }
    output = EXPERIMENT_ROOT / "results" / "d2_p0_train_smoke.json"
    # 保存完整检查结果，而不是只返回进程退出码。
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
