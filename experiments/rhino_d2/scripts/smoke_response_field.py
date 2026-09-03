#!/usr/bin/env python3
"""Run the synthetic-only P2-04 Response-Field implementation gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics.nn.foundation import (
    GLOBAL_BATCH_INDEX_VERSION,
    P4AlignmentProjector,
    build_response_field_paired_view,
    preserve_batchnorm_buffers,
    response_field_kd_loss,
    strict_cosine_kd_loss,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/p2_response_field/smoke/d2_v3_p2_response_field_smoke.json"
DEFAULT_LOG = EXPERIMENT_ROOT / "results/p2_response_field/smoke/d2_v3_p2_response_field_smoke.log"
INPUT_PATHS = (
    "experiments/rhino_d2/DINOV3_P2_RESPONSE_FIELD_TRAIN_PROTOCOL.md",
    "experiments/rhino_d2/DINOV3_P2_RESPONSE_FIELD_TRAIN_PROTOCOL_AMENDMENT.md",
    "experiments/rhino_d2/scripts/smoke_response_field.py",
    "experiments/rhino_d2/tests/test_response_field.py",
    "ultralytics/nn/foundation/__init__.py",
    "ultralytics/nn/foundation/response.py",
)


class TinyStudent(nn.Module):
    """Synthetic Student with train-mode BatchNorm semantics."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=False)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.silu(self.bn(self.conv(images)))


def sha256(path: Path) -> str:
    """Return one file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    """Run a read-only Git query from the repository root."""
    return subprocess.check_output(("git", *args), cwd=REPO_ROOT, text=True, encoding="utf-8").strip()


def teacher_feature(images: torch.Tensor) -> torch.Tensor:
    """Return a deterministic four-channel frozen synthetic Teacher feature."""
    return torch.cat((images, images.mean(dim=1, keepdim=True)), dim=1).detach()


def bn_buffers(roots: dict[str, nn.Module]) -> dict[str, torch.Tensor]:
    """Clone all protected BatchNorm buffers with stable names."""
    result = {}
    for root_name, root in roots.items():
        for module_name, module in root.named_modules():
            if not isinstance(module, _BatchNorm):
                continue
            prefix = f"{root_name}.{module_name}"
            for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
                result[f"{prefix}.{buffer_name}"] = getattr(module, buffer_name).detach().clone()
    return result


def tensor_map_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    """Return exact equality for two named tensor maps."""
    return left.keys() == right.keys() and all(torch.equal(left[name], right[name]) for name in left)


def paired_batch() -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    """Build the deterministic synthetic paired view used by all arms."""
    generator = torch.Generator(device="cpu").manual_seed(20260824)
    clean = torch.rand((4, 3, 16, 16), generator=generator)
    perturbed, records = build_response_field_paired_view(
        clean,
        [f"train2017/synthetic_{index:04d}.jpg" for index in range(clean.shape[0])],
        seed=20260824,
        epoch_index=7,
        batch_index_within_epoch=3,
        num_batches_per_epoch=512,
    )
    return clean, perturbed, records


def run_supervision_arm(
    arm: str, clean: torch.Tensor, perturbed: torch.Tensor
) -> tuple[dict[str, Any], dict[str, nn.Module]]:
    """Run one synthetic A/B/C batch through the exact paired BN path."""
    torch.manual_seed(17)
    student = TinyStudent().train()
    projector = P4AlignmentProjector(4, 4, align_dim=4, use_norm=True).train()
    roots = {"student": student, "projector": projector}
    phases = ["clean"]
    flags: dict[str, dict[str, list[bool]]] = {"clean": {}, "perturbed": {}}
    hooks = []
    for root_name, root in roots.items():
        for module_name, module in root.named_modules():
            if isinstance(module, _BatchNorm):
                name = f"{root_name}.{module_name}"
                hooks.append(
                    module.register_forward_pre_hook(
                        lambda current, _inputs, name=name: (
                            flags[phases[0]].setdefault(name, []).append(current.training)
                        )
                    )
                )

    clean_feature = student(clean)
    student_clean, teacher_clean = projector(clean_feature, teacher_feature(clean))
    post_clean_buffers = bn_buffers(roots)
    with preserve_batchnorm_buffers(roots) as snapshot:
        phases[0] = "perturbed"
        perturbed_feature = student(perturbed)
        perturbed_feature.retain_grad()
        student_perturbed, teacher_perturbed = projector(perturbed_feature, teacher_feature(perturbed))
        changed_during_perturbed_forward = not snapshot.matches()
    restored_buffers = bn_buffers(roots)
    for hook in hooks:
        hook.remove()

    task_loss = clean_feature.square().mean()
    static_clean = strict_cosine_kd_loss(student_clean, teacher_clean)
    static_perturbed = strict_cosine_kd_loss(student_perturbed, teacher_perturbed)
    response = response_field_kd_loss(student_clean, student_perturbed, teacher_clean, teacher_perturbed)
    if arm == "a":
        foundation = task_loss.new_zeros(())
    elif arm == "b":
        foundation = 0.15 * (static_clean + static_perturbed)
    elif arm == "c":
        foundation = 0.15 * (static_clean + response)
    else:
        raise ValueError(f"Unknown synthetic arm: {arm!r}.")
    total = task_loss + foundation
    total.backward()
    perturbed_gradient = perturbed_feature.grad
    projector_gradient = projector.student_proj[0].weight.grad
    record = {
        "arm": arm.upper(),
        "task_loss": float(task_loss.detach()),
        "foundation_loss": float(foundation.detach()),
        "static_clean_loss": float(static_clean.detach()),
        "static_perturbed_loss": float(static_perturbed.detach()),
        "response_loss": float(response.detach()),
        "bn_changed_during_perturbed_forward": changed_during_perturbed_forward,
        "bn_buffers_restored_bitwise": tensor_map_equal(post_clean_buffers, restored_buffers),
        "clean_bn_training_flags": flags["clean"],
        "perturbed_bn_training_flags": flags["perturbed"],
        "bn_training_flags_match_and_true": flags["clean"] == flags["perturbed"]
        and all(all(values) for values in flags["clean"].values()),
        "perturbed_feature_gradient_norm": (
            0.0 if perturbed_gradient is None else float(torch.linalg.vector_norm(perturbed_gradient.detach()))
        ),
        "projector_gradient_norm": (
            0.0 if projector_gradient is None else float(torch.linalg.vector_norm(projector_gradient.detach()))
        ),
    }
    return record, roots


def run_task_only_step(
    student: TinyStudent,
    projector: P4AlignmentProjector,
    clean: torch.Tensor,
    perturbed: torch.Tensor,
    *,
    include_unused_perturbed_forward: bool,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run the Arm-A or clean-only reference step and capture exact state."""
    roots = {"student": student, "projector": projector}
    optimizer = torch.optim.SGD((*student.parameters(), *projector.parameters()), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    clean_feature = student(clean)
    projector(clean_feature, teacher_feature(clean))
    if include_unused_perturbed_forward:
        with preserve_batchnorm_buffers(roots):
            projector(student(perturbed), teacher_feature(perturbed))
    clean_feature.square().mean().backward()
    gradients = {
        f"student.{name}": parameter.grad.detach().clone()
        for name, parameter in student.named_parameters()
        if parameter.grad is not None
    }
    gradients.update(
        {
            f"projector.{name}": parameter.grad.detach().clone()
            for name, parameter in projector.named_parameters()
            if parameter.grad is not None
        }
    )
    optimizer.step()
    parameters = {f"student.{name}": parameter.detach().clone() for name, parameter in student.named_parameters()}
    parameters.update(
        {f"projector.{name}": parameter.detach().clone() for name, parameter in projector.named_parameters()}
    )
    return {"gradients": gradients, "parameters": parameters, "buffers": bn_buffers(roots)}


def arm_a_matches_clean_reference(clean: torch.Tensor, perturbed: torch.Tensor) -> bool:
    """Verify that the unused Arm-A branch has no gradient, update, or persistent-buffer effect."""
    torch.manual_seed(23)
    reference_student = TinyStudent().train()
    reference_projector = P4AlignmentProjector(4, 4, align_dim=4, use_norm=True).train()
    arm_a_student = copy.deepcopy(reference_student)
    arm_a_projector = copy.deepcopy(reference_projector)
    reference = run_task_only_step(
        reference_student,
        reference_projector,
        clean,
        perturbed,
        include_unused_perturbed_forward=False,
    )
    arm_a = run_task_only_step(
        arm_a_student,
        arm_a_projector,
        clean,
        perturbed,
        include_unused_perturbed_forward=True,
    )
    return all(tensor_map_equal(reference[group], arm_a[group]) for group in reference)


def optimize_response_loss(clean: torch.Tensor, perturbed: torch.Tensor, steps: int = 12) -> list[float]:
    """Show that the train-semantics response objective can decrease without BN leakage."""
    torch.manual_seed(31)
    student = TinyStudent().train()
    projector = P4AlignmentProjector(4, 4, align_dim=4, use_norm=True).train()
    roots = {"student": student, "projector": projector}
    optimizer = torch.optim.SGD((*student.parameters(), *projector.parameters()), lr=0.08, momentum=0.0)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        clean_feature = student(clean)
        student_clean, teacher_clean = projector(clean_feature, teacher_feature(clean))
        with preserve_batchnorm_buffers(roots):
            perturbed_feature = student(perturbed)
            student_perturbed, teacher_perturbed = projector(perturbed_feature, teacher_feature(perturbed))
        loss = strict_cosine_kd_loss(student_clean, teacher_clean) + response_field_kd_loss(
            student_clean,
            student_perturbed,
            teacher_clean,
            teacher_perturbed,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Synthetic response objective became non-finite.")
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    return losses


def build_result() -> dict[str, Any]:
    """Execute the synthetic-only gate and return its auditable record."""
    torch.use_deterministic_algorithms(True)
    arm_inputs = {arm: paired_batch() for arm in ("A", "B", "C")}
    clean, perturbed, manifest = arm_inputs["A"]
    arm_records = {}
    for arm, (arm_clean, arm_perturbed, _) in arm_inputs.items():
        record, _ = run_supervision_arm(arm.lower(), arm_clean, arm_perturbed)
        arm_records[arm] = record
    loss_curve = optimize_response_loss(clean, perturbed)
    status = git("status", "--porcelain", "--", *INPUT_PATHS)
    checks = {
        "experiment_inputs_clean": not status,
        "same_clean_tensor_for_abc": all(torch.equal(clean, values[0]) for values in arm_inputs.values()),
        "same_perturbed_tensor_for_abc": all(torch.equal(perturbed, values[1]) for values in arm_inputs.values()),
        "same_paired_manifest_for_abc": all(manifest == values[2] for values in arm_inputs.values()),
        "all_bn_buffers_restored_bitwise": all(
            record["bn_buffers_restored_bitwise"] for record in arm_records.values()
        ),
        "all_bn_flags_match_and_are_train": all(
            record["bn_training_flags_match_and_true"] for record in arm_records.values()
        ),
        "bc_perturbed_gradients_finite_nonzero": all(
            arm_records[arm]["perturbed_feature_gradient_norm"] > 0 and arm_records[arm]["projector_gradient_norm"] > 0
            for arm in ("B", "C")
        ),
        "arm_a_perturbed_branch_no_effect": arm_a_matches_clean_reference(clean, perturbed),
        "response_objective_decreased": loss_curve[-1] < loss_curve[0],
        "no_validation_or_formal_metrics_read": True,
        "no_calibration_executed": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"P2-04 synthetic gate failed closed: {checks}")
    return {
        "schema_version": 1,
        "status": "passed_synthetic_implementation_gate",
        "authorization_boundary": "synthetic_only_no_alpha_calibration_no_formal_training",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_state": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "experiment_inputs_dirty": bool(status),
            "experiment_input_status": status.splitlines(),
            "input_sha256": {path: sha256(REPO_ROOT / path) for path in INPUT_PATHS},
        },
        "protocol": {
            "payload_version": "dinov3-p2-response-field-v1",
            "global_batch_index_version": GLOBAL_BATCH_INDEX_VERSION,
            "lambda": 0.15,
            "synthetic_alpha": 1.0,
            "cosine_epsilon": 1e-6,
            "bn_contract": "train_semantics_with_post_clean_buffer_snapshot_restore",
        },
        "paired_view": {
            "shape": list(clean.shape),
            "records": manifest,
        },
        "arms": arm_records,
        "optimization": {
            "steps": len(loss_curve),
            "objective": "unweighted_clean_static_plus_response_synthetic_only",
            "loss_curve": loss_curve,
            "first": loss_curve[0],
            "last": loss_curve[-1],
        },
        "checks": checks,
        "claim": (
            "The Response-Field code path, deterministic paired-view contract, gradients, and BN rollback work on "
            "synthetic tensors; this is not alpha calibration and contains no detection efficacy evidence."
        ),
    }


def main() -> None:
    """Run the gate and write JSON plus a complete text log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    result = build_result()
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    args.log.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
