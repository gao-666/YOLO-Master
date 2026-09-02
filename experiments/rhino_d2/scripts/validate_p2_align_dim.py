#!/usr/bin/env python3
"""Fail closed unless DINOv3 P2-02 changes only the registered alignment dimension."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
P1_CONFIG = EXPERIMENT_ROOT / "configs/d2_v3_p1_on.yaml"
P2_CONFIG = EXPERIMENT_ROOT / "configs/d2_v3_p2_align128.yaml"
PROTOCOL = EXPERIMENT_ROOT / "DINOV3_P2_ALIGN_DIM_PROTOCOL.md"
DATA_MANIFEST = EXPERIMENT_ROOT / "datasets/d2_coco_mini_2048_seed20260901/manifest.json"
TEACHER_MANIFEST = EXPERIMENT_ROOT / "env/dinov3_teacher_manifest.json"
ENVIRONMENT = EXPERIMENT_ROOT / "env/environment.json"
STUDENT_AUDIT = EXPERIMENT_ROOT / "results/d2_v3_student_init_audit.json"
P1_RESULT = EXPERIMENT_ROOT / "results/d2_v3_p1_three_seed_results.json"
DEFAULT_OUTPUT = REPO_ROOT / "runs/rhino_d2/p2_align_dim/preflight/validation.json"
ALLOWED_CONFIG_DIFFERENCES = {"foundation_align_dim", "name", "project"}
P1_TRAINING_COMMITS = {
    20260824: "323ae1f8490b17737bf0fb62deae285025b2ffd6",
    20260825: "4ac7f7af70127bef848ed0910d9b4606c6b7a7d4",
    20260826: "ef16bbf90a1d79a84dbad35558c4efd856673355",
}
CORE_SOURCE_PATHS = (
    "ultralytics/cfg/default.yaml",
    "ultralytics/cfg/__init__.py",
    "ultralytics/cfg/models/26/yolo26-master-n.yaml",
    "ultralytics/engine/trainer.py",
    "ultralytics/models/yolo/detect/train.py",
    "ultralytics/nn/tasks.py",
    "ultralytics/utils/loss.py",
    "ultralytics/nn/foundation",
    "ultralytics/nn/foundation_distill_model.py",
)
P2_INPUT_PATHS = (
    "experiments/rhino_d2/DINOV3_P2_ALIGN_DIM_PROTOCOL.md",
    "experiments/rhino_d2/configs/d2_v3_p2_align128.yaml",
    "experiments/rhino_d2/scripts/validate_p2_align_dim.py",
    "experiments/rhino_d2/scripts/run_p2_align_dim.py",
    "experiments/rhino_d2/scripts/summarize_p2_align_dim.py",
    "experiments/rhino_d2/tests/test_p2_align_dim_protocol.py",
)


def sha256(path: Path) -> str:
    """Return one file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON mapping."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON mapping: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a YAML mapping: {path}")
    return value


def aggregate_files(root: Path, paths: list[Path]) -> dict[str, Any]:
    """Reproduce the frozen dataset path/size/content inventory hash."""
    lines = []
    total_bytes = 0
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        lines.append(f"{relative}\0{size}\0{sha256(path)}\n")
    inventory = hashlib.sha256("".join(lines).encode()).hexdigest()
    return {"count": len(paths), "bytes": total_bytes, "inventory_sha256": inventory}


def config_differences(p1: dict[str, Any], p2: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return every resolved YAML difference between P1 ON64 and P2 ON128."""
    return {
        key: {"p1_on64": p1.get(key), "p2_on128": p2.get(key)}
        for key in sorted(set(p1) | set(p2))
        if p1.get(key) != p2.get(key)
    }


def core_source_audit() -> dict[str, Any]:
    """Verify relevant training code is byte-identical in Git to all formal P1 training nodes."""
    per_seed = {}
    for seed, commit in P1_TRAINING_COMMITS.items():
        changed = subprocess.run(
            ["git", "diff", "--name-only", commit, "--", *CORE_SOURCE_PATHS],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        per_seed[str(seed)] = {"commit": commit, "changed_paths": changed, "unchanged": not changed}
    current_files = []
    for value in CORE_SOURCE_PATHS:
        path = REPO_ROOT / value
        if path.is_dir():
            current_files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        elif path.is_file():
            current_files.append(path)
    inventory_lines = [
        f"{path.relative_to(REPO_ROOT).as_posix()}\0{path.stat().st_size}\0{sha256(path)}\n"
        for path in sorted(set(current_files))
    ]
    return {
        "per_seed": per_seed,
        "all_unchanged": all(value["unchanged"] for value in per_seed.values()),
        "current_inventory_sha256": hashlib.sha256("".join(inventory_lines).encode()).hexdigest(),
        "current_file_count": len(inventory_lines),
    }


def dataset_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify dataset metadata, lists, images, and labels against the frozen manifest."""
    root = Path(manifest["dataset_root"]).resolve()
    train = root / "train2017.txt"
    val = root / "val2017.txt"
    images = [path for path in (root / "images").rglob("*") if path.is_file()]
    labels = [path for path in (root / "labels").rglob("*.txt") if path.is_file()]
    image_inventory = aggregate_files(root, images)
    label_inventory = aggregate_files(root, labels)
    checks = {
        "train_list_hash": sha256(train) == manifest["selection"]["splits"]["train2017"]["list_sha256_lf_canonical"],
        "val_list_hash": sha256(val) == manifest["selection"]["splits"]["val2017"]["list_sha256_lf_canonical"],
        "image_inventory": image_inventory == manifest["payload"]["images"],
        "label_inventory": label_inventory == manifest["payload"]["labels"],
        "missing_images_zero": manifest["payload"]["missing_images"] == 0,
    }
    return {
        "root": str(root),
        "checks": checks,
        "images": image_inventory,
        "labels": label_inventory,
        "passed": all(checks.values()),
    }


def teacher_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify the local DINOv3-S assets used by P1 remain unchanged."""
    model = manifest["vits16"]
    root = Path(model["local_path"]).resolve()
    files = {}
    for name, expected in model["files"].items():
        path = root / name
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "matches": path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"],
        }
    return {"root": str(root), "files": files, "passed": all(value["matches"] for value in files.values())}


def runtime_audit(environment: dict[str, Any]) -> dict[str, Any]:
    """Compare the active runtime with the previously recorded D2 environment."""
    expected = environment["runtime"]
    package_names = tuple(expected["packages"])
    packages = {name: importlib.metadata.version(name) for name in package_names}
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    checks = {
        "python": sys.version.split()[0] == expected["python"].split()[0],
        "platform": platform.platform() == expected["platform"],
        "packages": packages == expected["packages"],
        "cuda_available": torch.cuda.is_available() == environment["cuda"]["available"],
        "cuda_runtime": torch.version.cuda == environment["cuda"]["runtime"],
        "gpu": gpu == environment["cuda"]["devices"][0],
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "current": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": gpu,
        },
    }


def git_input_state() -> dict[str, Any]:
    """Return HEAD and scoped dirty state for all registered P2 inputs."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain", "--", *P2_INPUT_PATHS, *CORE_SOURCE_PATHS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "head": head,
        "scope_dirty": bool(porcelain.strip()),
        "scope_porcelain_sha256": hashlib.sha256(porcelain.encode()).hexdigest(),
    }


def build_payload(*, precommit: bool) -> dict[str, Any]:
    """Build the complete P2-02 admission decision."""
    p1 = load_yaml(P1_CONFIG)
    p2 = load_yaml(P2_CONFIG)
    differences = config_differences(p1, p2)
    dataset_manifest = load_json(DATA_MANIFEST)
    teacher_manifest = load_json(TEACHER_MANIFEST)
    environment = load_json(ENVIRONMENT)
    student_audit = load_json(STUDENT_AUDIT)
    p1_result = load_json(P1_RESULT)
    source = core_source_audit()
    dataset = dataset_audit(dataset_manifest)
    teacher = teacher_audit(teacher_manifest)
    runtime = runtime_audit(environment)
    git = git_input_state()
    student_path = REPO_ROOT / p2["pretrained"]
    checks = {
        "config_differences_exact": set(differences) == ALLOWED_CONFIG_DIFFERENCES,
        "align_dim_64_to_128": p1["foundation_align_dim"] == 64 and p2["foundation_align_dim"] == 128,
        "weight_not_recalibrated": p1["foundation_loss_weight"] == p2["foundation_loss_weight"] == 0.15,
        "same_student_initialization_path": p1["pretrained"] == p2["pretrained"],
        "student_initialization_hash": sha256(student_path) == student_audit["source"]["sha256"],
        "student_initialization_audit_passed": student_audit["status"] == "passed",
        "same_data": p1["data"] == p2["data"],
        "dataset_unchanged": dataset["passed"],
        "teacher_assets_unchanged": teacher["passed"],
        "runtime_materially_unchanged": runtime["passed"],
        "training_source_unchanged": source["all_unchanged"],
        "p1_reference_is_frozen_no_go": p1_result["decision"]["status"] == "no_go",
        "registered_inputs_clean": not git["scope_dirty"],
    }
    gating_checks = {
        key: value for key, value in checks.items() if not (precommit and key == "registered_inputs_clean")
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(gating_checks.values()) else "failed",
        "mode": "precommit" if precommit else "training_admission",
        "claim": "p2_align_dim_single_variable_admission_not_efficacy",
        "allowed_config_differences": sorted(ALLOWED_CONFIG_DIFFERENCES),
        "observed_config_differences": differences,
        "checks": checks,
        "gating_checks": gating_checks,
        "git": git,
        "source": source,
        "dataset": dataset,
        "teacher": teacher,
        "runtime": runtime,
        "files": {
            "protocol": {"path": str(PROTOCOL.relative_to(REPO_ROOT)), "sha256": sha256(PROTOCOL)},
            "p1_config": {"path": str(P1_CONFIG.relative_to(REPO_ROOT)), "sha256": sha256(P1_CONFIG)},
            "p2_config": {"path": str(P2_CONFIG.relative_to(REPO_ROOT)), "sha256": sha256(P2_CONFIG)},
            "student_init": {"path": str(student_path.relative_to(REPO_ROOT)), "sha256": sha256(student_path)},
            "data_manifest": {"path": str(DATA_MANIFEST.relative_to(REPO_ROOT)), "sha256": sha256(DATA_MANIFEST)},
            "teacher_manifest": {
                "path": str(TEACHER_MANIFEST.relative_to(REPO_ROOT)),
                "sha256": sha256(TEACHER_MANIFEST),
            },
            "p1_result": {"path": str(P1_RESULT.relative_to(REPO_ROOT)), "sha256": sha256(P1_RESULT)},
        },
        "decision": {"three_seed_on128_training_allowed": all(gating_checks.values()) and not precommit},
    }


def main() -> None:
    """Write the admission audit and reject training on any uncontrolled difference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--precommit", action="store_true")
    args = parser.parse_args()
    payload = build_payload(precommit=args.precommit)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
