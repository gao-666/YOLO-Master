#!/usr/bin/env python3
"""Run reproducible D2 P1 arms with complete console logs and epoch checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
ARM_CONFIGS = {
    "off": EXPERIMENT_ROOT / "configs" / "d2_off.yaml",
    "on": EXPERIMENT_ROOT / "configs" / "d2_on.yaml",
    "on-calibrated": EXPERIMENT_ROOT / "configs" / "d2_on_calibrated.yaml",
    "cosine-only": EXPERIMENT_ROOT / "configs" / "d2_ablation_cosine_only.yaml",
    "v3-baseline-sanity": EXPERIMENT_ROOT / "configs" / "d2_v3_off_sanity.yaml",
    "v3-baseline-recovery-a": EXPERIMENT_ROOT / "configs" / "d2_v3_baseline_recovery_a.yaml",
    "v3-baseline-recovery-b": EXPERIMENT_ROOT / "configs" / "d2_v3_baseline_recovery_b.yaml",
}


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict:
    """Record the exact repository state that produced a run."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout
    experiment_porcelain = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "experiments/rhino_d2/configs",
            "experiments/rhino_d2/datasets",
            "experiments/rhino_d2/experiment_matrix.csv",
            "experiments/rhino_d2/scripts",
            "experiments/rhino_d2/tests",
            "ultralytics/nn/foundation",
            "ultralytics/nn/foundation_distill_model.py",
            "ultralytics/cfg/default.yaml",
            "ultralytics/cfg/__init__.py",
            "ultralytics/engine/trainer.py",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(porcelain.strip()),
        "porcelain_sha256": hashlib.sha256(porcelain.encode()).hexdigest(),
        "experiment_inputs_dirty": bool(experiment_porcelain.strip()),
        "experiment_inputs_porcelain_sha256": hashlib.sha256(experiment_porcelain.encode()).hexdigest(),
    }


def build_command(yolo: str, arm: str, seed: int, project: Path) -> tuple[list[str], str]:
    """Build one deterministic CLI command without executing it."""
    name = f"{arm}-s{seed}"
    config = ARM_CONFIGS[arm].relative_to(REPO_ROOT)
    return [yolo, f"cfg={config}", f"seed={seed}", f"name={name}", f"project={project}", "exist_ok=False"], name


def checkpoint_inventory(run_dir: Path) -> list[dict]:
    """Inventory saved checkpoints without loading model code."""
    weights = run_dir / "weights"
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(weights.glob("*.pt"))
    ]


def resolve_yolo() -> str:
    """Resolve the CLI from PATH or the active Python environment."""
    executable = shutil.which("yolo")
    if executable:
        return executable
    executable_name = "yolo.exe" if os.name == "nt" else "yolo"
    candidates = [
        Path(sys.executable).with_name(executable_name),
        Path(sys.executable).parent / "Scripts" / executable_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("yolo CLI not found; activate the yolo-master-d2 environment first")


def console_write(line: str, stream=None) -> None:
    """Write subprocess output without failing on a narrow Windows console encoding."""
    stream = stream or sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    safe_line = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
    stream.write(safe_line)
    stream.flush()


def run_arm(arm: str, seed: int, project: Path, dry_run: bool) -> dict:
    """Run one arm, tee its complete console output, and write a local manifest."""
    yolo = resolve_yolo()
    command, name = build_command(yolo, arm, seed, project)
    run_dir = project / name
    log_dir = REPO_ROOT / "runs" / "rhino_d2" / "logs"
    manifest_dir = REPO_ROOT / "runs" / "rhino_d2" / "manifests"
    log_path = log_dir / f"{name}.log"
    manifest_path = manifest_dir / f"{name}.json"
    manifest = {
        "schema_version": 2,
        "arm": arm,
        "seed": seed,
        "config": str(ARM_CONFIGS[arm].relative_to(REPO_ROOT)),
        "config_sha256": sha256(ARM_CONFIGS[arm]),
        "command": command,
        "run_dir": str(run_dir),
        "log": str(log_path),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "source_state": git_state(),
        "runner_sha256": sha256(Path(__file__)),
    }
    if dry_run:
        return manifest
    if run_dir.exists():
        raise FileExistsError(f"refusing to mix results in existing run directory: {run_dir}")

    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / "runs" / "rhino_d2" / "config"))
    env.setdefault("HF_HOME", str(EXPERIMENT_ROOT / "cache" / "huggingface"))
    env.setdefault("HF_HUB_CACHE", str(EXPERIMENT_ROOT / "cache" / "huggingface" / "hub"))
    env.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / "rhino_d2" / "matplotlib"))
    env.setdefault("WANDB_DISABLED", "true")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            console_write(line)
        returncode = process.wait()
    manifest.update(
        {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode,
            "log_sha256": sha256(log_path),
            "checkpoints": checkpoint_inventory(run_dir),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if returncode:
        raise SystemExit(returncode)
    return manifest


def main() -> None:
    """Run selected arms sequentially so one GPU never receives overlapping jobs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=ARM_CONFIGS, default=["off", "on"])
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "rhino_d2" / "p1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for on_arm in ("on", "on-calibrated"):
        if on_arm not in args.arms:
            continue
        command = [sys.executable, EXPERIMENT_ROOT / "scripts" / "validate_pair.py"]
        if on_arm == "on-calibrated":
            command.extend(
                [
                    "--on",
                    ARM_CONFIGS[on_arm],
                    "--output",
                    EXPERIMENT_ROOT / "results" / "d2_calibrated_pair_validation.json",
                ]
            )
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    for arm in args.arms:
        print(json.dumps(run_arm(arm, args.seed, args.project.resolve(), args.dry_run), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
