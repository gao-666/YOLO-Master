#!/usr/bin/env python3
"""Run the three pre-registered DINOv3-S P2-02 ON128 seeds sequentially."""

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
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
CONFIG = EXPERIMENT_ROOT / "configs/d2_v3_p2_align128.yaml"
VALIDATOR = EXPERIMENT_ROOT / "scripts/validate_p2_align_dim.py"
DEFAULT_PROJECT = REPO_ROOT / "runs/rhino_d2/v3_p2_align_dim"
AUDIT_ROOT = REPO_ROOT / "runs/rhino_d2/p2_align_dim"
SEEDS = (20260824, 20260825, 20260826)


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_yolo() -> str:
    """Resolve the CLI from PATH or the active Python environment."""
    executable = shutil.which("yolo")
    if executable:
        return executable
    executable_name = "yolo.exe" if os.name == "nt" else "yolo"
    for candidate in (
        Path(sys.executable).with_name(executable_name),
        Path(sys.executable).parent / "Scripts" / executable_name,
    ):
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("yolo CLI not found; activate the frozen yolo-master-d2 environment")


def build_command(yolo: str, seed: int, project: Path) -> tuple[list[str], str]:
    """Build one deterministic ON128 command without executing it."""
    name = f"v3-p2-align128-s{seed}"
    config = CONFIG.relative_to(REPO_ROOT).as_posix()
    return [yolo, f"cfg={config}", f"seed={seed}", f"name={name}", f"project={project}", "exist_ok=False"], name


def git_state() -> dict[str, Any]:
    """Record the exact repository state used by a run."""
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


def checkpoint_inventory(run_dir: Path) -> list[dict[str, Any]]:
    """Inventory saved checkpoints without importing model code."""
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted((run_dir / "weights").glob("*.pt"))
    ]


def console_write(line: str) -> None:
    """Write subprocess output safely on a narrow Windows console."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(line.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    sys.stdout.flush()


def run_admission_audit() -> Path:
    """Run the fail-closed training-admission validator once per invocation."""
    output = AUDIT_ROOT / "preflight/validation.json"
    subprocess.run([sys.executable, str(VALIDATOR), "--output", str(output)], cwd=REPO_ROOT, check=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload["status"] != "passed" or not payload["decision"]["three_seed_on128_training_allowed"]:
        raise RuntimeError("P2-02 training admission was not granted")
    return output


def run_seed(seed: int, project: Path, preflight: Path, dry_run: bool) -> dict[str, Any]:
    """Run one seed, tee all output, and bind it to an immutable manifest."""
    yolo = resolve_yolo()
    command, name = build_command(yolo, seed, project)
    run_dir = project / name
    log_path = AUDIT_ROOT / "logs" / f"{name}.log"
    manifest_path = AUDIT_ROOT / "manifests" / f"{name}.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "DINOv3-S P2-02 align_dim 64 to 128",
        "arm": "on128",
        "seed": seed,
        "config": CONFIG.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": sha256(CONFIG),
        "preflight": preflight.relative_to(REPO_ROOT).as_posix(),
        "preflight_sha256": sha256(preflight),
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

    log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("YOLO_CONFIG_DIR", str(AUDIT_ROOT / "yolo_config"))
    environment.setdefault("HF_HOME", str(EXPERIMENT_ROOT / "cache/huggingface"))
    environment.setdefault("HF_HUB_CACHE", str(EXPERIMENT_ROOT / "cache/huggingface/hub"))
    environment.setdefault("MPLCONFIGDIR", str(AUDIT_ROOT / "matplotlib"))
    environment.setdefault("WANDB_DISABLED", "true")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
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
    """Validate once, then run selected seeds sequentially on one GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    preflight = run_admission_audit()
    for seed in args.seeds:
        payload = run_seed(seed, args.project.resolve(), preflight, args.dry_run)
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
