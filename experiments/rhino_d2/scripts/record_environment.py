#!/usr/bin/env python3
"""Write a redacted, machine-readable D2 environment manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
OUTPUT_PATH = EXPERIMENT_ROOT / "env" / "environment.json"


def _run(command: list[str]) -> str | None:
    """执行只读诊断命令并返回标准输出。

    Git 和 ``nvidia-smi`` 信息有助于解释不同机器上的速度、显存或指标差异，
    但不是实验必需依赖。因此命令不存在或失败时返回 ``None``，仍然生成
    其余环境清单，避免环境记录阻断训练实验。
    """
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=True, shell=False)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    """计算配置文件哈希，记录实验输入版本而不暴露文件内容。

    哈希不参与训练，也不会直接改善精度；它用于复盘时确认环境清单对应的
    配置版本，防止不同超参数下的结果被误当成同一组实验。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status_summary(arguments: list[str]) -> dict:
    """Return dirty-state evidence without publishing local file names."""
    status = _run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *arguments])
    lines = [] if not status else status.splitlines()
    return {
        "dirty": bool(lines),
        "entry_count": len(lines),
        "porcelain_sha256": hashlib.sha256((status or "").encode()).hexdigest(),
    }


def _hash_group(paths: list[Path]) -> dict[str, str]:
    """Hash an explicit set of version-sensitive experiment inputs."""
    return {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in sorted(paths) if path.is_file()}


def _redact_home(value: str) -> str:
    """将路径中的本机用户目录替换为稳定占位符。

    环境记录仍保留 Python 可执行文件等定位信息，同时不会把具体用户名写入
    可共享产物。可共享、可比较的清单有助于复现指标差异，但不参与训练。
    """
    home = str(Path.home())
    return value.replace(home, "$USER_HOME")


def main() -> None:
    """收集非机密的软件、硬件、Git 和配置身份信息。

    环境记录回答“指标是在什么代码、依赖、CUDA 和 GPU 上得到的”，用于
    发现依赖版本、驱动或配置变化造成的指标漂移，并为 D2 on/off 对照提供
    可审计上下文。它不改变训练结果，也不宣称提升精度。
    """
    configs = _hash_group(list((EXPERIMENT_ROOT / "configs").glob("*.yaml")))
    scripts = _hash_group(list((EXPERIMENT_ROOT / "scripts").glob("*.py")))
    protocols = _hash_group([EXPERIMENT_ROOT / "experiment_matrix.csv"])
    tests = _hash_group(
        [
            *list((EXPERIMENT_ROOT / "tests").glob("*.py")),
            REPO_ROOT / "tests" / "test_foundation_dinov2.py",
            REPO_ROOT / "tests" / "test_foundation_config.py",
            REPO_ROOT / "tests" / "test_foundation_distill_model.py",
            REPO_ROOT / "tests" / "test_engine.py",
        ]
    )
    implementation = _hash_group(
        [
            REPO_ROOT / "ultralytics" / "nn" / "foundation" / "preprocessing.py",
            REPO_ROOT / "ultralytics" / "nn" / "foundation" / "teachers" / "dinov2.py",
            REPO_ROOT / "ultralytics" / "nn" / "foundation_distill_model.py",
            REPO_ROOT / "ultralytics" / "engine" / "trainer.py",
            REPO_ROOT / "ultralytics" / "cfg" / "default.yaml",
            REPO_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26-master-n.yaml",
        ]
    )
    packages = {}
    for name in ("ultralytics", "torch", "torchvision", "transformers", "huggingface-hub", "pytest", "pyyaml"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "schema_version": 2,
        "git": {
            "base_commit": _run(["git", "merge-base", "HEAD", "origin/main"]),
            "experiment_commit": _run(["git", "rev-parse", "HEAD"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "branch": _run(["git", "branch", "--show-current"]),
            "origin": _run(["git", "remote", "get-url", "origin"]),
            "repository_state": _status_summary([]),
            "experiment_inputs_state": _status_summary(
                [
                    "experiments/rhino_d2/configs",
                    "experiments/rhino_d2/experiment_matrix.csv",
                    "experiments/rhino_d2/scripts",
                    "experiments/rhino_d2/tests",
                    "ultralytics/nn/foundation",
                    "ultralytics/nn/foundation_distill_model.py",
                    "ultralytics/cfg/default.yaml",
                    "ultralytics/cfg/__init__.py",
                    "tests/test_foundation_dinov2.py",
                    "tests/test_foundation_config.py",
                    "tests/test_foundation_distill_model.py",
                    "tests/test_engine.py",
                    "ultralytics/engine/trainer.py",
                ]
            ),
        },
        "runtime": {
            "python": sys.version,
            "executable": _redact_home(sys.executable),
            "platform": platform.platform(),
            "packages": packages,
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "driver_report": _run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
            ),
        },
        "configs": configs,
        "scripts": scripts,
        "protocols": protocols,
        "tests": tests,
        "implementation": implementation,
        "redaction": "No environment variables, credentials, tokens, or user-home contents are recorded.",
    }
    # 环境清单可提交或归档，用于复现时核对软件、硬件和配置身份。
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
