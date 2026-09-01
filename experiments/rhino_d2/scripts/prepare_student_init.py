#!/usr/bin/env python3
"""Fetch and verify the frozen official YOLO26n initialization for Candidate B."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "cache" / "yolo26n.pt"
SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
EXPECTED_BYTES = 5_544_453
EXPECTED_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"


def sha256(path: Path) -> str:
    """Return one file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path: Path) -> None:
    """Fail closed unless size and digest match the frozen asset."""
    if path.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(f"unexpected asset size: {path.stat().st_size} != {EXPECTED_BYTES}")
    digest = sha256(path)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"unexpected asset sha256: {digest} != {EXPECTED_SHA256}")


def main() -> None:
    """Download atomically when needed, then verify the immutable identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true", help="forbid download and verify the local asset only")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_file():
        verify(output)
        print(f"verified {output} sha256={EXPECTED_SHA256}")
        return
    if args.offline:
        raise FileNotFoundError(f"frozen Student initialization is absent: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with requests.get(SOURCE_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    try:
        verify(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"downloaded and verified {output} sha256={EXPECTED_SHA256}")


if __name__ == "__main__":
    main()
