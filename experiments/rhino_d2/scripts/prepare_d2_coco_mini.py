#!/usr/bin/env python3
"""Prepare the frozen D2 COCO mini subset without downloading full COCO archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT.parent / "datasets" / "coco_d2_mini_2048_seed20260901"
DEFAULT_ARCHIVE = REPO_ROOT.parent / "datasets" / "coco_d2_sources" / "coco2017labels.zip"
DATASET_STEM = "d2_coco_mini_2048_seed20260901"
SEED = 20260901
SPLIT_SIZES = {"train2017": 2048, "val2017": 512}
LABEL_ARCHIVE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
LABEL_ARCHIVE_SHA256 = "51a5175c894a7a1010f90eb4cba613473445f02633b684ed46c0292a997d0234"
IMAGE_URL = "http://images.cocodataset.org/{split}/{filename}"


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest for bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    """Return a SHA-256 digest for one file."""
    return sha256_bytes(path.read_bytes())


def sha256_text_lf(path: Path) -> str:
    """Hash UTF-8 text after canonical LF normalization for cross-platform identity."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return sha256_bytes(text.encode("utf-8"))


def split_seed(seed: int, split: str) -> int:
    """Derive independent deterministic split seeds from the protocol seed."""
    digest = hashlib.sha256(f"{seed}:{split}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def parse_source_list(raw: bytes, split: str) -> list[str]:
    """Normalize one official split list to unique JPEG basenames."""
    names = [Path(line.strip()).name for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)) or any(not name.endswith(".jpg") for name in names):
        raise ValueError(f"invalid or duplicate source entries for {split}")
    return names


def select_names(names: list[str], size: int, seed: int, split: str) -> list[str]:
    """Select and sort one frozen split."""
    if len(names) < size:
        raise ValueError(f"{split} provides {len(names)} images, fewer than requested {size}")
    return sorted(random.Random(split_seed(seed, split)).sample(names, size))


def write_list(path: Path, split: str, names: list[str]) -> None:
    """Write dataset-relative image paths in stable order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"./images/{split}/{name}\n" for name in names), encoding="utf-8", newline="\n")


def valid_jpeg(path: Path) -> bool:
    """Reject empty, truncated, or non-JPEG download payloads."""
    if not path.is_file() or path.stat().st_size < 4:
        return False
    with path.open("rb") as handle:
        start = handle.read(2)
        handle.seek(-2, os.SEEK_END)
        end = handle.read(2)
    return start == b"\xff\xd8" and end == b"\xff\xd9"


def download_image(split: str, filename: str, destination: Path, retries: int = 3) -> None:
    """Download one selected image atomically and fail on empty content."""
    if valid_jpeg(destination):
        return
    destination.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jpg.part")
    url = IMAGE_URL.format(split=split, filename=filename)
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "YOLO-Master-D2/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if not valid_jpeg(temporary):
                raise OSError(f"invalid or truncated JPEG response for {url}")
            os.replace(temporary, destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(attempt)


def aggregate_files(root: Path, paths: list[Path]) -> dict:
    """Hash a sorted path/size/content inventory without embedding local absolute paths."""
    lines: list[str] = []
    total_bytes = 0
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        lines.append(f"{relative}\0{size}\0{sha256(path)}\n")
    return {"count": len(paths), "bytes": total_bytes, "inventory_sha256": sha256_bytes("".join(lines).encode())}


def extract_selected_labels(archive: zipfile.ZipFile, dataset_root: Path, selections: dict[str, list[str]]) -> dict:
    """Extract only labels belonging to selected images; missing labels are valid backgrounds."""
    members = set(archive.namelist())
    missing: dict[str, int] = {}
    for split, names in selections.items():
        label_dir = dataset_root / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        found = 0
        for name in names:
            member = f"coco/labels/{split}/{Path(name).stem}.txt"
            if member not in members:
                continue
            destination = label_dir / f"{Path(name).stem}.txt"
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            found += 1
        missing[split] = len(names) - found
    return missing


def build_dataset_yaml(dataset_root: Path, output: Path) -> None:
    """Create a portable Ultralytics dataset YAML using the configured datasets directory."""
    source = yaml.safe_load((REPO_ROOT / "ultralytics/cfg/datasets/coco.yaml").read_text(encoding="utf-8"))
    data = {
        "path": dataset_root.name,
        "train": "train2017.txt",
        "val": "val2017.txt",
        "test": None,
        "names": source["names"],
        "d2_subset": {
            "source": "COCO 2017 official train/val splits",
            "train_size": SPLIT_SIZES["train2017"],
            "val_size": SPLIT_SIZES["val2017"],
            "seed": SEED,
            "selection": "independent split-specific SHA256-derived seed; random.sample; lexicographic output",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")


def main() -> None:
    """Prepare images, labels, frozen lists, YAML, and an integrity manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--label-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-download", action="store_true", help="Generate selections without fetching images.")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    archive_path = args.label_archive.resolve()
    expected_parent = (REPO_ROOT.parent / "datasets").resolve()
    if expected_parent not in dataset_root.parents:
        raise ValueError(f"dataset root must stay under {expected_parent}")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not archive_path.is_file() or sha256(archive_path) != LABEL_ARCHIVE_SHA256:
        raise ValueError(f"label archive missing or hash mismatch: {archive_path}")

    selection_root = EXPERIMENT_ROOT / "datasets" / DATASET_STEM
    dataset_yaml = EXPERIMENT_ROOT / "datasets" / f"{DATASET_STEM}.yaml"
    source_lists: dict[str, dict] = {}
    selections: dict[str, list[str]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for split, size in SPLIT_SIZES.items():
            raw = archive.read(f"coco/{split}.txt")
            names = parse_source_list(raw, split)
            selections[split] = select_names(names, size, SEED, split)
            source_lists[split] = {"count": len(names), "sha256": sha256_bytes(raw)}
        missing_labels = extract_selected_labels(archive, dataset_root, selections)

    for split, names in selections.items():
        write_list(selection_root / f"{split}.txt", split, names)
        write_list(dataset_root / f"{split}.txt", split, names)
    build_dataset_yaml(dataset_root, dataset_yaml)

    if not args.skip_download:
        jobs = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for split, names in selections.items():
                for name in names:
                    destination = dataset_root / "images" / split / name
                    jobs.append(executor.submit(download_image, split, name, destination))
            for index, future in enumerate(as_completed(jobs), start=1):
                future.result()
                if index % 100 == 0 or index == len(jobs):
                    print(f"downloaded/verified {index}/{len(jobs)} selected images", flush=True)

    selected_images = [dataset_root / "images" / split / name for split, names in selections.items() for name in names]
    missing_images = [path for path in selected_images if not valid_jpeg(path)]
    if missing_images and not args.skip_download:
        raise FileNotFoundError(f"{len(missing_images)} selected images are missing")
    selected_labels = [
        path
        for split, names in selections.items()
        for name in names
        if (path := dataset_root / "labels" / split / f"{Path(name).stem}.txt").is_file()
    ]
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_STEM,
        "protocol_role": "baseline_recovery_candidate_a_data_only",
        "generator": {
            "path": Path(__file__).relative_to(REPO_ROOT).as_posix(),
            "sha256_lf_canonical": sha256_text_lf(Path(__file__)),
        },
        "dataset_root": str(dataset_root),
        "source": {
            "coco_terms": "https://cocodataset.org/#termsofuse",
            "image_base_url": IMAGE_URL,
            "label_archive": {
                "url": LABEL_ARCHIVE_URL,
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "sha256": sha256(archive_path),
            },
            "source_lists": source_lists,
            "license_boundary": (
                "COCO images retain their individual Flickr terms; COCO annotations and the Ultralytics label "
                "conversion asset have separate terms. No image or label payload is committed to Git."
            ),
        },
        "selection": {
            "seed": SEED,
            "method": "independent split-specific SHA256-derived seed; random.sample; lexicographic output",
            "splits": {
                split: {
                    "size": len(names),
                    "list_path": (selection_root / f"{split}.txt").relative_to(REPO_ROOT).as_posix(),
                    "list_sha256_lf_canonical": sha256_text_lf(selection_root / f"{split}.txt"),
                    "missing_label_count": missing_labels[split],
                }
                for split, names in selections.items()
            },
        },
        "dataset_yaml": {
            "path": dataset_yaml.relative_to(REPO_ROOT).as_posix(),
            "sha256_lf_canonical": sha256_text_lf(dataset_yaml),
        },
        "payload": {
            "images": None if missing_images else aggregate_files(dataset_root, selected_images),
            "labels": aggregate_files(dataset_root, selected_labels),
            "missing_images": len(missing_images),
        },
    }
    manifest_path = selection_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
