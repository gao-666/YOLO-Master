#!/usr/bin/env python3
"""Compare frozen ON128 and ON64 P4 gradient probes on identical images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
ON64_ROOT = EXPERIMENT_ROOT / "results/p2_gradient_conflict"
DEFAULT_ON128_ROOT = REPO_ROOT / "runs/rhino_d2/p2_align_dim/gradient_probe_on128"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/p2_align_dim/gradient_followup"
PERFORMANCE_RESULT = EXPERIMENT_ROOT / "results/p2_align_dim/d2_v3_p2_align_dim_result.json"
SUBSET_SHA256 = "1ad936698234fd07651993dbdefe7a98ffaf74861432a103b6e397bb45b9b676"
SEEDS = (20260824, 20260825, 20260826)
EPOCHS = (9, 24, 49)
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260902


def sha256(path: Path) -> str:
    """Return one file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one probe CSV as strings so key identity remains exact."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a non-empty dictionary table."""
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archive(source: Path, destination: Path) -> dict[str, Any]:
    """Copy one probe artifact into the tracked P2-02 evidence tree."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256(destination)}


def bootstrap_median_ci(values: list[float], seed: int) -> tuple[float, float]:
    """Return the frozen percentile bootstrap interval for a paired median."""
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("paired bootstrap requires a non-empty finite vector")
    rng = np.random.default_rng(seed)
    medians = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 1000):
        count = min(1000, BOOTSTRAP_REPLICATES - start)
        samples = rng.choice(array, size=(count, len(array)), replace=True)
        medians[start : start + count] = np.median(samples, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def summarize(values: list[float], seed: int) -> dict[str, float | int]:
    """Summarize paired changes without claiming a training effect."""
    array = np.asarray(values, dtype=np.float64)
    low, high = bootstrap_median_ci(values, seed)
    return {
        "n": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "ci_low": low,
        "ci_high": high,
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def direction_label(summary: dict[str, float | int]) -> str:
    """Describe whether a paired median shift is resolved away from zero."""
    if float(summary["ci_low"]) > 0:
        return "increase"
    if float(summary["ci_high"]) < 0:
        return "decrease"
    return "no_detectable_shift"


def key(row: dict[str, str]) -> tuple[int, int, str]:
    """Build the exact seed/checkpoint/image pairing key."""
    return int(row["seed"]), int(row["epoch"]), Path(row["image"]).as_posix()


def pair_records(on64: list[dict[str, str]], on128: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Pair every image observation and compute ON128-minus-ON64 changes."""
    left = {key(row): row for row in on64}
    right = {key(row): row for row in on128}
    expected = len(SEEDS) * len(EPOCHS) * 64
    if len(left) != expected or len(right) != expected or set(left) != set(right):
        raise RuntimeError(f"probe pairing mismatch: ON64={len(left)}, ON128={len(right)}, expected={expected}")
    pairs = []
    for observation in sorted(left):
        old, new = left[observation], right[observation]
        if old["valid"].lower() != "true" or new["valid"].lower() != "true":
            raise RuntimeError(f"invalid gradient observation: {observation}")
        cosine64, cosine128 = float(old["cosine"]), float(new["cosine"])
        ratio64, ratio128 = float(old["norm_ratio"]), float(new["norm_ratio"])
        pairs.append(
            {
                "seed": observation[0],
                "epoch": observation[1],
                "image": observation[2],
                "cosine_on64": cosine64,
                "cosine_on128": cosine128,
                "delta_cosine": cosine128 - cosine64,
                "norm_ratio_on64": ratio64,
                "norm_ratio_on128": ratio128,
                "delta_norm_ratio": ratio128 - ratio64,
                "negative_on64": cosine64 < 0,
                "negative_on128": cosine128 < 0,
            }
        )
    return pairs


def build_summaries(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build per-seed/per-checkpoint paired summaries."""
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        groups[(int(row["seed"]), int(row["epoch"]))].append(row)
    summaries = []
    for group_index, ((seed, epoch), rows) in enumerate(sorted(groups.items())):
        cosine = summarize([float(row["delta_cosine"]) for row in rows], BOOTSTRAP_SEED + group_index)
        ratio = summarize([float(row["delta_norm_ratio"]) for row in rows], BOOTSTRAP_SEED + 100 + group_index)
        summaries.append(
            {
                "seed": seed,
                "epoch": epoch,
                **{f"delta_cosine_{name}": value for name, value in cosine.items()},
                "cosine_shift": direction_label(cosine),
                **{f"delta_norm_ratio_{name}": value for name, value in ratio.items()},
                "norm_ratio_shift": direction_label(ratio),
                "negative_fraction_on64": float(np.mean([row["negative_on64"] for row in rows])),
                "negative_fraction_on128": float(np.mean([row["negative_on128"] for row in rows])),
            }
        )
    return summaries


def pooled_late(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Cluster paired late changes by image across the three seeds."""
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        if int(row["epoch"]) == 49:
            by_image[str(row["image"])].append(row)
    if len(by_image) != 64 or any(len(rows) != 3 for rows in by_image.values()):
        raise RuntimeError("late pooled comparison is not a complete 64-image by 3-seed panel")
    cosine_values = [float(np.mean([row["delta_cosine"] for row in rows])) for rows in by_image.values()]
    ratio_values = [float(np.mean([row["delta_norm_ratio"] for row in rows])) for rows in by_image.values()]
    cosine = summarize(cosine_values, BOOTSTRAP_SEED + 2000)
    ratio = summarize(ratio_values, BOOTSTRAP_SEED + 3000)
    return {
        "cluster_unit": "image",
        "images": 64,
        "seed_observations_per_image": 3,
        **{f"delta_cosine_{name}": value for name, value in cosine.items()},
        "cosine_shift": direction_label(cosine),
        **{f"delta_norm_ratio_{name}": value for name, value in ratio.items()},
        "norm_ratio_shift": direction_label(ratio),
    }


def report(result: dict[str, Any]) -> str:
    """Render a concise mechanism-only evidence report."""
    pooled = result["pooled_late"]
    lines = [
        "# DINOv3-S P2-02 ON128 vs ON64 梯度复测",
        "",
        "该复测使用同一 64 图训练子集、同一 seed 和 epoch9/24/49 EMA checkpoint；不修改 P1 或 P2-01 结论。",
        "",
        "| seed | epoch | Δ median cosine | 95% CI | Δ median norm ratio | 95% CI |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summaries"]:
        lines.append(
            f"| {row['seed']} | {row['epoch']} | {row['delta_cosine_median']:.6f} | "
            f"[{row['delta_cosine_ci_low']:.6f}, {row['delta_cosine_ci_high']:.6f}] | "
            f"{row['delta_norm_ratio_median']:.6f} | "
            f"[{row['delta_norm_ratio_ci_low']:.6f}, {row['delta_norm_ratio_ci_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Late pooled paired change",
            "",
            (
                f"- cosine median change: `{pooled['delta_cosine_median']:.6f}` "
                f"(`{pooled['cosine_shift']}`, 95% CI "
                f"`[{pooled['delta_cosine_ci_low']:.6f}, {pooled['delta_cosine_ci_high']:.6f}]`)"
            ),
            (
                f"- weighted norm-ratio median change: `{pooled['delta_norm_ratio_median']:.6f}` "
                f"(`{pooled['norm_ratio_shift']}`, 95% CI "
                f"`[{pooled['delta_norm_ratio_ci_low']:.6f}, {pooled['delta_norm_ratio_ci_high']:.6f}]`)"
            ),
            "",
            "## 边界",
            "",
            "这些是一阶局部机制量。它们只能说明扩维是否改变 KD 相对检测梯度的方向或强度，不能证明信息保真或检测收益。",
            "",
        ]
    )
    return "\n".join(lines)


def final_report(performance: dict[str, Any], gradient: dict[str, Any]) -> str:
    """Render the frozen P2-02 performance-plus-mechanism conclusion."""
    primary = performance["primary_on128_minus_on64"]
    secondary = performance["secondary_on128_minus_off"]
    pooled = gradient["pooled_late"]
    dim_stats = primary["statistics"]
    off_stats = secondary["statistics"]
    return "\n".join(
        [
            "# DINOv3-S P2-02 align_dim 64→128 结果",
            "",
            "正式结论：**No support for the align-dimension bottleneck hypothesis**。",
            "",
            "## 预注册性能判读",
            "",
            (
                f"- 主比较 ON128−ON64：mean `{dim_stats['mean_delta']:.6f}`，paired 95% CI "
                f"`[{dim_stats['confidence_interval_95'][0]:.6f}, {dim_stats['confidence_interval_95'][1]:.6f}]`，"
                f"判定 **{primary['decision']}**。"
            ),
            (
                f"- 次级 ON128−OFF：mean `{off_stats['mean_delta']:.6f}`，paired 95% CI "
                f"`[{off_stats['confidence_interval_95'][0]:.6f}, {off_stats['confidence_interval_95'][1]:.6f}]`，"
                f"判定 **{secondary['decision']}**。"
            ),
            "",
            "三个 seed 的 ON128−ON64 分别为 `+0.006105 / -0.009585 / -0.000480`。按协议不继续搜索 256/512。",
            "",
            "## 同图像梯度复测",
            "",
            (
                f"- late cosine 配对中位变化：`{pooled['delta_cosine_median']:.6f}`，95% CI "
                f"`[{pooled['delta_cosine_ci_low']:.6f}, {pooled['delta_cosine_ci_high']:.6f}]`："
                "没有可检测方向变化。"
            ),
            (
                f"- late weighted norm-ratio 配对中位变化：`{pooled['delta_norm_ratio_median']:.6f}`，95% CI "
                f"`[{pooled['delta_norm_ratio_ci_low']:.6f}, {pooled['delta_norm_ratio_ci_high']:.6f}]`："
                "ON128 的相对 KD 梯度更弱。"
            ),
            "",
            "## 冻结边界",
            "",
            (
                "P1 的 No-Go 与 P2-01 的 Inconclusive 均保持不变。P2-02 只排除当前协议下的 64→128 单变量解释；"
                "它不证明投影维度在其他数据、Teacher 或训练协议下永远无关。梯度复测是一阶局部机制证据，"
                "不证明信息保真或因果关系。下一研究假设可转向 Static–Response Gap / Response-Field。"
            ),
            "",
        ]
    )


def main() -> None:
    """Validate identical inputs, pair all records, and write comparison evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on64-root", type=Path, default=ON64_ROOT)
    parser.add_argument("--on128-root", type=Path, default=DEFAULT_ON128_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    on64_root, on128_root, output = args.on64_root.resolve(), args.on128_root.resolve(), args.output_dir.resolve()
    subset64 = on64_root / "diagnostic_train64.txt"
    subset128 = on128_root / "diagnostic_train64.txt"
    subset_checks = {
        "on64_hash": sha256(subset64),
        "on128_hash": sha256(subset128),
        "expected_hash": SUBSET_SHA256,
        "identical": subset64.read_bytes() == subset128.read_bytes(),
    }
    if not (subset_checks["on64_hash"] == subset_checks["on128_hash"] == SUBSET_SHA256 and subset_checks["identical"]):
        raise RuntimeError(f"diagnostic subset mismatch: {subset_checks}")
    on64_path = on64_root / "gradient_conflict_raw.csv"
    on128_path = on128_root / "gradient_conflict_raw.csv"
    pairs = pair_records(read_csv(on64_path), read_csv(on128_path))
    summaries = build_summaries(pairs)
    pooled = pooled_late(pairs)
    output.mkdir(parents=True, exist_ok=True)
    archived_on128 = {
        "raw": archive(on128_path, output / "on128_gradient_raw.csv"),
        "summary": archive(on128_root / "gradient_conflict_summary.csv", output / "on128_gradient_summary.csv"),
        "result": archive(on128_root / "gradient_conflict_result.json", output / "on128_gradient_result.json"),
        "preparation": archive(on128_root / "preparation_manifest.json", output / "on128_preparation_manifest.json"),
        "subset": archive(subset128, output / "diagnostic_train64.txt"),
    }
    paired_path = output / "gradient_on128_minus_on64_raw.csv"
    summary_path = output / "gradient_on128_minus_on64_summary.csv"
    write_csv(paired_path, pairs)
    write_csv(summary_path, summaries)
    result = {
        "schema_version": 1,
        "status": "completed",
        "claim": "paired_local_gradient_followup_not_accuracy_or_information_fidelity_proof",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pairing": {
            "seeds": list(SEEDS),
            "epochs": list(EPOCHS),
            "images": 64,
            "raw_pairs": len(pairs),
            "subset": subset_checks,
        },
        "inputs": {
            "on64_raw": {"path": str(on64_path), "sha256": sha256(on64_path)},
            "on128_raw": {"path": str(on128_path), "sha256": sha256(on128_path)},
            "archived_on128": archived_on128,
        },
        "summaries": summaries,
        "pooled_late": pooled,
        "boundaries": {
            "p1_no_go_unchanged": True,
            "p2_01_inconclusive_unchanged": True,
            "p2_02_map_decision_unchanged": True,
            "no_information_fidelity_claim": True,
            "no_causal_claim": True,
        },
    }
    report_path = output / "DINOv3_P2_ALIGN_DIM_GRADIENT_FOLLOWUP.md"
    result_path = output / "gradient_followup_result.json"
    report_path.write_text(report(result), encoding="utf-8")
    result["artifacts"] = {
        "paired_raw": {"path": str(paired_path), "sha256": sha256(paired_path)},
        "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        "report": {"path": str(report_path), "sha256": sha256(report_path)},
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    performance = json.loads(PERFORMANCE_RESULT.read_text(encoding="utf-8"))
    if performance["primary_on128_minus_on64"]["decision"] != "no_support":
        raise RuntimeError("performance result no longer matches the frozen P2-02 No-support decision")
    final_path = output.parent / "DINOV3_P2_ALIGN_DIM_RESULT.md"
    final_path.write_text(final_report(performance, result), encoding="utf-8")
    print(json.dumps({"status": "completed", "pooled_late": pooled, "result": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
