#!/usr/bin/env python3
"""Build publication-facing P2-03 tables from the immutable formal raw evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = EXPERIMENT_ROOT / "results/p2_response_gap/formal"
RAW = FORMAL_ROOT / "d2_v3_p2_response_gap_raw.csv"
RESULT = FORMAL_ROOT / "d2_v3_p2_response_gap_result.json"
SUMMARY_CSV = FORMAL_ROOT / "d2_v3_p2_response_gap_condition_summary.csv"
REPORT = FORMAL_ROOT / "DINOV3_P2_RESPONSE_GAP_RESULT.md"
PUBLICATION_MANIFEST = FORMAL_ROOT / "d2_v3_p2_response_gap_publication_manifest.json"


def sha256(path: Path) -> str:
    """Return one file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(rows: list[dict[str, str]], key: str) -> float:
    """Return one finite arithmetic mean."""
    return statistics.fmean(float(row[key]) for row in rows)


def family(condition: str) -> str:
    """Return the perturbation family prefix."""
    return condition.split(":", 1)[0]


def build_condition_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Summarize every seed-condition pair without selecting favorable perturbations."""
    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), row["arm"], row["condition"])].append(row)
    output = []
    for seed in sorted({key[0] for key in grouped}):
        for condition in dict.fromkeys(row["condition"] for row in rows):
            off, on = grouped[(seed, "off", condition)], grouped[(seed, "on64", condition)]
            if len(off) != len(on) or len(off) != 128:
                raise RuntimeError(f"incomplete paired condition: {seed} {condition}")
            off_static, on_static = mean(off, "static_gap"), mean(on, "static_gap")
            off_response, on_response = mean(off, "response_gap"), mean(on, "response_gap")
            output.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "family": family(condition),
                    "n_images": len(off),
                    "off_static_gap": off_static,
                    "on64_static_gap": on_static,
                    "delta_static_on64_minus_off": on_static - off_static,
                    "off_response_gap": off_response,
                    "on64_response_gap": on_response,
                    "delta_response_on64_minus_off": on_response - off_response,
                    "c_gap": (on_response - off_response) - (on_static - off_static),
                    "off_detection_loss_increase": mean(off, "detection_loss_increase"),
                    "on64_detection_loss_increase": mean(on, "detection_loss_increase"),
                    "delta_detection_loss_increase": mean(on, "detection_loss_increase")
                    - mean(off, "detection_loss_increase"),
                    "off_confidence_drop": mean(off, "confidence_drop"),
                    "on64_confidence_drop": mean(on, "confidence_drop"),
                    "delta_confidence_drop": mean(on, "confidence_drop") - mean(off, "confidence_drop"),
                }
            )
    return output


def write_csv(rows: list[dict[str, Any]]) -> None:
    """Write all 24 paired seed-condition summaries."""
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    """Format a signed result compactly."""
    return f"{value:+.6f}"


def write_report(payload: dict[str, Any], condition_rows: list[dict[str, Any]]) -> None:
    """Write the frozen decision in language that preserves all evidence boundaries."""
    decisions = payload["decisions"]
    h1, h2, h3 = decisions["h1"], decisions["h2"], decisions["h3"]
    response_delta = statistics.fmean(float(row["delta_response_on64_minus_off"]) for row in condition_rows)
    per_seed_response = {
        seed: statistics.fmean(
            float(row["delta_response_on64_minus_off"]) for row in condition_rows if row["seed"] == seed
        )
        for seed in sorted({row["seed"] for row in condition_rows})
    }
    lines = [
        "# DINOv3-S P2-03 Static–Response Gap 结果",
        "",
        "正式判定：**H1=Support，H2=Support，H3=Inconclusive**。预注册的 Response-Field 机制准入 gate 已通过，",
        "但本 probe 不授权任何新训练；后续训练仍需另立协议并获得明确批准。",
        "",
        "## H1：clean static alignment",
        "",
        (
            f"- pooled `ON64−OFF` static gap：`{fmt(h1['pooled']['estimate'])}`，95% image-cluster CI "
            f"`[{h1['pooled']['ci95'][0]:.6f}, {h1['pooled']['ci95'][1]:.6f}]`；判定 **Support**。"
        ),
        "- 三个 seed 的点估计依次为 "
        + " / ".join(f"`{fmt(item['estimate'])}`" for item in h1["by_seed"])
        + "；前两个 seed 的 CI 全低于 0，第三个跨 0。",
        "",
        "这说明原 P1 static KD 确实使 projection-free clean P4 spatial relation 更接近 Teacher；并非只有 projector objective 下降。",
        "",
        "## H2：static 改善是否迁移到 response",
        "",
        (
            f"- pooled `C_gap=Δresponse−Δstatic`：`{fmt(h2['pooled']['estimate'])}`，95% image-cluster CI "
            f"`[{h2['pooled']['ci95'][0]:.6f}, {h2['pooled']['ci95'][1]:.6f}]`；判定 **Support**。"
        ),
        "- 三个 seed 的 `C_gap` 点估计依次为 "
        + " / ".join(f"`{fmt(item['estimate'])}`" for item in h2["by_seed"])
        + "；前两个 seed 的 CI 全高于 0，第三个跨 0。",
        f"- 仅作描述的 pooled `Δresponse=ON64−OFF` 点估计为 `{fmt(response_delta)}`；三个 seed 分别为 "
        + " / ".join(f"`{fmt(value)}`" for value in per_seed_response.values())
        + "。该点估计表示 response gap 没有随 static gap 一起下降，反而略增；它没有替代预注册的 `C_gap` 判据。",
        "",
        "因此当前证据支持更严格的表述：**Student 在 clean 点上更接近 Teacher，但这种改善没有迁移到冻结扰动下的局部响应。**",
        "",
        "## H3：response gap 与检测退化的关联",
        "",
        (
            f"- `mean_z(rho_response)`：`{h3['mean_z_response']:+.6f}`，95% CI "
            f"`[{h3['mean_z_response_ci95'][0]:.6f}, {h3['mean_z_response_ci95'][1]:.6f}]`。"
        ),
        (
            f"- `Delta_rho`：`{h3['delta_rho_z']:+.6f}`，95% CI "
            f"`[{h3['delta_rho_z_ci95'][0]:.6f}, {h3['delta_rho_z_ci95'][1]:.6f}]`；"
            "6 个 seed-arm 中 4 个点估计为正。"
        ),
        "- 两个正式 CI 均跨 0，因此判定 **Inconclusive**。不能声称 response gap 导致检测退化，也不能声称它比 static gap 具有稳定更强的解释力。",
        "",
        "## 完整性与边界",
        "",
        "- 3 seeds × 2 arms × 128 images × 8 conditions = **6,144** 条唯一 paired rows；48 个 cell 均为 128 图。",
        "- Bootstrap 单位是 image；同图像的 seed/arm/condition 观测一起重采样。",
        "- 六个 EMA checkpoint 的 `state_dict` SHA-256 运行前后完全一致；训练次数和 optimizer steps 均为 0。",
        "- 24 个 seed-condition 配对结果全部保存在 `d2_v3_p2_response_gap_condition_summary.csv`；单条件结果只作 secondary breakdown。",
        "- P1 No-Go、P2-01 Inconclusive 与 P2-02 No-Support 均保持不变。",
        "- Gate 通过只允许提出新的 Response-Field 训练协议，不等于该训练方法已有效，更不构成因果证明。",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Verify immutable inputs and write secondary/publication-facing artifacts."""
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    if sha256(RAW) != payload["raw"]["sha256"]:
        raise RuntimeError("formal raw SHA-256 mismatch")
    with RAW.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6144:
        raise RuntimeError(f"expected 6144 raw rows, found {len(rows)}")
    condition_rows = build_condition_rows(rows)
    if len(condition_rows) != 24:
        raise RuntimeError(f"expected 24 seed-condition rows, found {len(condition_rows)}")
    write_csv(condition_rows)
    write_report(payload, condition_rows)
    manifest = {
        "source_result_sha256": sha256(RESULT),
        "source_raw_sha256": sha256(RAW),
        "summarizer_sha256": sha256(Path(__file__)),
        "condition_summary": {"path": SUMMARY_CSV.name, "rows": len(condition_rows), "sha256": sha256(SUMMARY_CSV)},
        "report": {"path": REPORT.name, "sha256": sha256(REPORT)},
        "hypothesis_decisions_unchanged": True,
        "secondary_breakdown_does_not_override_primary": True,
    }
    PUBLICATION_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
