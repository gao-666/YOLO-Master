#!/usr/bin/env python3
"""Generate auditable D2 P0 loss curves directly from committed smoke JSON evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
ALIGNMENT_PATH = RESULTS_ROOT / "d2_alignment_smoke.json"
P0_PATH = RESULTS_ROOT / "d2_p0_train_smoke.json"
CSV_PATH = RESULTS_ROOT / "d2_p0_loss_curves.csv"
FIGURE_PATH = RESULTS_ROOT / "d2_p0_loss_curves.png"
MANIFEST_PATH = RESULTS_ROOT / "d2_p0_loss_curves_manifest.json"


def _sha256(path: Path) -> str:
    """Return the SHA-256 identity of one evidence artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_passed(path: Path) -> dict[str, Any]:
    """Load one smoke result and reject failed or malformed evidence."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "passed":
        raise ValueError(f"{path.name} must contain a passed smoke result")
    return payload


def _curve_rows(alignment: dict[str, Any], p0: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert source JSON histories into a stable long-form table."""
    rows = [
        {"experiment": "alignment_only", "series": "kd_loss", "step": step, "loss": float(loss)}
        for step, loss in enumerate(alignment["optimization"]["history"])
    ]
    for record in p0["steps"]:
        for source_key, series in (
            ("foundation_loss", "weighted_kd_loss"),
            ("task_loss", "yolo_detection_loss"),
            ("total_loss", "total_loss"),
        ):
            rows.append(
                {
                    "experiment": "integrated_fixed_batch",
                    "series": series,
                    "step": int(record["step"]),
                    "loss": float(record[source_key]),
                }
            )
    return rows


def _write_csv(rows: list[dict[str, Any]]) -> None:
    """Write the exact plotted points for independent inspection."""
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("experiment", "series", "step", "loss"))
        writer.writeheader()
        writer.writerows(rows)


def _plot(alignment: dict[str, Any], p0: dict[str, Any]) -> None:
    """Render alignment-only and integrated fixed-batch curves without smoothing."""
    alignment_history = [float(value) for value in alignment["optimization"]["history"]]
    records = p0["steps"]
    steps = [int(record["step"]) for record in records]
    kd = [float(record["foundation_loss"]) for record in records]
    task = [float(record["task_loss"]) for record in records]
    total = [float(record["total_loss"]) for record in records]

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.subplots_adjust(left=0.07, right=0.93, top=0.83, bottom=0.19, wspace=0.3)
    fig.suptitle("D2 P0 Single-Stage (P4) Distillation — Loss Evidence", fontsize=15, fontweight="bold")

    left = axes[0]
    left.plot(range(len(alignment_history)), alignment_history, color="#1565C0", marker="o", markersize=3)
    left.set_title("A. Alignment-only projector smoke")
    left.set_xlabel("Optimization step")
    left.set_ylabel("Cosine KD loss")
    left.grid(alpha=0.25)
    left.annotate(f"{alignment_history[0]:.5f}", (0, alignment_history[0]), xytext=(7, -14), textcoords="offset points")
    left.annotate(
        f"{alignment_history[-1]:.5f}",
        (len(alignment_history) - 1, alignment_history[-1]),
        xytext=(-48, 10),
        textcoords="offset points",
    )

    right = axes[1]
    right.plot(steps, task, color="#455A64", marker="o", label="YOLO detection loss")
    right.plot(steps, total, color="#EF6C00", marker="s", linestyle="--", label="Total loss")
    right.set_title("B. Integrated fixed-batch smoke")
    right.set_xlabel("Optimizer step")
    right.set_ylabel("Detection / total loss")
    right.set_xticks(steps)
    right.grid(alpha=0.25)
    kd_axis = right.twinx()
    kd_axis.plot(steps, kd, color="#2E7D32", marker="D", linewidth=2.2, label="Weighted KD loss")
    kd_axis.set_ylabel("Weighted KD loss", color="#2E7D32")
    kd_axis.tick_params(axis="y", labelcolor="#2E7D32")
    kd_axis.annotate(f"{kd[0]:.5f}", (steps[0], kd[0]), xytext=(7, 9), textcoords="offset points")
    kd_axis.annotate(f"{kd[-1]:.5f}", (steps[-1], kd[-1]), xytext=(-52, 10), textcoords="offset points")
    handles, labels = right.get_legend_handles_labels()
    kd_handles, kd_labels = kd_axis.get_legend_handles_labels()
    right.legend(handles + kd_handles, labels + kd_labels, loc="best", frameon=False)

    fig.text(
        0.5,
        0.035,
        "Raw points only (no smoothing). Real YOLO loss implementation + deterministic synthetic targets.\n"
        "Fixed-batch gradient-chain evidence, not an accuracy claim.",
        ha="center",
        fontsize=9,
        color="#455A64",
    )
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Generate the curve image, plotted-point CSV, and source/output hash manifest."""
    alignment = _load_passed(ALIGNMENT_PATH)
    p0 = _load_passed(P0_PATH)
    rows = _curve_rows(alignment, p0)
    _write_csv(rows)
    _plot(alignment, p0)
    kd = [float(record["foundation_loss"]) for record in p0["steps"]]
    manifest = {
        "schema_version": 1,
        "claim": "visualization_of_committed_p0_smoke_evidence_no_accuracy_claim",
        "sources": {
            str(ALIGNMENT_PATH.relative_to(EXPERIMENT_ROOT)): _sha256(ALIGNMENT_PATH),
            str(P0_PATH.relative_to(EXPERIMENT_ROOT)): _sha256(P0_PATH),
        },
        "outputs": {
            str(CSV_PATH.relative_to(EXPERIMENT_ROOT)): _sha256(CSV_PATH),
            str(FIGURE_PATH.relative_to(EXPERIMENT_ROOT)): _sha256(FIGURE_PATH),
        },
        "points": len(rows),
        "summary": {
            "alignment_initial": float(alignment["optimization"]["history"][0]),
            "alignment_final": float(alignment["optimization"]["history"][-1]),
            "integrated_kd_initial": kd[0],
            "integrated_kd_final": kd[-1],
            "single_stage": "p4",
            "target_source": p0["data_contract"]["target_source"],
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
