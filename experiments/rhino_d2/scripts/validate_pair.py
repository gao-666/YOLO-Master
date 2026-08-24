#!/usr/bin/env python3
"""Fail closed when the D2 on/off configs contain uncontrolled differences."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OFF_PATH = EXPERIMENT_ROOT / "configs" / "d2_off.yaml"
ON_PATH = EXPERIMENT_ROOT / "configs" / "d2_on.yaml"
OUTPUT_PATH = EXPERIMENT_ROOT / "results" / "d2_config_pair_validation.json"
# 只有这些字段允许在关闭/开启蒸馏的配置之间发生变化。
ALLOWED_DIFFERENCES = {
    "foundation_enabled",
    "foundation_loss_weight",
    "foundation_model",
    "foundation_revision",
    "foundation_teacher",
    "name",
}


def _load(path: Path) -> dict:
    """读取一份配对实验配置并确认其根节点是 YAML mapping。

    统一的配置类型检查能在消融实验启动前发现格式问题，避免把无效配置
    产生的指标误认为蒸馏效果。
    """
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def _sha256(path: Path) -> str:
    """计算配置文件哈希，用于记录本次比较的确切输入版本。

    哈希不提升模型指标，但能在结果归档后确认 on/off 对照使用的文件没有
    被替换，从而提高实验结论的可追溯性。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_configs(off: dict, on: dict) -> dict:
    """逐键比较 D2 off/on 配置并返回可审计的差异报告。

    函数同时报告实际差异、未授权差异和缺失的预期差异。只允许蒸馏相关
    字段变化，可以把后续 mAP、召回率或小目标指标差异更可信地归因于
    foundation distillation，而不是数据、模型或优化器等混杂变量。它不
    直接提高指标，却能提高指标结论的有效性。
    """
    all_keys = sorted(set(off) | set(on))
    differences = {key: {"off": off.get(key), "on": on.get(key)} for key in all_keys if off.get(key) != on.get(key)}
    unexpected = sorted(set(differences) - ALLOWED_DIFFERENCES)
    missing = sorted(ALLOWED_DIFFERENCES - set(differences))
    return {
        "status": "passed" if not unexpected and not missing else "failed",
        "allowed_differences": sorted(ALLOWED_DIFFERENCES),
        "observed_differences": differences,
        "unexpected_differences": unexpected,
        "missing_expected_differences": missing,
    }


def main() -> None:
    """验证并持久化 D2 on/off 配置的消融实验契约。

    主流程读取两份 YAML、执行严格比较、记录文件哈希并写出 JSON。发现未
    授权差异或缺少预期差异时以失败退出，阻止不可靠的对照实验继续进行，
    从实验设计层面保护最终指标结论的可信度。
    """
    result = compare_configs(_load(OFF_PATH), _load(ON_PATH))
    result["files"] = {
        "off": {"path": str(OFF_PATH.relative_to(EXPERIMENT_ROOT.parent.parent)), "sha256": _sha256(OFF_PATH)},
        "on": {"path": str(ON_PATH.relative_to(EXPERIMENT_ROOT.parent.parent)), "sha256": _sha256(ON_PATH)},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 持久化哈希和差异明细，方便审计本次实际比较的文件。
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
