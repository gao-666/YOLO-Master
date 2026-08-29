# D2 实验修正：从“证据齐全”转为“排除竞争解释”

## 修正后的结论

原始 `weight=0.05` 与校准后的 `weight=0.10` 在同一 COCO128、10 epoch、scratch 协议下都满足预注册 no-go。权重翻倍后 foundation/task ratio 从约 `1.7%` 提高到三 seed 平均约 `3.75%`，但 mean ΔmAP50-95 仅从 `+0.0000233` 变为 `+0.0001267`，校准实验 95% CI 仍为 `[-0.0005717, 0.0008250]` 并包含 0。

因此可以更严格地说：**当前 no-go 不太可能只由 0.05 权重过小造成；但由于 absolute mAP 接近 0，仍不能外推为 Foundation 特征没有价值。**

## 问题一：配置相同，实际运行也相同吗？

事前的 `validate_pair.py` 只能证明 YAML 意图。新增 `audit_resolved_args.py` 比较三个 seed 训练完成后的 `args.yaml`：

- model、data、epochs、batch、imgsz、seed、pretrained 完全相同；
- 声明 optimizer 与 `effective_optimizer=SGD` 相同；
- 五个实际 parameter-group learning rates 相同；
- 只有 Foundation 开关、教师身份、权重以及输出路径不同。

原始和校准实验的审计均通过。证据分别为 [`results/d2_p1_resolved_args_audit.json`](results/d2_p1_resolved_args_audit.json) 与 [`results/d2_p1_corrected_resolved_args_audit.json`](results/d2_p1_corrected_resolved_args_audit.json)。

## 问题二：KD 是否弱到几乎没参与优化？

权重选择不使用 validation AP，避免把 COCO128 验证集调参。固定 seed、初始化和一个完整训练 epoch，仅观察训练信号：

| Weight | Foundation/task ratio | Foundation/mixture ratio | 机制恒等式 |
| ---: | ---: | ---: | --- |
| 0.01 | 0.350% | 1.82% | 通过 |
| 0.05 | 1.742% | 9.08% | 通过 |
| 0.10 | 3.501% | 18.20% | 通过 |
| 0.15 | 5.243% | 27.32% | 通过 |

事前规则固定为：选择进入 3%–6% foundation/task ratio 的最小权重，同时要求

`(cosine_raw + relational_raw) × weight × batch_size = foundation_loss`

误差小于 `1e-5`。因此选择 `0.10`，不是根据最好 mAP 选择。完整证据见 [`results/d2_p1_weight_calibration.json`](results/d2_p1_weight_calibration.json)。

## 问题三：提高 KD 量级会改变 no-go 吗？

校准实验仅把 treatment 权重 `0.05→0.10`，复用已通过 resolved-args 审计的 baseline：

| Seed | off | calibrated on | Δ |
| ---: | ---: | ---: | ---: |
| 20260824 | 0.00002 | 0.00001 | -0.00001 |
| 20260825 | 0.00015 | 0.00009 | -0.00006 |
| 20260826 | 0.00002 | 0.00047 | +0.00045 |

- Mean Δ：`+0.0001267`
- Sample SD：`0.0002811`
- Paired 95% t CI：`[-0.0005717, 0.0008250]`
- 判定：`|mean Δ| < 0.003` 且 CI 含 0，仍为 **no-go**。

机器可核验结果见 [`results/d2_p1_corrected_results.json`](results/d2_p1_corrected_results.json) 与 [`results/d2_p1_corrected_results.png`](results/d2_p1_corrected_results.png)。

## 竞争解释更新

| 解释 | 修正前 | 修正后 |
| --- | --- | --- |
| B：KD 权重太小 | 无法排除 | 明显削弱：权重翻倍、训练占比进入预设区间后仍 no-go |
| C：KD 被 mixture auxiliary 淹没 | 可能 | 削弱但未完全排除：Foundation/mixture ratio 已提高到约 18%–27% |
| D：10 epoch/scratch/COCO128 欠拟合 | 很强 | 仍是首要解释：所有 absolute mAP 近 0 |
| 静态对齐不等于任务效用 | 未检验 | 保留为下一阶段机制假设 |

## 下一刀实验

暂不直接实现 Response-Field Loss。已有 detector 几乎不会检测时，ResponseGap 与 robustness drop 的相关性没有可靠任务尺度。下一步先用单变量方式获得有解释力的 baseline（增加训练预算或采用许可明确的学生预训练权重）；判读线不变。只有 baseline mAP 脱离近零区间后，再执行固定 brightness/blur/occlusion/scale/translation 的 Counterfactual Response Probe：

1. 先检验 static alignment 与 response gap 是否脱钩；
2. 再检验 response gap 是否预测 perturbation mAP drop；
3. 只有两个现象成立，才实现 matched-compute Response-Field Distillation。

这条停止规则避免把“有趣隐喻”直接缝成新 loss，也避免在无任务信号时制造漂亮但不可解释的相关性。
