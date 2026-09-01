# D2 Foundation 蒸馏绿地：8.24 准入包

本目录把仓库已有的 Foundation Distillation alpha 收敛为一个可审计的 D2 最小实验。它不声称新增整套蒸馏框架，也不在 smoke 结果上声称精度提升。

8.24 在线表格的简版状态和审核备注见 [`submission_824.md`](submission_824.md)。“单 stage 原型 + loss 曲线 + 设计依据”的集中验收页见 [`P0_EVIDENCE.md`](P0_EVIDENCE.md)。

8.25–9.7 的首轮同预算数字、关键消融、竞争性解释和中期讲稿见 [`P1_MIDTERM.md`](P1_MIDTERM.md)；权重标定、事后真实参数审计和修正实验见 [`EXPERIMENT_CORRECTION.md`](EXPERIMENT_CORRECTION.md)；正式判读见 [`GO_NO_GO.md`](GO_NO_GO.md)；评审四节结构见 [`PR_DRAFT.md`](PR_DRAFT.md)。

2026-08-31 起的 DINOv3 复验是独立 protocol，不覆盖上述 DINOv2 历史证据。研究红线和 gate 顺序见 [`DINOv3_PROTOCOL.md`](DINOv3_PROTOCOL.md)，S/L BF16 与 P0 证据见 [`DINOv3_P0_EVIDENCE.md`](DINOv3_P0_EVIDENCE.md)。OFF-only 50 epoch baseline sanity 的失败结果与停止决定见 [`DINOv3_BASELINE_SANITY.md`](DINOv3_BASELINE_SANITY.md)。9 月 1 日的数据规模单变量诊断与两步停止规则见 [`BASELINE_RECOVERY_PROTOCOL.md`](BASELINE_RECOVERY_PROTOCOL.md)；Candidate A 的负结果见 [`BASELINE_RECOVERY_A.md`](BASELINE_RECOVERY_A.md)，Candidate B 的通过结果见 [`BASELINE_RECOVERY_B.md`](BASELINE_RECOVERY_B.md)。baseline admission 已通过，下一步冻结正式 DINOv3-S P1 protocol 与 train-only KD weight。

## 准入状态

| 环境安装 | 基线/最小任务 | 复现命令 | 配置文件 | 完整日志 | 结果证据 | 设计说明 | 风险与降级 | 代码/方案链接 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Conda `yolo-master-d2`；Python 3.11；PyTorch CUDA 12.8；`pip install -e ".[foundation,dev]"` | YOLO-Master-N 的 P4 与冻结 Foundation Teacher 特征对齐；真实 YOLO detection loss 实现 + 确定性合成 target 的 fixed-batch smoke，KD 与检测 loss 共同反传 | 见“复现命令” | [`configs/d2_p0.yaml`](configs/d2_p0.yaml)、[`configs/d2_off.yaml`](configs/d2_off.yaml)、[`configs/d2_on.yaml`](configs/d2_on.yaml)、[`configs/d2_smoke.yaml`](configs/d2_smoke.yaml) | [`results/d2_p0_train_smoke.json`](results/d2_p0_train_smoke.json)（检测实现+合成 target+KD）；[`results/d2_alignment_smoke.json`](results/d2_alignment_smoke.json)（对齐逐 step）；[`results/d2_p0_loss_curves.png`](results/d2_p0_loss_curves.png)（loss 曲线）；[`results/pytest-foundation.xml`](results/pytest-foundation.xml)（93 tests）；[`env/environment.json`](env/environment.json) | 总 loss 分解、学生/投影梯度、教师冻结、教师不在 optimizer、teacher revision/权重 hash、[`results/d2_config_pair_validation.json`](results/d2_config_pair_validation.json) | [`P0_EVIDENCE.md`](P0_EVIDENCE.md)、[`design.md`](design.md)、[`P0_HANDOVER.md`](P0_HANDOVER.md)、[`statistical_distillation_proposal.md`](statistical_distillation_proposal.md) | [`limitations.md`](limitations.md) | 本目录；base/experiment commit 与 dirty 状态见 `env/environment.json` |

## 复现命令

以下命令从仓库根目录执行。Windows 必须使用 `workers=0`。

```powershell
conda activate yolo-master-d2
python experiments/rhino_d2/scripts/record_environment.py
python experiments/rhino_d2/scripts/validate_pair.py
python experiments/rhino_d2/scripts/d2_alignment_smoke.py --config experiments/rhino_d2/configs/d2_smoke.yaml
python experiments/rhino_d2/scripts/d2_p0_train_smoke.py --config experiments/rhino_d2/configs/d2_p0.yaml
python experiments/rhino_d2/scripts/plot_p0_loss_curves.py
python experiments/rhino_d2/scripts/run_p1.py --arms off on --seed 20260825
python experiments/rhino_d2/scripts/run_p1.py --arms off on --seed 20260826
python experiments/rhino_d2/scripts/run_p1.py --arms cosine-only --seed 20260824
python experiments/rhino_d2/scripts/summarize_p1.py
python experiments/rhino_d2/scripts/audit_resolved_args.py
python experiments/rhino_d2/scripts/calibrate_p1_weight.py --weights 0.01,0.05,0.1,0.15
python experiments/rhino_d2/scripts/run_p1.py --arms on-calibrated --seed 20260824 --project runs/rhino_d2/p1_corrected
python experiments/rhino_d2/scripts/run_p1.py --arms on-calibrated --seed 20260825 --project runs/rhino_d2/p1_corrected
python experiments/rhino_d2/scripts/run_p1.py --arms on-calibrated --seed 20260826 --project runs/rhino_d2/p1_corrected
python experiments/rhino_d2/scripts/audit_resolved_args.py --corrected
python experiments/rhino_d2/scripts/summarize_corrected_p1.py
$env:PYTHONUTF8="1"
pytest experiments/rhino_d2/tests/test_admission_contract.py tests/test_foundation_taps.py tests/test_foundation_projectors.py tests/test_foundation_losses.py tests/test_foundation_distill_model.py tests/test_foundation_config.py tests/test_foundation_dinov2.py tests/test_default_config_integrity.py --junitxml=experiments/rhino_d2/results/pytest-foundation.xml -v
```

首次运行真实 DINOv2 smoke 会从 Hugging Face 下载公开教师权重到本目录的 `cache/`；缓存不进入 Git。复跑时追加 `--offline`，强制只使用已锁定的本地 snapshot。

## 同预算 on/off

`d2_off.yaml` 与 `d2_on.yaml` 只允许以下字段不同：`name`、`foundation_enabled`、`foundation_teacher`、`foundation_model`、`foundation_revision`、`foundation_loss_weight`。`validate_pair.py` 会失败关闭（fail closed），防止 epoch、batch、imgsz、seed、模型、数据、优化器或增广发生混杂。

当前 on 配置正式指向 `facebook/dinov2-small` 的锁定 revision。DINOv3 访问申请已被拒绝，因此不再保留为不可执行的主实验；DINOv2 已进入正式构造器，P1 不使用社区镜像或未授权权重。

## 8.24 汇报口径

1. 锁定 YOLO-Master commit、Python/PyTorch/CUDA、学生配置和教师 revision/hash。
2. 展示 YOLO-Master P4 与教师 dense feature 的原始 shape，以及投影后相同 shape。
3. 展示固定单 batch 的 task/KD/total loss，证明 KD 真正进入 total loss，学生和 projector 收到梯度且 KD loss 下降；不声称检测 total loss 单调。
4. 展示 on/off 配置差异审计结果为通过。
5. 明确 smoke 只证明链路，不证明 mAP；P1 采用相同预算、至少 3 seed 的成对实验。

## Go / No-Go 判读线

- P0 go：对齐 shape 合法、教师冻结、学生投影层有梯度、KD loss 有限且下降、配置对照无混杂。
- P1 go：蒸馏相对 baseline 的 `mAP50-95` 平均增益至少 0.003（即 0.3 个百分点），且成对差值的 95% 置信区间不包含 0。
- P1 no-go：`|ΔmAP| < 0.003` 且 95% 置信区间包含 0；保持判读线不变，转入容量/维度/优化/数据诊断。

注意：Ultralytics 的指标通常用 0–1 表示，任务书中的“0.3”按 0.3 个百分点解释为 `0.003`，该口径需在 8.24 与导师确认。

## P1 当前状态（2026-08-29）

Seeds 20260824/25/26 的 off/on 共 6 次训练，以及 seed 20260824 的 cosine-only 消融均已跑完 10 epoch。正式配对统计见 [`results/d2_p1_three_seed_results.json`](results/d2_p1_three_seed_results.json)、[`results/d2_p1_three_seed_results.csv`](results/d2_p1_three_seed_results.csv) 和 [`results/d2_p1_three_seed_results.png`](results/d2_p1_three_seed_results.png)。平均 ΔmAP50-95 为 `+0.0000233`，95% CI `[-0.0004050, 0.0004517]`，满足预注册 no-go 条件。该结论只适用于当前 COCO128、从零初始化、10 epoch 协议，不等于 Foundation 蒸馏在更充分训练下必然无效。

随后完成了不使用 validation AP 的权重标定与修正实验。`weight=0.10` 将 foundation/task ratio 提高到三 seed 平均约 `3.75%`；修正后的 mean Δ=`+0.0001267`、95% CI=`[-0.0005717, 0.0008250]`，仍为 no-go。该结果削弱“权重太小”解释，但 absolute mAP 仍近 0，下一步应先处理训练预算/初始化，而不是直接增加新 loss。修正证据的 13 项契约测试记录见 [`results/pytest-p1-correction.xml`](results/pytest-p1-correction.xml)。
