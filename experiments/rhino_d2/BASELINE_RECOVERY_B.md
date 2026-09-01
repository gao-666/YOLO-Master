# D2 Baseline Recovery Candidate B：预训练初始化通过工程门

## 结论

2026-09-01 的 Candidate B 完成 50/50 epoch，并通过全部预注册 baseline 工程门。相比 Candidate A，唯一研究变量是 Student 初始化：Student 架构、固定 COCO mini、预算、优化器、seed、增广、验证和 Foundation 关闭状态均不变。

这支持“当前预算下，YOLO-Master-N scratch 初始化是 baseline 任务学习不足的重要原因”。它只解锁正式 DINOv3-S P1 protocol 的冻结，不证明 DINOv3 或 KD 有效。

## 初始化资格

Student 仍为 `yolo26-master-n.yaml`，没有替换成上游模型。官方 Ultralytics YOLO26n 权重按同名同 shape 迁移：目标参数覆盖率 `41.41%`、源参数覆盖率 `82.35%`，stem、共享深层 backbone 和完整检测 head 均 100% 迁移；MoE 专属参数保留 Student 初始化。权重 SHA-256 为 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`，资格结果见 [`results/d2_v3_student_init_audit.json`](results/d2_v3_student_init_audit.json)。

## 预注册门与观察

| 检查项 | 门槛 | 观察值 | 结论 |
| --- | --- | --- | --- |
| 训练完成 | 50 epoch | 50 | 通过 |
| 后 10 epoch mAP50-95 中位数 | `>=0.01` | `0.048125` | 通过 |
| 后 10 epoch Precision 中位数 | `>0` | `0.26385` | 通过 |
| 后 10 epoch Recall 中位数 | `>0` | `0.10655` | 通过 |
| 最终/初始检测 loss | `<=0.90` | `0.45103`（`11.05673→4.98691`） | 通过 |

训练返回码为 0；实验输入绑定 commit `a967a830330169fbe63ec868527e31fe9888b2e6`，`experiment_inputs_dirty=false`，53 个 checkpoint 已登记。判定不使用 best epoch。

## 下一步边界

正式 P1 必须让 OFF 与 ON 从同一份冻结 Student 初始化开始，并保持同数据、预算、seed、增广和验证。DINOv3-S KD weight 只能根据 training signal 标定，不能看 validation AP。三 seed 配对完成前不启动 DINOv3-L、不增加 multi-stage 或新 loss，也不修改 `0.003 + paired 95% CI` 判读线。

## 可复现证据

- 判定：[`results/d2_v3_baseline_recovery_b.json`](results/d2_v3_baseline_recovery_b.json)
- 逐 epoch：[`results/d2_v3_baseline_recovery_b_s20260824.csv`](results/d2_v3_baseline_recovery_b_s20260824.csv)
- resolved args：[`results/d2_v3_baseline_recovery_b_s20260824.args.yaml`](results/d2_v3_baseline_recovery_b_s20260824.args.yaml)
- 完整日志：[`results/d2_v3_baseline_recovery_b_s20260824.log`](results/d2_v3_baseline_recovery_b_s20260824.log)
- 运行与 checkpoint 哈希：[`results/d2_v3_baseline_recovery_b_s20260824.manifest.json`](results/d2_v3_baseline_recovery_b_s20260824.manifest.json)
