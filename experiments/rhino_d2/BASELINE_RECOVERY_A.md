# D2 Baseline Recovery Candidate A：数据规模单变量负结果

## 结论

2026-09-01 的 Candidate A 完成 50/50 epoch，但**未通过预注册 baseline 工程门**。正式 DINOv3 P1 继续锁定；本实验没有启用 Foundation Teacher，因此不能据此判断 DINOv3 或知识蒸馏是否有效。

相比此前 `COCO128 + scratch`，固定 COCO mini（train=2048、val=512）使检测 loss 明显下降，且 Precision、Recall 进入非零区间；但最后 10 epoch 的 mAP50-95 中位数仅为 `0.00546`，没有达到预注册的 `0.01`。因此“仅扩大数据即可获得可解释 baseline”的 Candidate A 假设未通过。

## 唯一变量

Candidate A 与失败的 `d2_v3_off_sanity.yaml` 仅有三项配置差异：`data`、`name`、`project`。其中只有 `data` 是研究变量，后两项只隔离输出目录。Student、scratch 初始化、50 epoch、batch、imgsz、优化器、学习率、seed、增广和 Foundation 关闭状态均保持不变。

## 预注册门与观察

| 检查项 | 预注册门槛 | 观察值 | 结论 |
| --- | --- | --- | --- |
| 训练完成 | 50 epoch | 50 | 通过 |
| 后 10 epoch mAP50-95 中位数 | `>=0.01` | `0.00546` | **失败** |
| 后 10 epoch Precision 中位数 | `>0` | `0.245115` | 通过 |
| 后 10 epoch Recall 中位数 | `>0` | `0.01684` | 通过 |
| 最终/初始检测 loss | `<=0.90` | `0.70445`（`9.36099→6.59439`） | 通过 |

门槛不使用 best epoch，也没有根据结果修改。训练返回码为 0；实验输入绑定到 commit `3d40f3b3f775f1985d9ff58e3de643a9a14eb6d8`，运行前 `experiment_inputs_dirty=false`，53 个 checkpoint 已登记在运行清单中。

## 诊断与下一步

Candidate A 削弱了“COCO128 太小是唯一原因”的解释，但 loss、Precision 与 Recall 的恢复表明数据规模确实改善了任务学习。当前证据仍不能区分 scratch 初始化预算不足、Student/MoE 优化困难、低分辨率限制或评测信号偏弱。

按照冻结的停止规则，下一步是 Candidate B：在完全相同的数据、训练预算和工程门下，只把 scratch 改为许可、来源、哈希与迁移覆盖率明确的 Student 预训练初始化。若没有与当前 Student 架构兼容且迁移覆盖率足够的资产，Candidate B 必须判为不可执行，不能偷偷换模型结构；若 B 执行后仍失败，则停止 KD efficacy，转为训练管线正控。

在 baseline 通过前，禁止正式 ON/OFF、基于 validation AP 选择 KD weight、DINOv3-L capacity、multi-stage 或新增 loss。

## 可复现证据

- 判定：[`results/d2_v3_baseline_recovery_a.json`](results/d2_v3_baseline_recovery_a.json)
- 逐 epoch：[`results/d2_v3_baseline_recovery_a_s20260824.csv`](results/d2_v3_baseline_recovery_a_s20260824.csv)
- resolved args：[`results/d2_v3_baseline_recovery_a_s20260824.args.yaml`](results/d2_v3_baseline_recovery_a_s20260824.args.yaml)
- 完整 UTF-8 日志：[`results/d2_v3_baseline_recovery_a_s20260824.log`](results/d2_v3_baseline_recovery_a_s20260824.log)
- 运行/commit/checkpoint 哈希：[`results/d2_v3_baseline_recovery_a_s20260824.manifest.json`](results/d2_v3_baseline_recovery_a_s20260824.manifest.json)
- 数据选择与 payload 哈希：[`datasets/d2_coco_mini_2048_seed20260901/manifest.json`](datasets/d2_coco_mini_2048_seed20260901/manifest.json)

复现命令：

```powershell
conda activate yolo-master-d2
$env:YOLO_CONFIG_DIR="E:\2026YOLO\YOLO-Master\runs\rhino_d2\config"
$env:PYTHONUTF8="1"
python experiments/rhino_d2/scripts/run_p1.py --arms v3-baseline-recovery-a --seed 20260824 --project runs/rhino_d2/v3_baseline_recovery_a
python experiments/rhino_d2/scripts/assess_v3_baseline_sanity.py --run-dir runs/rhino_d2/v3_baseline_recovery_a/v3-baseline-recovery-a-s20260824 --output-name d2_v3_baseline_recovery_a.json --failure-next-action run_pretrained_candidate_b_on_same_dataset
```

判定脚本在 gate 失败时返回非零状态，这是 fail-closed 行为，不代表训练进程异常。
