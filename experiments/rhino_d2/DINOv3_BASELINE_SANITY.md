# D2 DINOv3 baseline sanity：负结果与停止决定

## 结论

2026-08-31 的 OFF-only 50 epoch 检测 baseline **未通过预注册工程门**。因此 DINOv3 正式 P1 没有解锁：没有创建正式 ON/OFF 配置，没有做 KD weight 标定，没有启动三 seed 配对训练，也没有启动 ViT-L capacity 实验。

这不是 DINOv3 的 no-go。该训练完全关闭 Foundation Teacher，失败只能说明当前 `COCO128 + scratch + 50 epochs + 256 imgsz` 协议没有产生足以检验蒸馏效果的 baseline 信号。

## 冻结协议与观察

| 检查项 | 预注册门槛 | 观察值 | 结论 |
| --- | --- | --- | --- |
| 训练完成 | 50 epoch | 50 | 通过 |
| 后 10 epoch mAP50-95 中位数 | `>=0.01` | `0.00000` | 失败 |
| 后 10 epoch Precision 中位数 | `>0` | `0.00016` | 通过 |
| 后 10 epoch Recall 中位数 | `>0` | `0.002345` | 通过 |
| 最终检测 loss / 初始检测 loss | `<=0.90` | `1.02247`（`9.28819→9.49694`） | 失败 |

门槛不使用 best epoch，也没有在看到结果后修改。训练返回码为 0；实验输入绑定到 commit `1a6ea0075862f4e8e652726dd145d767f294b19a`，`experiment_inputs_dirty=false`。53 个 checkpoint（best、epoch0–49、last、last_healthy）均登记在运行清单中。

## 诊断边界

现有证据支持“baseline 任务学习不足”，但还不能只凭这一次运行区分下列原因：

1. 从零初始化在 50 epoch/COCO128 下预算不足；
2. 数据规模过小且 train/val 同源，检测指标高度不稳定；
3. 当前 Student/MoE 配置在该低预算协议下优化困难；
4. 256 分辨率削弱小目标学习信号。

不能把失败归因于 DINOv3、投影层、KD loss 或 Teacher capacity，因为这些机制在本次 OFF 运行中均未启用。

## 下一步与红线

下一轮先做一个单变量 baseline 修正，优先采用**许可明确的 Student 预训练初始化**；若该资产不能锁定，再单独增加训练预算。不得同时更换初始化、数据、分辨率和优化器。

修正后仍使用同一 late-window 工程门。只有 baseline 通过，才允许依次执行：冻结正式 P1 协议 → train-only KD ratio 标定 → 配置对审计 → 三 seed `OFF vs DINOv3-S/16`。在 S 主实验完成前不训练 L；不得改 `0.003 + paired 95% CI` 判读线，也不得增加 multi-stage 或新 loss。

## 可复现证据

- 判定 JSON：[`results/d2_v3_baseline_sanity.json`](results/d2_v3_baseline_sanity.json)
- 逐 epoch CSV：[`results/d2_v3_baseline_sanity_s20260824.csv`](results/d2_v3_baseline_sanity_s20260824.csv)
- resolved args：[`results/d2_v3_baseline_sanity_s20260824.args.yaml`](results/d2_v3_baseline_sanity_s20260824.args.yaml)
- 完整 UTF-8 日志：[`results/d2_v3_baseline_sanity_s20260824.log`](results/d2_v3_baseline_sanity_s20260824.log)
- 运行/commit/checkpoint 哈希清单：[`results/d2_v3_baseline_sanity_s20260824.manifest.json`](results/d2_v3_baseline_sanity_s20260824.manifest.json)

复现命令：

```powershell
conda activate yolo-master-d2
$env:YOLO_CONFIG_DIR="E:\2026YOLO\YOLO-Master\runs\rhino_d2\config"
$env:HF_HUB_OFFLINE="1"
$env:PYTHONUTF8="1"
python experiments/rhino_d2/scripts/run_p1.py --arms v3-baseline-sanity --seed 20260824 --project runs/rhino_d2/v3_baseline_sanity
python experiments/rhino_d2/scripts/assess_v3_baseline_sanity.py
```

判定脚本在 gate 失败时返回非零状态，这是 fail-closed 行为，不表示 50 epoch 训练进程异常。
