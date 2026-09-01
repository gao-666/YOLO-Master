# D2 DINOv3-S 正式 P1：第一组配对结果

## 结论边界

截至 2026-09-01，正式 P1 已完成 seed `20260824` 的同预算 OFF/ON 配对。该结果是**方向性单 seed 证据**，不是 Go/No-Go 结论；还需完成 `20260825`、`20260826` 才能计算预注册的 paired 95% CI。

## 冻结条件

- Student、预训练初始化、数据、50 epoch、batch、imgsz、优化器、增强和 seed 全部相同。
- ON 只启用 DINOv3-S P4 单 stage cosine 蒸馏，`foundation_loss_weight=0.15`。
- 每个 arm 的正式指标是第 41–50 轮 `mAP50-95` 中位数，不使用 best epoch。
- 两个 arm 均从 source commit `323ae1f8490b17737bf0fb62deae285025b2ffd6` 运行，experiment inputs clean，退出码均为 0。

## 第一组数字

| seed | OFF 末 10 轮中位数 | ON 末 10 轮中位数 | ON−OFF |
| --- | ---: | ---: | ---: |
| 20260824 | 0.048125 | 0.049465 | +0.001340 |

单 seed 方向为正，但绝对差值小于预注册尺度 `0.003`。这**不能**直接判为 No-Go，因为单个差值不能估计 paired 95% CI。

## 蒸馏机制证据

- OFF 的 foundation loss 全程为 0。
- ON 的 foundation loss 从 `0.597270` 降至 `0.190722`，证明辅助损失真实进入训练并可优化。
- ON 最后 10 轮的 foundation/task ratio 中位数为 `0.026109`。末 10 轮关闭 mosaic 后输入分布变化，ratio 可与 1 epoch 标定值不同；OFF/ON 的 `close_mosaic=10` 相同，不构成配对混杂。

## 当前决定

状态：`pending_more_seeds`。

下一步只运行相同冻结协议的 seed `20260825`、`20260826`。三组齐全后计算 mean Δ、sample SD 和 paired 95% CI，再原样应用 Go/No-Go 判读线。此期间不扩 DINOv3-L、不改 align_dim、不加 multi-stage 或新 loss。

## 证据入口

- 机器可读汇总：[`results/d2_v3_p1_first_pair.json`](results/d2_v3_p1_first_pair.json)
- 单行配对表：[`results/d2_v3_p1_first_pair.csv`](results/d2_v3_p1_first_pair.csv)
- 完整曲线：[`results/d2_v3_p1_first_pair.png`](results/d2_v3_p1_first_pair.png)
- OFF/ON 的 CSV、resolved args、完整日志和 runtime manifest 均以 `results/d2_v3_p1_{off,on}_s20260824.*` 固化并带 SHA-256。
