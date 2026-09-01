# D2 DINOv3-S P1 权重标定：无候选入区间

## 结论

2026-09-01 的 train-only KD weight calibration 完成四个冻结候选，但没有候选进入预注册的 Foundation/task ratio `[0.03, 0.06]`。因此状态为 `no_candidate_in_band`，正式 P1 ON/OFF **未获准启动**。

该结果不是 KD no-go，也没有使用 validation AP。它只说明当前候选上界 0.10 对 DINOv3-S cosine loss 仍略低于目标 training-signal 区间。

## 冻结规则与结果

四次运行使用同一 Student 初始化、同一数据与顺序、同一 seed，均为 1 epoch、`val=false`、`save=false`。选择规则是在 `[3%,6%]` 内选择最小候选，并要求 `cosine_raw × weight × batch = foundation_loss` 恒等式通过。

| weight | Foundation/task ratio | Foundation loss | 恒等式 | 入区间 |
| ---: | ---: | ---: | --- | --- |
| 0.010 | 0.002623（0.262%） | 0.039904 | 通过 | 否 |
| 0.025 | 0.006606（0.661%） | 0.099780 | 通过 | 否 |
| 0.050 | 0.013207（1.321%） | 0.199645 | 通过 | 否 |
| 0.100 | 0.026325（2.633%） | 0.398861 | 通过 | 否 |

实验输入绑定 commit `9c5f63b9d3c47cd4d4ee9fcc0daea27bf2f57c7a`，`experiment_inputs_dirty=false`。仓库整体 dirty 仅来自不在 D2 输入白名单内的用户目录 `experiments/study/`。

## 停止边界

不能把最接近下界的 0.10 事后改判为合格，也不能直接看 mAP 挑权重。正式 OFF/ON、三 seed 和 DINOv3-L 保持停止。

若要继续，必须先把“扩展标定候选”作为新的、显式版本化 protocol correction。根据近似线性 ratio，0.15 预计约为 3.95%，但这只是由训练信号外推的候选设计，不是已完成结果；新 protocol 必须先 commit，再运行，且仍禁止 validation AP。

该 correction 后续已独立预注册并运行：0.15 的 ratio 为 `0.0392177`，通过 unchanged band 与机制恒等式。详见 [`DINOV3_P1_CALIBRATION_EXTENSION_PROTOCOL.md`](DINOV3_P1_CALIBRATION_EXTENSION_PROTOCOL.md) 与 [`results/d2_v3_p1_weight_calibration_extension.json`](results/d2_v3_p1_weight_calibration_extension.json)。

## 证据

- 汇总与选择状态：[`results/d2_v3_p1_weight_calibration.json`](results/d2_v3_p1_weight_calibration.json)
- weight 0.01：[`CSV`](results/d2_v3_calibration_w0p01_s20260824.csv) · [`args`](results/d2_v3_calibration_w0p01_s20260824.args.yaml) · [`完整日志`](results/d2_v3_calibration_w0p01_s20260824.log)
- weight 0.025：[`CSV`](results/d2_v3_calibration_w0p025_s20260824.csv) · [`args`](results/d2_v3_calibration_w0p025_s20260824.args.yaml) · [`完整日志`](results/d2_v3_calibration_w0p025_s20260824.log)
- weight 0.05：[`CSV`](results/d2_v3_calibration_w0p05_s20260824.csv) · [`args`](results/d2_v3_calibration_w0p05_s20260824.args.yaml) · [`完整日志`](results/d2_v3_calibration_w0p05_s20260824.log)
- weight 0.10：[`CSV`](results/d2_v3_calibration_w0p1_s20260824.csv) · [`args`](results/d2_v3_calibration_w0p1_s20260824.args.yaml) · [`完整日志`](results/d2_v3_calibration_w0p1_s20260824.log)
- extension weight 0.15：[`CSV`](results/d2_v3_calibration_w0p15_s20260824.csv) · [`args`](results/d2_v3_calibration_w0p15_s20260824.args.yaml) · [`完整日志`](results/d2_v3_calibration_w0p15_s20260824.log)
