# D2 DINOv3-S P1 标定扩展协议

## 原因与边界

首轮冻结候选 `0.01/0.025/0.05/0.10` 均未进入 Foundation/task ratio `[0.03,0.06]`，最大候选 0.10 得到 `0.0263252`。四个候选的机制恒等式均通过，ratio 对 weight 近似线性，且全程没有 validation AP。

为避免无限搜索，本 correction 只增加**一个**由训练信号外推的候选：`0.15`。按 0.10 的观测线性外推，预期 ratio 约 `0.03949`。目标区间、seed、初始化、数据、1 epoch、`val=false`、loss 形式及最小入区间规则均不变。

## 冻结决策

- 若 0.15 ratio 在 `[0.03,0.06]` 且机制恒等式通过：冻结 `foundation_loss_weight=0.15`，生成正式 ON 配置并做 pair audit。
- 若失败：停止正式 P1，不能继续增加候选，需导师决定是否修改 training-signal 目标或 loss 设计。
- 本扩展不产生检测或 KD efficacy 结论，不运行 validation。

## 结果（2026-09-01）

0.15 得到 Foundation/task ratio `0.0392177`，机制恒等式误差为 0，满足冻结规则。正式 P1 weight 因此锁定为 `0.15`；这仍不是检测或 KD efficacy 结论。
