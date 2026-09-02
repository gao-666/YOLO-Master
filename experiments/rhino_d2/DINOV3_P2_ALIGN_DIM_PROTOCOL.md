# D2 DINOv3-S P2-02 对齐维度诊断协议

## 研究问题与冻结边界

P1 的 `align_dim=64` No-Go 永久保留；P2-01 亦冻结为：未发现稳定负向梯度冲突是 P1 No-Go 主要机制的证据，
更明显的观测是加权 KD 梯度相对检测梯度较弱且近乎正交。P2-02 不追溯修改这两个结论。

P2-02 只检验：

> `align_dim=64` 是否构成 Foundation 信息传递的投影瓶颈？

扩大维度不预设 KD 梯度一定增大。更宽的 projector 可能提高信息保真，也可能只在 projector 内吸收对齐任务而不改善 Student 检测表征。

## 唯一操纵变量

正式 P1 ON64 与 P2-02 ON128 只允许以下差异：

- `foundation_align_dim: 64 -> 128`；
- `project`、`name`、运行生成的 `save_dir` 仅作为输出身份变化。

以下条件全部保持不变：YOLO-Master-N、同一预训练 Student 资产、COCO mini `2048/512`、50 epoch、imgsz 256、
batch 4、SGD、DINOv3-S/16、BF16、P4、cosine-only、KD weight 0.15、恒定 schedule、seed
`20260824/20260825/20260826`。不重新标定 weight，不换 Teacher，不增加 loss、stage、epoch 或数据。

只新增三次 ON128 训练；复用冻结的三组 OFF 与 ON64，不重新运行 reference。

## 训练前 fail-closed 审计

训练前必须自动通过：

1. ON128 与 ON64 配置差异严格等于 `align_dim` 与输出身份；
2. Foundation/检测相关源码与正式 P1 训练时版本一致；
3. 数据 YAML、train/val 清单和实际图像/标签 inventory hash 一致；
4. Student 初始化资产 SHA-256 仍为 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`；
5. DINOv3-S 五个本地资产哈希与冻结 manifest 一致；
6. Python、PyTorch、CUDA、GPU 和关键包版本无实质变化；
7. 预注册协议、配置、runner、validator、summarizer、测试和相关训练源码均为 clean tracked state。

任一检查失败则禁止训练，不允许通过额外 CLI override 绕开。

## 主要与次级比较

每个 arm/seed 的正式指标为 epoch 41--50 的 `metrics/mAP50-95(B)` 中位数，不使用 best epoch。

主要机制比较：

```text
delta_dim,i = ON128_i - ON64_i
```

- Strong support：paired mean `>= +0.003` 且 paired 95% t CI 下界 `> 0`；
- No support：`|paired mean| < 0.003` 且 CI 包含 0；
- 其它：Inconclusive。

次级 efficacy 比较：

```text
delta_off,i = ON128_i - OFF_i
```

采用相同三级规则。只有次级比较达到 Strong support，才可表述为“在 P2 新协议下恢复了可检测的 Foundation KD 收益”；
不得改写 P1 ON64 No-Go。

如果维度比较为 No support，不继续搜索 256/512；如果 Inconclusive，只能补 seed，不得通过继续扩维追结果。

## 机制复测

三次训练完成后，使用 P2-01 已冻结的同一 `diagnostic_train64.txt`（SHA-256
`1ad936698234fd07651993dbdefe7a98ffaf74861432a103b6e397bb45b9b676`），在 ON128 的 epoch9/24/49 EMA checkpoint
复测 `cos(g_task,g_KD)`、`||g_KD||/||g_task||` 和负余弦比例，并与 ON64 做按图像、seed、时点配对比较。

该复测解释“维度变化是否改变监督方向或强度”，不能替代 mAP 判读，也不能形成新的 weight calibration。

