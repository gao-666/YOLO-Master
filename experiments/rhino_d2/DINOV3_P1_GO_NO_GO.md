# D2 DINOv3-S 正式 P1 Go / No-Go 建议书

## 决策

对冻结的 DINOv3-S 正式 P1 协议给出 **No-Go**：不把当前 DINOv3-S→YOLO-Master-N 的 P4 单 stage cosine 蒸馏作为“已验证涨点方案”继续扩展到 DINOv3-L、multi-stage、默认训练配置或新 loss 组合。

该结论只覆盖本次冻结条件：COCO mini train/val=`2048/512`、许可明确的 Student 预训练部分迁移、50 epoch、imgsz 256、batch 4、SGD、DINOv3-S/16 BF16、P4、align_dim 64、cosine-only、KD weight 0.15。它不等于 DINOv3 或 Foundation 蒸馏普遍无效。

## 三组正式结果

每个 arm 均使用第 41–50 轮 mAP50-95 中位数，不选择 best epoch。

| seed | OFF | ON | ON−OFF |
| --- | ---: | ---: | ---: |
| 20260824 | 0.048125 | 0.049465 | +0.001340 |
| 20260825 | 0.054500 | 0.053940 | -0.000560 |
| 20260826 | 0.056960 | 0.049670 | -0.007290 |

统计量：

- paired mean Δ：`-0.002170`
- sample SD：`0.004535`
- standard error：`0.002618`
- paired 95% t CI（df=2）：`[-0.013435, 0.009095]`
- CI 包含 0：是

预注册 No-Go 规则为 `|mean Δ|<0.003` 且 paired 95% CI 包含 0。本结果同时满足两个条件，判读线在观察结果后未修改。

## 机制与工程结论

三组 ON 的 foundation loss 均明显下降，说明教师特征、P4 projector 和 cosine loss 的梯度链真实工作。因此 No-Go 不是“loss 没接上”的工程失败，而是当前协议下没有检测到稳定、可行动的检测收益。

六次正式运行均完成 50 epoch、退出码为 0、experiment inputs clean。三个 seed 的 OFF config hash 恒定、ON config hash 恒定、runner hash 恒定；每次均归档完整日志、resolved args、epoch CSV、runtime manifest 和 checkpoint hashes。

## 停止与后续边界

立即停止：

- 不因 seed24 的正值或 seed26 的负值事后调 KD weight。
- 不直接扩 DINOv3-L、multi-stage、align_dim 128 或新 loss。
- 不改成 best/last mAP 重新判读。

如进入 P2，必须把它版本化为新的机制诊断协议，并一次只检验容量、维度、优化或数据中的一个因素；新实验不得改写本次正式 P1 结论。

## 证据入口

- 机器可读统计：[`results/d2_v3_p1_three_seed_results.json`](results/d2_v3_p1_three_seed_results.json)
- 三 seed 表：[`results/d2_v3_p1_three_seed_results.csv`](results/d2_v3_p1_three_seed_results.csv)
- 配对图：[`results/d2_v3_p1_three_seed_results.png`](results/d2_v3_p1_three_seed_results.png)
- seed26 独立证据：[`results/d2_v3_p1_pair_s20260826.json`](results/d2_v3_p1_pair_s20260826.json)
