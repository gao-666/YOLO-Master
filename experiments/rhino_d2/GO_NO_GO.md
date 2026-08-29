# D2 P1 Go / No-Go 建议书

## 决策

对当前固定协议给出 **no-go**：不把 DINOv2-small→YOLO-Master-N 的 P4 hybrid 蒸馏作为“已验证涨点方案”继续扩张到多 stage，也不据此申请合入默认训练配置。

该结论仅覆盖 COCO128、从零初始化、10 epoch、imgsz 256、batch 4、KD 权重 0.05、align_dim 64 的实验协议；它不等于 Foundation 蒸馏普遍无效。

## 预注册规则与证据

- Go：mean ΔmAP50-95 ≥ `0.003`，且 paired 95% CI 不含 0。
- No-go：`|mean ΔmAP50-95| < 0.003`，且 paired 95% CI 含 0。
- 三个 paired delta：`+0.00021 / -0.00013 / -0.00001`。
- Mean delta：`+0.0000233`；sample SD：`0.0001724`。
- Paired 95% t CI：`[-0.0004050, 0.0004517]`，包含 0。

因此结果满足 no-go 的两个条件，且判读线在观察结果后未修改。机器可核验数据见 [`results/d2_p1_three_seed_results.json`](results/d2_p1_three_seed_results.json)。

## 保留与停止

保留：P4 单 stage 原型、DINOv2 许可与 revision 锁定、独立 projector/loss、配置差异审计、逐 epoch checkpoint、日志/结果哈希和回归测试。它们证明训练链完整且可继续用于诊断。

停止：当前阶段不增加 multi-stage、MoE 路由蒸馏或更多 loss 拼接；单 seed cosine-only 的 `+0.00015` 方向性差值不足以证明 relational 机制有效。

## 下一步建议

1. 先处理协议欠拟合风险 H6：增加训练预算或使用 COCO mini/中等集，使 baseline 获得可解释的 absolute mAP。
2. 若预算只能支持一次单变量诊断，优先将 KD 权重从 `0.05` 降到 `0.01` 检验 H4；保持其他字段不变。
3. 其次检验 align_dim `64→128` 的 H3。不得同时改权重和维度，否则无法归因。
4. 只有新协议重新达到 go 条件，才扩展多 stage；若仍 no-go，提交负结果与容量/维度/优化/数据归因，不移动阈值。

## 风险说明

当前 absolute mAP 极低，主要风险是 128 图、从零初始化与 10 epoch 导致欠拟合，paired best 指标只能回答“该固定短预算下有没有可行动信号”。另外，DINOv2 patch-14 grid 到 YOLO P4 stride-16 grid 的插值可能削弱空间监督；在获得许可明确的替代教师前，不通过未经授权权重绕开该风险。
