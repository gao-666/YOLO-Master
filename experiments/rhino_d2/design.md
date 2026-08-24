# D2 最小设计说明

## 研究问题

在严格相同的学生结构、初始化、数据、预算、优化器和评测口径下，训练期加入冻结 Foundation Teacher 的单 stage 特征监督，是否能稳定改善小型 YOLO-Master 的检测指标？

## 已有底座与本课题边界

当前仓库已经提供 `FoundationDistillationModel`、`StudentFeatureTap`、P4 projector、cosine/relational/hybrid loss、训练日志和 effect-gate。D2 不重复实现这些能力。本准入包负责锁定实验协议、验证真实 YOLO-Master stage 对齐、提供合规教师降级、生成资产哈希，并把 on/off 对照约束自动化。

## P0 单 stage 原型

```text
输入图像
  ├─ YOLO-Master-N → Detect head 的 P4 来源层 → student [B,Cs,Hs,Ws]
  └─ 冻结教师      → 最后层 patch tokens      → teacher [B,Ct,Ht,Wt]
                                                   │
                           teacher 双线性插值到 (Hs,Ws)
                                                   │
student 1×1 trainable projection ─┐               ├─ frozen 1×1 teacher projection
                                  └─ cosine KD loss ─→ 只更新学生投影/学生侧参数
```

- 学生 stage：P4。代码通过 Detect head 的 `f=[P3,P4,P5]` 自动解析源层，不依赖脆弱的硬编码行号。
- 教师层：最终 patch-token dense feature。
- 空间对齐：只缩放教师特征，保留学生 P4 网格。
- 通道对齐：学生侧 1×1 可训练投影；教师侧投影冻结。
- P0 loss：cosine，形式简单、尺度稳定，便于证明单 batch 下降。
- P1 候选：hybrid = cosine + sampled relational；只有 P0 稳定后才启用。

## 教师双版本锁定

| 角色 | 模型 | 用途 | 许可/访问 |
| --- | --- | --- | --- |
| P1 主教师 | `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056` | 正式 on/off 与 P0 smoke | Hugging Face 模型卡标注 Apache-2.0，公开下载；revision 锁定 |
| 已拒绝备选 | `facebook/dinov3-vits16-pretrain-lvd1689m` | 不执行 | 受控访问申请已被拒绝，不使用社区镜像绕过门禁 |

每次 smoke 记录 Hugging Face snapshot commit、模型文件 SHA-256、配置 SHA-256。更换教师或 revision 后旧缓存和旧证据自动视为失效。

DINOv2 预处理契约为 `dinov2_dense_spatial_preserving_v1`：输入 `[0,1]`，使用 DINOv2 自有 mean/std，保留输入分辨率并仅在右侧/底部补齐 patch 倍数。它有意不执行分类预训练常见的 resize-shortest-edge/center-crop，因为检测蒸馏需要保留 P4 的空间对应；该差异写入教师 metadata 和证据 JSON。

## P1 无混杂变量实验

配对单位为 `(seed, budget)`，同一对中只改变 Foundation 开关与教师字段。计划 seed 为 `20260824, 20260825, 20260826`。主要指标为 `mAP50-95`，辅助指标包括 `mAP50`、训练时间、峰值显存、`foundation_cosine_raw`、`foundation_relational_raw` 和 `foundation_task_ratio`。

统计单位是每个 seed 的 `on - off` 成对差值。报告均值、样本标准差和 t 区间；样本只有 3 个时必须同时声明区间不稳定，不用单次最好结果代替均值。

## 失败诊断顺序

1. 链路：teacher 是否冻结、projector 是否有梯度、loss 是否进入训练总损失。
2. 维度：P4 语义与教师末层是否失配，插值比例是否过大。
3. 优化：raw cosine 是否下降，task/foundation loss 比例是否失衡。
4. 容量：N 模型是否没有足够容量同时拟合检测任务与教师表示。
5. 数据：COCO mini 的类别/尺度和样本数是否造成高方差。

只有完成以上证据链后，才能把负结果解释为“当前蒸馏设计 no-go”。

## P0 后续探索边界

统计对齐是记录完毕但尚未实现/宣称有效的后续假设，完整的可证伪设计、实验臂、测试红线和 logit 蒸馏边界见 [`statistical_distillation_proposal.md`](statistical_distillation_proposal.md)。P0/P1 基线不因该设想而改变，避免把多个 loss、动态权重和第二教师同时缝合进一次实验。
