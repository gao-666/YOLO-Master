# D2 DINOv3 受控复验协议

## 研究问题

> 在具有有效检测学习信号的 YOLO-Master-N 协议下，冻结 DINOv3 ViT-S/16 的 P4 对齐特征是否相对于无 Foundation KD 的 baseline 改善 mAP50-95？

历史 DINOv2-small 结果冻结在 commit `317f8a97f12262e91c2de07dc5becb731eda5d8a`，继续作为历史 no-go，不被本阶段配置或结果覆盖。新主实验固定为 `OFF vs DINOv3 ViT-S/16`；ViT-L/16 只作为主实验结束后的第二级 Teacher-capacity 实验，不能事后改写为主实验。

## 十二条红线

| 红线 | 执行要求 |
| --- | --- |
| 1. 不覆盖旧证据 | 原 DINOv2 配置、JSON/CSV、GO/NO-GO 全部保留。 |
| 2. DINOv3 是新 protocol | 使用新配置、run 名、results 文件和报告。 |
| 3. Teacher 来源固定 | 只使用导师提供的 ModelScope 本地资产；记录模型 ID、本地目录、文件 hash 和许可证据。 |
| 4. 禁止 NaN 修补 | Teacher feature 出现 NaN/Inf 立即失败；禁止 `nan_to_num()` 后继续。 |
| 5. ViT-L 禁止 FP16 | 正式 ViT-L 至少使用 BF16；S/L 容量比较必须使用共同稳定 dtype。 |
| 6. ON/OFF 同预算 | Student、数据、epoch、batch、imgsz、优化器、lr、增广、seed、val 完全相同。 |
| 7. 不用 val AP 调权重 | KD weight 只按事前冻结的训练 loss ratio 规则标定。 |
| 8. P1 前冻结 loss | P1 开跑后不切换 cosine/hybrid/relational。 |
| 9. P1 前冻结判读线 | 结果出来后不修改 `0.003 + paired 95% CI` 规则。 |
| 10. 不挑 seed | 固定 paired seeds `20260824/25/26`，完整报告。 |
| 11. baseline 不健康就停止 | baseline 仍近零时停止 P1，先排查检测 pipeline。 |
| 12. 不同时改多个机制 | Teacher、align_dim、multi-stage、weight、data 不得同轮一起变。 |

## Gate 顺序

1. **资产 gate**：S/L 配置、权重、LICENSE 和 model card 的 SHA-256 完整登记。
2. **dtype gate**：先验证 S-BF16；S/L 正式容量比较统一 BF16，若任一不稳定则共同退至 FP32。
3. **P0 gate**：S/L 分别通过 shape、finite、冻结、optimizer 隔离、学生/projector 梯度、KD 入总损失和 fixed-batch 下降检查。
4. **baseline sanity gate**：只改变 epoch，先验证 YOLO-Master-N 检测 pipeline 能学习。工程门槛预注册为 late/final mAP50-95 `>=0.01`，且 Precision/Recall 非零、检测 loss 有下降趋势；这不是 efficacy 的 Go/No-Go 线。
5. **protocol freeze gate**：仅在 baseline sanity 通过后冻结正式 dataset、epoch、初始化、loss、weight、seed 和统计规则。
6. **P1 efficacy**：三个 paired seed 的 `OFF vs DINOv3-S/16`；结果模糊时增加 seed，不移动阈值。

## 当前固定项与未冻结项

- Teacher 主路线：`facebook/dinov3-vits16-pretrain-lvd1689m`，online frozen Teacher。
- Teacher 容量备路线：`facebook/dinov3-vitl16-pretrain-lvd1689m`，主 P1 后执行。
- Student stage：P4；`align_dim=64`；patch16 与 stride16 对齐。
- dtype：S-BF16 与 L-BF16 本机 gate 均已通过；后续 S/L 正式实验统一使用 BF16。
- 正式 P1 dataset/epochs/pretrained/weight：**尚未冻结**，不得从 pilot 结果倒推选择。
- cache：本轮不启用；online/cache 等价性以后作为独立工程实验。

## 2026-08-31 Gate 状态

- 资产 gate：通过；S/L 权重、配置、预处理、LICENSE 与 model card 已登记 SHA-256。
- dtype gate：通过；S=`[2,384,14,14]`、L=`[2,1024,14,14]`，BF16 输出均全有限。
- P0-S：通过；P4 对齐无需 resize，KD `0.09785→0.09156`。
- P0-L：通过；P4 对齐无需 resize，KD `0.09852→0.09682`。
- baseline sanity：尚未运行；P1 协议仍未冻结。

## 判读规则

- 主指标：mAP50-95。
- 每个 seed：`d_i = mAP_ON,i - mAP_OFF,i`。
- Go：mean Δ `>=0.003`，且 paired 95% t CI 全部大于 0。
- No-go：`|mean Δ|<0.003`，且 paired 95% t CI 包含 0。
- `0.003` 表示 0.3 个百分点；正式 P1 前仍需导师确认该单位解释，确认后写入冻结协议。
