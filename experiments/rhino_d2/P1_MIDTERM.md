# D2 P1 中期汇报：首轮同预算闭环与关键消融

## 一句话结论

COCO128 三 seed 同预算闭环已完成。三个 paired ΔmAP50-95 为 `+0.00021 / -0.00013 / -0.00001`，平均 `+0.0000233`，95% t CI 为 `[-0.0004050, 0.0004517]`。由于 `|mean Δ| < 0.003` 且区间包含 0，按预注册规则给出 **P1 no-go**：当前 10-epoch、从零初始化协议没有检测到可行动的蒸馏收益。它不是“蒸馏永远无效”的结论。

## 已完成范围

- 8.25–8.31：COCO128、10 epoch、seeds 20260824/25/26 的 off/on 最小闭环；每个 epoch 都保存可恢复 checkpoint，共 60 个 paired epoch checkpoint。
- 9.1–9.7：完成去除 relational 项的 cosine-only 关键消融，再保存 10 个 epoch checkpoint；生成配对统计、曲线、CSV、checkpoint 哈希、完整日志运行器、回归测试和 PR 草稿。
- 训练链修复：正式训练首次保存健康 checkpoint 时，Polars 在本机因 `sse3` CPU 特性检测崩溃。引擎现在会在 Polars 不可用/运行失败时退回标准 CSV 读取；未改变训练、损失或指标逻辑。

## 严格同预算设置

配对实验共享：YOLO-Master-N、COCO128、从零初始化、10 epoch、imgsz 256、batch 4、SGD、lr0 0.01、AMP off、相同增广与验证集。每个 seed 内只切换蒸馏字段。

| Seed | off 最佳 mAP50-95 | hybrid-on 最佳 mAP50-95 | paired Δ |
| ---: | ---: | ---: | ---: |
| 20260824 | 0.00002 | 0.00023 | +0.00021 |
| 20260825 | 0.00015 | 0.00002 | -0.00013 |
| 20260826 | 0.00002 | 0.00001 | -0.00001 |

配对统计：mean Δ=`+0.0000233`，sample SD=`0.0001724`，95% t CI=`[-0.0004050, 0.0004517]`，正式判定 `no_go`。

关键消融固定 seed 20260824：

| 臂 | 唯一机制差异 | 最佳 epoch | 最佳 mAP50-95 | 最终 foundation loss |
| --- | --- | ---: | ---: | ---: |
| off | 无教师、KD 权重 0 | 7 | 0.00002 | — |
| hybrid-on | DINOv2-small，P4，cosine + sampled relational，权重 0.05 | 7 | 0.00023 | 0.266586 |
| cosine-only | 与 hybrid-on 相同，仅 relational 权重从 1 改为 0 | 5 | 0.00008 | 0.193054 |

单 seed 消融比较：

- hybrid-on − off：`+0.00021`，即 `+0.021` 个百分点，小于 `0.3` 个百分点判读线。
- hybrid-on − cosine-only：`+0.00015`。它是“relational 项值得继续检验”的方向性信号，不是有效性证明。
- 所有 absolute mAP 都极低，说明 128 图、从零初始化、10 epoch 的评估方差/欠拟合风险很大，不能把本轮数字包装成模型能力结论。

正式证据见 [`results/d2_p1_three_seed_results.json`](results/d2_p1_three_seed_results.json)、[`results/d2_p1_three_seed_results.png`](results/d2_p1_three_seed_results.png) 和 6 份逐 epoch CSV；seed 20260824 的消融证据保留在 [`results/d2_p1_first_results.json`](results/d2_p1_first_results.json)。

## 为什么不是“复制一个 GramLoss”

主假设不是逐元素模仿教师，而是学生需要吸收教师对纹理、遮挡、前景背景和上下文关系的统计结构。当前原型把假设压缩为可证伪的单变量实验：在教师、stage、投影、权重、预算都固定时，只移除 sampled relational 对齐。如果多 seed 后 hybrid 稳定优于 cosine-only，才支持“关系统计提供额外信息”；否则应删除该复杂度。

标签检测损失始终负责“检测内容正确”，foundation loss 只提供中间表示约束。当前没有引入强化学习：没有环境、动作或 episode，只有监督训练中的附加标量损失。

## 竞争性解释与最小诊断

| 假设 | 当前证据 | 最小可证伪实验 | 优先级 |
| --- | --- | --- | --- |
| H1：教师特征在当前预算下没有增量价值 | 三 seed mean 接近 0 且 CI 含 0，当前协议下暂不支持 | 若扩充训练预算/数据后仍复现 no-go，则增强该解释 | 当前主结论 |
| H2：DINOv2 patch-14 到 YOLO stride-16 的插值损伤监督 | 当前 teacher grid 必须 resize 到 P4 | 获得许可明确的 patch-16 teacher 后，只替换 teacher/grid；未获许可前不执行 | 受许可阻塞 |
| H3：64-d 投影瓶颈过窄 | 未检验 | 固定其他变量，align_dim 64 vs 128 | 次高 |
| H4：KD 权重与 detection task 冲突 | task ratio 约 0.019，但 val cls loss 不稳定 | 固定其他变量，loss_weight 0.05 vs 0.01 | 次高 |
| H5：N 型学生容量不足 | 未检验 | 仅在 P1 多 seed 后比较 N/S；不提前扩大搜索空间 | 后置 |
| H6：COCO128 方差掩盖效应 | absolute mAP 极低 | 保持判读线，扩为 COCO mini/中等集，而不是修改阈值 | 若 CI 过宽则执行 |

## 每日 checkpoint / 测试纪律

- 配置固定 `save_period: 1`，每个 epoch 产出 `epochN.pt`；6 个 paired run 加 1 个消融共 70 个 epoch checkpoint，JSON 中记录逐文件 SHA-256。
- 每日结束保留该日 `last_healthy.pt`、最新 epoch checkpoint、完整 console log 和测试结果；checkpoint/log 留在 `runs/rhino_d2/`，不进入 Git，哈希与逐 epoch CSV进入证据包。Seeds 20260825/26 的 4 个 run 已由统一运行器完整 tee 并校验日志哈希；最早的 seed 20260824 在运行器加入前执行，只有 args、逐 epoch CSV 与 checkpoint 哈希，没有原始 console tee，此缺口在 JSON 中显式记录，不伪造日志。
- 新机器使用 `run_p1.py` 顺序运行，拒绝已有目录，防止不同实验混写；`validate_pair.py` 在 off/on 启动前 fail closed。

## 中期演示顺序（3 分钟）

1. 先展示 P4 学生特征、DINOv2 dense teacher feature、投影后同 shape，说明教师冻结且 KD 真进入 total loss。
2. 展示三臂配置差异：off/on 仅蒸馏字段不同；关键消融仅将 relational 权重从 1 变 0。
3. 展示曲线和表格，明确讲出三个 paired delta、mean 与 CI，以及 absolute mAP 很低。
4. 给出结论：“链路、三 seed 闭环和统计判读均完成；当前协议正式 no-go，不宣称涨点，也不外推为蒸馏普遍无效。”
5. 若导师追问创新点：回答“创新假设是统计关系对齐是否比逐点模仿提供额外信息；目前只是方向性信号，消融和多 seed 决定是否保留。”

## no-go 后的建议顺序

1. 不修改判读线，不把 seed 20260824 的正值单独挑出来报告。
2. 优先做 H4（KD weight 0.05→0.01）或 H3（align_dim 64→128）的单变量诊断；二选一，不同时改。
3. 若 absolute mAP 仍接近 0，优先扩训练预算或数据验证 H6，再讨论学生容量和多 stage，避免在欠拟合协议上堆机制。
