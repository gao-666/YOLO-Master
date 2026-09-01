# D2 Baseline Recovery / Protocol Qualification

## 阶段定位

DINOv3 S/L 的 BF16 与单 stage P0 已通过，但 `COCO128 + scratch + 50e` 的 OFF baseline 未通过预注册工程门。当前阶段只诊断 Student baseline，不产生 KD、Teacher capacity 或 DINOv3 efficacy 结论。

## 诊断预算与停止规则

只允许两项按顺序执行的单变量候选：

1. **Candidate A（数据规模）**：从 COCO128 改为冻结的 COCO mini，其他训练项不变。
2. **Candidate B（Student 初始化）**：仅当 A 失败时，在同一 COCO mini 上只把 scratch 改为许可与哈希明确的 Student 初始化。

若 B 仍失败，停止 KD efficacy，转为数据、标签、loss、优化器、模型构造和评测管线的正控检查。禁止形成 epoch、optimizer、lr、imgsz、batch、pretrained、data 同时试探的无解释调参链。

## Candidate A：2026-09-01 预注册

研究问题：在保持 Student 与训练配方不变时，扩大并规范化数据是否足以让 OFF baseline 进入可解释区间？

| 项目 | 冻结值 |
| --- | --- |
| 唯一研究变量 | `coco128.yaml` → `d2_coco_mini_2048_seed20260901.yaml` |
| 数据来源 | COCO 2017 官方 train2017/val2017 split |
| 子集 | train=2048，val=512，官方 split 互斥 |
| 选择 | seed `20260901`；split-specific SHA-256 派生 seed；`random.sample`；输出排序 |
| Student | `ultralytics/cfg/models/26/yolo26-master-n.yaml` |
| 初始化 | scratch，`pretrained=false` |
| 预算 | 50 epoch，batch=4，imgsz=256，workers=0 |
| 优化 | SGD，lr0=0.01，AMP=false |
| 训练 seed | `20260824`，deterministic=true |
| Foundation | disabled；Teacher=none；weight=0 |
| 评测 | 每 epoch val；不使用 best epoch 过门 |

`d2_v3_baseline_recovery_a.yaml` 与失败的 `d2_v3_off_sanity.yaml` 只允许 `data/name/project` 不同。数据图像与标签不进入 Git；Git 只保存生成脚本、选择清单、dataset YAML、源/选择/payload 哈希清单。

## 工程门与分叉

继续使用既有门槛，不根据 Candidate A 结果修改：

- 完成 50 epoch；
- 最后 10 epoch mAP50-95 中位数 `>=0.01`；
- 最后 10 epoch Precision、Recall 中位数均 `>0`；
- final/initial detection loss `<=0.90`；
- 不使用 best epoch 选择协议。

若全部通过：将数据规模视为强解释，冻结新的 DINOv3 P1 protocol，再做不看 validation AP 的 DINOv3-S train-only KD weight calibration。若任一失败：不启动 KD，进入 Candidate B；数据、预算和工程门全部沿用 A，只允许 Student 初始化改变。

## Candidate A 结果（2026-09-01）

Candidate A 完成 50/50 epoch，但后 10 epoch mAP50-95 中位数为 `0.00546`，低于 `0.01`，因此 **失败关闭**。Precision 中位数 `0.245115`、Recall 中位数 `0.01684` 均非零，检测 loss `9.36099→6.59439`（retention=`0.70445`），其余四项检查通过。

正式 P1 仍未解锁。下一步只审计并运行 Candidate B 的 Student 预训练初始化；详细证据和边界见 [`BASELINE_RECOVERY_A.md`](BASELINE_RECOVERY_A.md)。

## Candidate B：训练前资格协议

Candidate B 不替换 Student 架构。`model` 仍为 `yolo26-master-n.yaml`，仅通过训练器的 same-name/same-shape 迁移，把官方 Ultralytics YOLO26n 初始化加载到兼容参数；MoE 专属参数保持 Student 自己的初始化。

资产冻结为 `ultralytics/assets` release `v8.4.0` 的 `yolo26n.pt`，SHA-256 为 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`，许可记录为 AGPL-3.0，权重 payload 不提交 Git。训练前必须通过：目标参数覆盖率 `>=40%`、源参数覆盖率 `>=80%`，并且 stem、共享深层 backbone、完整检测 head 均 100% 迁移。覆盖不足时 Candidate B 不执行。

资格门只回答初始化是否足够兼容，不回答检测或 KD 是否有效。Candidate B 与 A 仅允许 `pretrained/name/project` 不同；其中唯一研究变量是 `pretrained`，其余两项只隔离输出。

## 许可边界

COCO 图像保留各自 Flickr 许可，使用者需遵守 COCO 官方 Terms of Use；COCO annotations 与 Ultralytics 转换标签资产有各自条款。本仓库不提交图像或标签 payload，只提交可复现选择与完整性哈希。该记录不是法律意见。
