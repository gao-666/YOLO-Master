# D2 DINOv3-S 正式 P1 Protocol

## 准入与研究问题

Candidate B 已通过预注册 baseline 工程门，因此 2026-09-01 起允许冻结正式 P1。P1 只回答：在同一可解释 Student baseline 上，单 stage DINOv3-S/16 特征蒸馏是否改善 YOLO-Master-N？

Candidate B 本身不产生 KD 结论。DINOv3-L 保持 `ready-but-not-started`，直到 S 的三 seed Go/No-Go 完成。

## 冻结共同条件

| 项目 | 冻结值 |
| --- | --- |
| source protocol commit | 本文件、配置与标定脚本所在 commit；每次运行清单另记精确 commit/hash/dirty |
| Student | `ultralytics/cfg/models/26/yolo26-master-n.yaml` |
| Student init | Ultralytics YOLO26n v8.4.0 partial transfer；SHA-256 `9b09cc8b...4fef` |
| init coverage | target `41.41%`、source `82.35%`；stem/shared-deep/head 100% |
| dataset | frozen COCO mini train=2048、val=512；选择与 payload hash 见 dataset manifest |
| budget | 50 epoch、imgsz=256、batch=4、workers=0 |
| optimizer | SGD、lr0=0.01、AMP=false |
| augmentation | hsv_h/s/v=`0.015/0.7/0.4`、translate=`0.1`、scale=`0.5`、fliplr=`0.5`、mosaic=`1.0`、close_mosaic=`10`；其余几何/mixup/cutmix/copy-paste 为 0 |
| validation | 每 epoch；每个 seed 使用最后 10 epoch mAP50-95 中位数，不选 best epoch |
| paired seeds | `20260824`、`20260825`、`20260826` |

## Teacher 与 treatment

| 项目 | 冻结值 |
| --- | --- |
| Teacher | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| local asset | `E:/2026YOLO/model_cache/dinov3-vits16-pretrain-lvd1689m`；仅本地加载 |
| weights hash | `model.safetensors` SHA-256 `4610ad75...f91d` |
| license | DINOv3 License；不是 Apache-2.0 |
| dtype/device | BF16 / CUDA:0 |
| Student stage | P4，256 输入时 `16×16` |
| Teacher dense grid | patch16，256 输入时 `16×16`；禁止空间 resize |
| projector | 1×1，align_dim=64 |
| loss | cosine-only feature alignment；label detection loss 保持主任务正确性 |
| KD weight | 先按下述 train-only 规则标定；未冻结前禁止正式 ON/OFF |

OFF 与 ON 从同一权重资产、同一模型构造和同一 seed 开始。正式配对只允许 `foundation_enabled`、`foundation_loss_weight` 和输出 identity 不同；Teacher 元数据保留在 OFF 配置中但不会构造 Teacher。

## Train-only KD weight 标定

候选固定为 `0.01, 0.025, 0.05, 0.10`。每个候选使用相同 seed/init/data 训练 1 epoch，`val=false`、`save=false`。选择规则是：Foundation/task training-loss ratio 落在 `[0.03, 0.06]` 的**最小**候选，并且 loss 机制恒等式通过。不得读取或使用 validation AP。

若没有候选入区间，停止正式 P1，先报告 `no_candidate_in_band`；不得看 AP 后扩展候选。选定后生成正式 ON 配置、做 fail-closed pair audit 并 commit，三 seed 完成前不再修改 protocol。

### 标定结果（2026-09-01）

四个候选 ratio 分别为 `0.002623 / 0.006606 / 0.013207 / 0.026325`，均未进入 `[0.03,0.06]`，因此正式 P1 继续停止。完整证据与可能的 protocol correction 边界见 [`DINOV3_P1_CALIBRATION.md`](DINOV3_P1_CALIBRATION.md)。

首轮结果固化后，单候选 0.15 的扩展规则另行版本化在 [`DINOV3_P1_CALIBRATION_EXTENSION_PROTOCOL.md`](DINOV3_P1_CALIBRATION_EXTENSION_PROTOCOL.md)。它保持同一 ratio 区间且仍不读取 validation AP；若该候选失败，不再扩展搜索。

扩展候选得到 ratio `0.0392177` 且机制恒等式通过，因此正式 `foundation_loss_weight=0.15`。正式 OFF/ON 配置为 [`configs/d2_v3_p1_off.yaml`](configs/d2_v3_p1_off.yaml) 与 [`configs/d2_v3_p1_on.yaml`](configs/d2_v3_p1_on.yaml)，训练前必须通过独立 pair validator。

Pair validator 已通过，观察差异严格为 `foundation_enabled`、`foundation_loss_weight` 与输出 `name`；正式三 seed 训练已获准。审计证据见 [`results/d2_v3_p1_pair_validation.json`](results/d2_v3_p1_pair_validation.json)。

### 正式训练进展（2026-09-01）

第一组 paired seed `20260824` 已完成：OFF/ON 的末 10 轮 mAP50-95 中位数分别为 `0.048125 / 0.049465`，配对差值 `+0.001340`。当前仅为方向性单 seed 证据，状态保持 `pending_more_seeds`，不得计算或声称 Go/No-Go。完整说明见 [`DINOV3_P1_FIRST_PAIR.md`](DINOV3_P1_FIRST_PAIR.md)。

第二组 paired seed `20260825` 也已完成：OFF/ON 分别为 `0.054500 / 0.053940`，差值 `-0.000560`。两组方向不一致，当前仍保持 `pending_more_seeds`；只剩冻结协议的 seed `20260826`。完整说明见 [`DINOV3_P1_SECOND_PAIR.md`](DINOV3_P1_SECOND_PAIR.md)。

第三组 seed `20260826` 的 OFF/ON 为 `0.056960 / 0.049670`，差值 `-0.007290`。三 seed mean Δ=`-0.002170`，paired 95% t CI=`[-0.013435,0.009095]`。由于 `|mean Δ|<0.003` 且 CI 包含 0，正式判定 **No-Go**。见 [`DINOV3_P1_GO_NO_GO.md`](DINOV3_P1_GO_NO_GO.md)。

## 统计与 Go/No-Go

对每个 seed 计算：

`d_i = median_last10(ON_i mAP50-95) - median_last10(OFF_i mAP50-95)`。

报告 mean Δ、sample SD、paired 95% CI。Go：mean Δ `>=0.003` 且 CI 下界 `>0`。No-Go：`|mean Δ|<0.003` 且 CI 含 0。其余为 inconclusive；结果模糊时增加 seed，不修改判读线。

## 停止规则

在 S 主实验结束前禁止 DINOv3-L capacity、multi-stage、新 loss、align_dim 消融或基于 validation AP 调权。若正式 S 为 no-go，才进入 P2 的容量/维度/优化/数据机制诊断。
