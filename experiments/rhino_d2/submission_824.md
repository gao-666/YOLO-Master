# D2 8.24 准入表填写稿

## 在线表格简版

| 第一志愿 | 环境安装 | 基线/最小任务 | 复现命令 | 配置文件 | 完整日志 | 结果证据 | 设计说明 | 风险与降级 | 代码/方案链接 | 检查结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D2 | 已完成 | 已完成 | 已完成 | 已完成 | 已完成 | 已完成 | 已完成 | 已完成 | [D2 准入包](https://github.com/gao-666/YOLO-Master/tree/rhino-d2-foundation-distillation/experiments/rhino_d2) | 留给导师填写 |

## 审核备注详版

| 字段 | 可粘贴内容 |
| --- | --- |
| 环境安装 | Conda `yolo-master-d2`；Python 3.11.15；PyTorch 2.11.0+cu128；YOLO-Master 8.4.101 editable；RTX 5070 Ti 16GB；CUDA 可用。 |
| 基线/最小任务 | YOLO-Master-N P4 与冻结 DINOv2-small 对齐；真实 YOLO detection loss 实现 + 确定性合成检测 target 的 fixed-batch smoke；cosine KD 与检测 loss 共同反传 5 步。 |
| 复现命令 | `python experiments/rhino_d2/scripts/d2_p0_train_smoke.py --config experiments/rhino_d2/configs/d2_p0.yaml --offline`；`python experiments/rhino_d2/scripts/plot_p0_loss_curves.py`；专项 `pytest` 93 passed。 |
| 配置文件 | `configs/d2_p0.yaml`、`d2_smoke.yaml`、`d2_off.yaml`、`d2_on.yaml`；教师 revision、许可、seed、预算和配置 hash 已锁定。 |
| 完整日志 | `results/d2_p0_train_smoke.json`、`d2_alignment_smoke.json`、`d2_p0_loss_curves.png/csv/manifest.json`、`d2_config_pair_validation.json`、`pytest-foundation.xml`、`env/environment.json`。 |
| 结果证据 | P0 9 项检查全部通过；最终离线重跑 KD `0.09991→0.09764`（原值 `0.0999130011→0.0976409763`）；独立 projector smoke `0.97372→0.38375`；两组原始 loss 曲线及数据哈希已提交；KD 进入 total；学生/投影梯度非零；teacher frozen 且不在 optimizer；93 tests passed。 |
| 设计说明 | `P0_EVIDENCE.md` 集中提供单 stage P4 原型图、两组 loss 曲线及设计依据；学生 1×1 trainable projector；教师投影/教师冻结；cosine P0；同预算 on/off；统计蒸馏作为 P0 后独立实验臂，不与基线缝合。 |
| 风险与降级 | DINOv3 访问申请已被拒绝，P1 已正式切换到锁定 revision 的 DINOv2-small（Apache-2.0）；不使用社区镜像；P0 不声称 mAP 提升；OOM 同时降低 on/off batch；结果按多 seed/CI 判定。 |
| 代码/方案链接 | `https://github.com/gao-666/YOLO-Master/tree/rhino-d2-foundation-distillation/experiments/rhino_d2` |

