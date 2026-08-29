# PR 草稿：D2 Foundation 单 stage 蒸馏实验闭环

## Summary

- 将许可明确且 revision 锁定的 `facebook/dinov2-small` dense feature 接入 YOLO-Master-N P4 单 stage 蒸馏。
- 提供严格同预算 off/on 配置，以及只移除 relational 对齐的 cosine-only 关键消融。
- 增加可复现实验运行器、逐 epoch checkpoint、日志哈希、结果汇总、曲线和 checkpoint 哈希。
- 提交预注册规则驱动的 [`GO_NO_GO.md`](GO_NO_GO.md)，不以单个正向 seed 代替正式判断。
- 修复训练首次健康 checkpoint 序列化被 Polars 原生 CPU 特性检查阻断的问题：结果读取失败时使用标准库 CSV fallback。

## Tests

- `pytest experiments/rhino_d2/tests/test_admission_contract.py -q`
- `pytest tests/test_foundation_checkpoint.py tests/test_foundation_taps.py tests/test_foundation_projectors.py tests/test_foundation_losses.py tests/test_foundation_distill_model.py tests/test_foundation_config.py tests/test_foundation_dinov2.py -q`
- `pytest tests/test_ddp_checkpoint_coordination.py tests/test_ddp_lifecycle_ema_nan.py -q`
- `pytest tests/test_engine.py -k read_results_csv -q`（2 passed）
- 完整 `tests/test_engine.py` 已执行；本机离线沙箱中 11 passed、24 failed。失败主要是未缓存的 coco8/imagenet10/DOTA/权重下载，另有仓库既有的 multitask stride 对齐失败；本 PR 的新增 CSV fallback 两测均通过。
- off/on 三 seed 共 6 个 COCO128 训练，以及 cosine-only 关键消融均完成 10 epoch；每次均生成 `best.pt`、`last.pt`、`last_healthy.pt` 和 10 个 epoch checkpoint。
- Foundation/checkpoint/DDP 回归合并执行：147 passed、1 skipped；D2 契约与引擎定向回归：12 passed。

## Ablation

COCO128、从零初始化、10 epoch 的 paired off/on：

| Seed | off | hybrid-on | Δ |
| ---: | ---: | ---: | ---: |
| 20260824 | 0.00002 | 0.00023 | +0.00021 |
| 20260825 | 0.00015 | 0.00002 | -0.00013 |
| 20260826 | 0.00002 | 0.00001 | -0.00001 |

Mean Δ=`+0.0000233`，sample SD=`0.0001724`，paired 95% t CI=`[-0.0004050, 0.0004517]`。按预注册规则，`|mean Δ| < 0.003` 且 CI 含 0，因此本协议为正式 **no-go**。

Seed 20260824 的 relational 关键消融：

| Arm | Best mAP50-95 | Δ vs off |
| --- | ---: | ---: |
| off | 0.00002 | — |
| hybrid-on | 0.00023 | +0.00021 |
| cosine-only | 0.00008 | +0.00006 |

Hybrid 相对 cosine-only 为 `+0.00015`，仅作为 relational 项的方向性信号，不足以改变三 seed no-go。

## Limitations

- 三 seed CI 已完成，但 n=3 的 t 区间仍较宽；结论只覆盖当前固定预算协议。
- COCO128 只有 128 张图；学生从零初始化且只训练 10 epoch，absolute mAP 极低，首轮仅验证相对方向和实验闭环。
- DINOv2 patch-14 dense grid 到 YOLO P4 stride-16 grid 需要插值，可能损伤空间监督；替换教师受模型许可约束。
- 当前只验证 P4 单 stage、64-d projector、固定权重 0.05；多 stage、align_dim 和 loss-weight 搜索被刻意后置，避免首轮混杂。
- 教师权重缓存与训练 checkpoint 不提交 Git；提交的是 revision、配置/结果 CSV 和文件 SHA-256。
- Seeds 20260825/26 的 4 个 run 有完整 console tee 及校验哈希；最早的 seed 20260824 在统一运行器加入前执行，仅保留 args、逐 epoch CSV 和 checkpoint 哈希，该证据缺口已显式标注。
