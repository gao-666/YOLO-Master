# D2 DINOv3-S 正式 P1：第二组配对结果

## 当前状态

截至 2026-09-01，seed `20260824` 与 `20260825` 的正式 OFF/ON 配对均已完成。protocol、训练配置和 runner 内容未改变；seed25 从第一组远程证据节点 `4ac7f7a` 启动。

## 已完成数字

| seed | OFF 末 10 轮中位数 | ON 末 10 轮中位数 | ON−OFF |
| --- | ---: | ---: | ---: |
| 20260824 | 0.048125 | 0.049465 | +0.001340 |
| 20260825 | 0.054500 | 0.053940 | -0.000560 |

两个差值方向不一致，且都小于 `0.003` 的预注册尺度。但正式设计要求三组 paired seed 后再计算 mean Δ、sample SD 和 paired 95% t CI，因此当前仍为 `pending_more_seeds`，不得提前判 No-Go。

## 机制与审计

- seed25 OFF 的 foundation loss 全程为 0。
- seed25 ON 的 foundation loss 从 `0.598250` 降至 `0.200051`，辅助优化链真实生效。
- seed25 ON 最后 10 轮 foundation/task ratio 中位数为 `0.027563`。
- OFF25/ON25 均完成 50 epoch、退出码为 0、experiment inputs clean，并各有 53 个 checkpoint 条目。
- seed24 与 seed25 的同 arm config SHA-256 相同，runner SHA-256 相同；两个 seed 之间新增的只是第一组报告、汇总器和证据文件，不是实验输入变化。

## 下一步

只运行冻结协议的 `OFF26 → ON26`。在完成 seed26 前，不调整 KD weight、align_dim、Teacher、loss、stage、epoch、数据、初始化、增强、汇总口径或判读线。

## 证据入口

- 机器可读汇总：[`results/d2_v3_p1_pair_s20260825.json`](results/d2_v3_p1_pair_s20260825.json)
- 单行配对表：[`results/d2_v3_p1_pair_s20260825.csv`](results/d2_v3_p1_pair_s20260825.csv)
- 完整曲线：[`results/d2_v3_p1_pair_s20260825.png`](results/d2_v3_p1_pair_s20260825.png)
- OFF/ON 的 CSV、resolved args、完整日志与 runtime manifest：`results/d2_v3_p1_{off,on}_s20260825.*`
