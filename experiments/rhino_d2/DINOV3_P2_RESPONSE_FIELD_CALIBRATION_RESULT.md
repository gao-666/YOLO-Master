# DINOv3-S P2-04 train64-only alpha calibration 结果

## 结论

**Calibration Failed。** 五个冻结候选均未同时满足 pooled `r_loss/r_grad` 位于 `[0.5, 2.0]` 且至少 2/3 seed 同时通过的条件。因此：

```text
selected_alpha = null
formal_training_authorized = false
failure_action = stop_p2_04_no_formal_training
```

本结果只说明冻结候选集中没有可接受的 **signal-matched alpha**；不得把任何候选称为 optimal alpha，也不评价检测性能。

## 冻结校准结果

| alpha | pooled r_loss | pooled r_grad | 通过 seed | score | 判定 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.25 | 0.848179 | 4.259234 | 0/3 | 1.613752 | reject |
| 0.50 | 1.199391 | 8.289098 | 0/3 | 2.296755 | reject |
| 1.00 | 1.894171 | 16.467243 | 0/3 | 3.440154 | reject |
| 2.00 | 3.293676 | 32.853580 | 0/3 | 4.684065 | reject |
| 4.00 | 6.098335 | 65.641474 | 0/3 | 5.992223 | reject |

最小候选 `alpha=0.25` 的 loss 量级进入接受区间，但 pooled `r_grad=4.259234`，三个 seed 的 `r_grad` 分别为 `4.698584 / 4.135039 / 4.000393`，均超过冻结上界 2.0。协议禁止继续向下扩展 alpha、放宽区间或只按 loss 选择，因此没有 winner。

## 执行与审计边界

- 输入：3 个 P1 ON64 `epoch49.pt` EMA checkpoint；64 张 `diagnostic_train64` 图像；每图 8 个冻结扰动；`lambda=0.15`。
- 观测：每 seed 128 个 batch-condition observation，共 384 个 raw observation；rolling manifest 共 1536 个 image-condition 记录。
- 五个 alpha 全部由同一批 cached raw loss/gradient 解析缩放，`alpha_forward_reruns=0`。
- 梯度只由 `torch.autograd.grad` 取得；`optimizer_steps=0`，没有 scheduler step 或 model-EMA update。
- 三个 checkpoint 文件 SHA-256 和内存 `state_dict` digest 在校准前后逐位一致；参数 `.grad` 全程保持 `None`。
- 真实 train-mode task loss 会短暂更新 `_mixture_loss_ema_buf`；实现对每个 observation 快照并逐位恢复全部 buffers，消除了处理顺序带来的隐藏状态。
- `no_validation_access=true`、`no_response128_access=true`、`formal_training_started=false`。

机器可读证据位于 [`results/p2_response_field/calibration/`](results/p2_response_field/calibration/)：

```text
calibration_raw.csv
calibration_summary.csv
calibration_result.json
calibration_manifest.json
calibration.log
```

## 停止线

根据 `3da75f7 + 6ed5809` 的预注册规则，P2-04 在本 gate 停止。不得启动 A24，不得读取 formal A/B/C 指标来反向选择 alpha。若未来要研究更小 alpha 或其他归一化方式，必须作为新的、独立预注册课题，而不是修改本次判读线。
