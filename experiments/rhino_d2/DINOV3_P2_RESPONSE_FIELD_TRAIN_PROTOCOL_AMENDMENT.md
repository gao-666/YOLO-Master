# DINOv3-S P2-04 Response-Field 训练协议修正案

## 0. 状态与边界

本修正案是提交 `3da75f7da2ec49b254b646915e58574b30c9bb45` 的**实现前规范性修正**。原提交继续作为不可变的预注册节点保留；本文件只覆盖原协议第 3 节的 perturbed Student BatchNorm 处理方式、第 5 节的 `global_batch_index` 定义，以及第 10 节对应的 fail-closed 测试要求。

本修正发生在 response loss 实现、alpha calibration 和任何 formal training 之前，不修改研究问题、arm、estimand、候选 alpha、判读线或停止规则。本提交不授权 alpha calibration、A/B/C formal training 或读取 formal mAP。

## 1. BatchNorm 规范性修正

原协议中“perturbed Student forward 临时把所有 BatchNorm 切到 eval”的条款作废。该做法会使

```text
Z_S(x)     = Student(train-BN) -> Projector(train-BN)
Z_S(tau x) = Student(eval-BN)  -> Projector(eval-BN)
```

从而把 perturbation response 与 train/eval normalization mode shift 混在 `R_S` 中。

替代流程冻结为：

```text
1. clean Student + shared projector forward 保持正常 train mode；
2. clean forward 正常更新一次 BatchNorm running buffers；
3. clean forward 后，逐模块 clone 全部 BatchNorm buffers：
   running_mean / running_var / num_batches_tracked；
4. perturbed Student + 同一 shared projector forward 继续保持 train mode，
   使用 batch statistics 并保留 autograd；
5. perturbed forward 完成后，在 finally 路径逐位恢复第 3 步快照；
6. 再执行本 arm 已声明的 loss、backward 和 optimizer step。
```

快照范围包括 Student backbone/head 路径和 `P4AlignmentProjector` 中参与两次 forward 的所有 `_BatchNorm` 子类。不得只处理 backbone 而遗漏 projector。Teacher 始终 `eval` 且 `no_grad`，不属于该快照范围。

A/B/C 必须执行完全相同的 clean-forward、snapshot、perturbed-forward、restore 路径。一次 batch 结束后的持久 BatchNorm 状态必须与“只执行 clean detection forward”的反事实路径相同；perturbed view 不得产生持久 running-stat corruption adaptation。与此同时，`Z_S(x)` 与 `Z_S(tau x)` 的 BatchNorm `training` flag 必须逐模块一致且均为 `True`，使 `R_S` 不含人为的 train/eval mode jump。

restore 必须 fail-closed：模块集合、buffer 名、shape、dtype 或 device 与快照不一致，或恢复后不能逐位相等时立即终止；不得忽略缺失项、做近似恢复或继续训练。

Arm A 的 perturbed 输出不得进入任意 loss。相对于只运行 clean task 的 matched reference，A 的 perturbed branch 不得改变任何参数梯度、optimizer 更新或持久 buffer；A 仍只作为 forward-matched null reference。

## 2. Resume-stable `global_batch_index`

`global_batch_index` 冻结为与进程启动次数无关的**逻辑训练位置**：

```text
global_batch_index = epoch_index * num_batches_per_epoch + batch_index_within_epoch
```

其中：

- `epoch_index` 为零基的逻辑 epoch 编号，resume 后沿用 checkpoint 中的 epoch，不从 0 重置；
- `batch_index_within_epoch` 为当前逻辑 epoch 内 dataloader 的零基 batch 编号；
- `num_batches_per_epoch = len(train_loader)`，由冻结的数据清单、batch、sampler 和 drop-last 规则决定；
- 三臂同 seed 必须记录并核验相同的 `(epoch_index, batch_index_within_epoch, num_batches_per_epoch, global_batch_index)`；
- checkpoint/runtime manifest 必须记录该定义版本 `epoch-major-v1`、最后完成的逻辑位置和 resume history。

恢复到某个 epoch 内 batch 时，必须从 checkpoint 保存的逻辑位置继续，或由上式对同一位置重新计算；禁止使用“本 Python 进程已看到多少 batch”的临时 counter。uninterrupted 与 resume 路径对相同逻辑训练位置必须生成完全相同的 condition id、noise seed、clean tensor digest 和 perturbed tensor digest，否则 fail-closed。

## 3. 实现前冻结的 synthetic/fail-closed tests

response loss implementation commit 必须至少加入并通过以下测试：

1. perturbed forward 前后，Student 与 projector 的 `running_mean`、`running_var`、`num_batches_tracked` 逐位相同；
2. clean 与 perturbed forward 捕获到的全部 BatchNorm `training` flag 一致且为 `True`；
3. B/C 的 perturbed branch 对 Student perturbed feature 与共享 `P_S` 产生 finite、nonzero gradient；
4. A 与 clean-task-only reference 完成同一 batch 后，parameter gradients、optimizer 更新和持久 buffers 逐位相同；
5. 任一 BN buffer 缺失、shape/dtype/device 不一致或 restore 后不相等时 fail-closed；
6. uninterrupted 与 simulated-resume 对同一逻辑位置得到相同 `global_batch_index`、condition/noise seed 和 paired tensor digest；进程内临时 counter 重置不得改变结果。

这些测试是进入 alpha calibration 的必要条件，不是 formal result。只完成实现或 loss 可下降仍不得跳过 calibration gate。

## 4. 解释边界

P2-04 的 mechanism endpoint 继续使用 P2-03 冻结的 `diagnostic_response128` 和同一四类八条件。如果未来得到 Mechanism Support，允许的表述仅为：

> 在预先冻结的 P2-03 response probe 上，Response-Field intervention 降低了 ResponseGap。

不得据此声称对未知扰动或新分布普遍恢复了 Teacher 动态响应；跨 perturbation family 泛化需要另行预注册 untouched corruption family。

除本文件明确覆盖的条款外，`DINOV3_P2_RESPONSE_FIELD_TRAIN_PROTOCOL.md` 的其他内容继续有效。
