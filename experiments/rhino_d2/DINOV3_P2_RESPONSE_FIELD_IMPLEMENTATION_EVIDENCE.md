# DINOv3-S P2-04 Response-Field 实现与 synthetic smoke 证据

## 状态

**Implementation + synthetic/P0 smoke：通过。**

该结论只证明 response-field 公式、确定性 paired-view、train-BN buffer 回滚、梯度链和 fail-closed 保护在合成张量上工作。它不是 alpha calibration，不包含 validation mAP，也不授权 A/B/C formal training。

不可变节点：

```text
3da75f7  P2-04 原始预注册
6ed5809  实现前 BN / resume 规范修正
52b5bb6  response-field primitives 与首批测试
ead45e3  补齐 buffer-missing fail-closed 测试
```

## 实现内容

[`ultralytics/nn/foundation/response.py`](../../ultralytics/nn/foundation/response.py) 新增：

- `response_field_kd_loss`：以 FP32 计算 `1-cos(Z_S(tau x)-Z_S(x), Z_T(tau x)-Z_T(x))`，Teacher 两支均 detach；zero norm、NaN、Inf 均 fail-closed；
- `strict_cosine_kd_loss`：供 P2 静态 anchor 使用的严格逐 token cosine；
- `BatchNormBufferSnapshot` / `preserve_batchnorm_buffers`：clean forward 后同时覆盖 Student 与共享 projector 的 `running_mean`、`running_var`、`num_batches_tracked`；perturbed forward 保持 train semantics，结束后重新绑定快照克隆并逐位核验；
- `logical_global_batch_index`：冻结为 `epoch_index * num_batches_per_epoch + batch_index_within_epoch`，版本为 `epoch-major-v1`；
- `build_response_field_paired_view`：按预注册 payload 选择四类八条件、生成 noise seed，并归档逐图 clean/perturbed tensor SHA-256。

BN restore 使用“注册 buffer 重新绑定”而非原地 `copy_`。原因是原地改写会增加 autograd version，导致 BatchNorm backward 拒绝在 forward 与 backward 之间被修改的 running-stat Tensor；重新绑定既恢复模块的持久状态，又不修改计算图仍引用的旧 Tensor。

## Synthetic smoke

运行入口：

```powershell
python experiments/rhino_d2/scripts/smoke_response_field.py
```

运行绑定到 clean commit `ead45e35aee4ac4b70a6258b7e620f25422d2907`；六个实验输入均记录 SHA-256，`experiment_inputs_dirty=false`。

核心结果：

| 检查 | 结果 |
| --- | --- |
| A/B/C clean tensor、perturbed tensor、rolling manifest | 完全一致 |
| clean / perturbed BN training flags | 全部一致且为 `True` |
| perturbed forward 后 Student + projector BN buffers | 逐位恢复 |
| Arm B perturbed feature / projector gradient norm | `0.007922 / 0.090336` |
| Arm C perturbed feature / projector gradient norm | `0.029709 / 0.103637` |
| Arm A 相对 clean-only 的 gradient/update/buffer 差异 | 无，逐位一致 |
| 12 步 synthetic static+response objective | `2.109428 -> 1.139725` |
| validation/formal metric access | 无 |
| alpha calibration | 未执行 |

## 测试覆盖

新增测试覆盖：response 公式、Teacher detach、zero norm/non-finite、B/C 梯度、A null effect、clean/perturbed BN flag、buffer bitwise restore、缺失/metadata/mode fail-closed，以及 uninterrupted/resume 的逻辑索引与 tensor digest 恒等。

完整回归包括 `experiments/rhino_d2/tests` 与 `tests/test_foundation_distill_model.py`。结果文件见下方证据入口。

## 证据入口

- [`results/p2_response_field/smoke/d2_v3_p2_response_field_smoke.json`](results/p2_response_field/smoke/d2_v3_p2_response_field_smoke.json)：机器可读 gate、输入 hash、逐图 digest、arm 梯度与 loss curve；
- [`results/p2_response_field/smoke/d2_v3_p2_response_field_smoke.log`](results/p2_response_field/smoke/d2_v3_p2_response_field_smoke.log)：完整 smoke 输出；
- [`results/p2_response_field/smoke/pytest-response-field.xml`](results/p2_response_field/smoke/pytest-response-field.xml)：回归测试记录；
- [`tests/test_response_field.py`](tests/test_response_field.py)：实现级 fail-closed 测试；
- [`tests/test_response_field_evidence.py`](tests/test_response_field_evidence.py)：归档证据一致性测试。

归档 SHA-256：

```text
d2_v3_p2_response_field_smoke.json  d5bca9025f773b3926296d398bbb44bb68aebbcf27aaeb8600437a430390122f
d2_v3_p2_response_field_smoke.log   d5bca9025f773b3926296d398bbb44bb68aebbcf27aaeb8600437a430390122f
pytest-response-field.xml           422ad8192817964be219943bc0379e1c3a90167d82da0b7a764e50a42eb8aa10
```

## 当前停止线

当前阶段停在 `passed_synthetic_implementation_gate`。下一阶段只能在明确批准后执行 train64-only alpha calibration；在 calibration 单独冻结并通过以前，不得启动 formal run、读取 formal mAP 或调整候选集。
