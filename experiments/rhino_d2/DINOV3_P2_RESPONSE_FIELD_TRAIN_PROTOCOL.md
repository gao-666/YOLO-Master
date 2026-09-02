# DINOv3-S P2-04 Response-Field matched-compute 干预协议

## 0. 状态、证据来源与授权边界

本文件只预注册 P2-04 干预实验，不实现 response loss，不执行 calibration，不启动训练，也不授权修改 P1/P2 的既有结论。

冻结证据链：

```text
931f9db  P1 Static Foundation KD: No-Go
9efa7f2  P2-01 gradient conflict: Inconclusive
dbad130  P2-02 align_dim 64→128: No support
0ebffb9  P2-03 preregistration
6a0074b  P2-03 implementation
4d0d67e  P2-03 H1/H2 Support, H3 Inconclusive
```

P2-03 正式结果文件 SHA-256 为
`2e1e84027872ff94340aefadc87a5c8dacb928f2d44b169e1a15774dc6382aaf`。它只支持：clean static alignment
得到改善，但该改善没有同步迁移到冻结扰动下的 response alignment。它不证明 response gap 导致检测退化，也不证明 Response-Field
训练一定有效。

本协议必须先独立 commit、push 并通过审阅。批准前禁止编写训练实现；批准后仍须按“实现与 synthetic smoke → calibration →
冻结 alpha → 三 seed formal training”的顺序推进，各阶段证据节点分开。

## 1. 唯一研究问题与主要 estimand

P2-04 只回答：

> 在相同 paired views、相同 Teacher/Student forward 次数和相同 clean static anchor 下，把第二份 Foundation supervision 从
> perturbed static snapshot 改成 Teacher–Student 的有限差分 response，是否能降低 response gap，并进一步改善 clean detection？

主要机制比较：

```text
C Response-Field - B Static-2V
```

主要任务比较：

```text
late10_median_mAP50-95(C) - late10_median_mAP50-95(B)
```

Arm A 只提供“没有 Foundation supervision 时双视图 forward 本身”的 reference；它不能替代 C-vs-B 的主要比较。

## 2. 冻结的基础训练条件

除本协议明确列出的 arm 差异外，全部沿用正式 P1：

- Student：YOLO-Master-N，同一部分迁移初始化资产，SHA-256
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`；
- 数据：COCO mini train/val `2048/512`，不得换 split；
- 训练：50 epoch、imgsz 256、batch 4、workers 0、single GPU、deterministic；
- optimizer：SGD，`lr0=0.01`，其余 schedule 与 P1 相同；
- Teacher：DINOv3-S/16，本地冻结权重、BF16、P4；
- alignment：共享的 P4 `align_dim=64` projector；
- static loss：cosine-only；clean static coefficient `lambda=0.15`；
- seeds：`20260824 / 20260825 / 20260826`；
- 每 epoch 执行相同 clean validation，正式 clean 指标为 epoch 41--50 的 mAP50-95 中位数；不使用 best epoch；
- 不使用 ViT-L、multi-stage、额外 feature level、新 Teacher、新数据或延长 epoch。

三个 arm 的 Student 初始化必须逐 seed 相同。B/C 的 Student projector 与冻结 Teacher projection 也必须逐 seed、逐参数完全相同；
初始化清单和 SHA-256 必须在训练前自动比对。

## 3. 三臂设计与计算边界

| Arm | clean Student | perturbed Student | clean Teacher | perturbed Teacher | Foundation supervision |
| --- | ---: | ---: | ---: | ---: | --- |
| A Null-2V | yes | yes | yes | yes | none |
| B Static-2V | yes | yes | yes | yes | clean static + perturbed static |
| C Response-Field | yes | yes | yes | yes | clean static + response |

三臂必须执行相同数量的 clean/perturbed Teacher 和 Student forward。A 的额外输出不得进入任何 loss，也不得更新 Student；A 因此只是
**forward-matched null reference**，不是 backward-FLOP 完全匹配的 control。B/C 使用相同两份 aligned features、相同 projector、两个
Foundation loss terms 和相同 task loss，是唯一主要 matched-compute 对照。必须报告每 arm 的 wall time、峰值显存、Teacher/Student
forward counts 和 backward counts；不得把“相同 forward 次数”写成“完全相同运行时间”。

clean Student forward 使用正常 train mode 并承担唯一的 task loss。perturbed Student forward 必须保留 autograd，但所有 BatchNorm
模块临时切到 eval，使 perturbed view 不更新 running mean/variance；完成后恢复原状态。Teacher 的两个 forward 始终为 eval/no-grad。
三臂使用完全相同的模式切换。这样 perturbed view 只能通过 B/C 明示的 Foundation loss 影响参数，不能通过隐藏的 BatchNorm state
更新形成 corruption adaptation；A 的 perturbed forward 不得改变任何 parameter 或 buffer。

正式训练为 9 个 run。启动第一个 formal run 后，直到 9 个 run 完成，不得根据任何 seed 的 mAP、loss 或 ResponseGap 修改 alpha、
扰动、训练顺序、checkpoint 或代码。运行顺序冻结为：

```text
A24, B24, C24, A25, B25, C25, A26, B26, C26
```

任一 run 技术失败只能从最后一个已验证 checkpoint 按同一代码和配置恢复；不得以新 seed 替换，也不得静默排除。

## 4. 冻结的 feature 与 loss 定义

对同一共享 alignment projector：

```text
Z_S(x)     = P_S(F_S^P4(x))
Z_T(x)     = stopgrad(P_T(F_T^P4(x)))
Z_S(tau x) = P_S(F_S^P4(tau x))
Z_T(tau x) = stopgrad(P_T(F_T^P4(tau x)))
```

`P_S` 在 clean/perturbed view 间共享；`P_T` 与 P1 一样冻结。Static cosine 按 P1 定义为逐 spatial token cosine distance 的平均：

```text
L_static(v) = mean_token[1 - cos(Z_S(v), Z_T(v))].
```

有限差分 response：

```text
R_S = Z_S(tau x) - Z_S(x)
R_T = Z_T(tau x) - Z_T(x)
L_response = mean_token[1 - cos(R_S, R_T)].
```

所有 cosine 在 FP32 计算，epsilon 固定 `1e-6`；Teacher 始终 detach。出现 zero-norm、NaN 或 Inf 时 fail-closed，禁止以常数替代或
跳过样本。第一版禁止 magnitude loss、relational response、foreground weighting、gating、adaptive scheduler 或任意新 loss term。

三个 arm 的总目标：

```text
L_A = L_task(x)
L_B = L_task(x) + lambda * [L_static(x) + L_static(tau x)]
L_C = L_task(x) + lambda * [L_static(x) + alpha * L_response(x, tau x)]
lambda = 0.15
```

`L_task` 只在 clean/base YOLO training view 上计算。perturbed view 在三个 arm 中均不得进入 detection label loss；否则实验会混入
corruption augmentation training，必须中止而不是继续解释。

## 5. Paired-view 与扰动策略

沿用 P2-03 的 8 个 condition，公式、裁剪、Gaussian kernel 和数值 dtype 不变：

```text
brightness:0.8, brightness:0.6
contrast:0.75, contrast:0.5
gaussian_blur:1.0, gaussian_blur:2.0
gaussian_noise:0.03, gaussian_noise:0.06
```

每个基础样本、每个 epoch 只采一个 condition。condition 必须在 base YOLO training view 形成后施加；clean 与 perturbed Tensor 同时送给
Student/Teacher。对 normalized image path 定义：

```text
payload = "dinov3-p2-response-field-v1"
          || NUL || seed
          || NUL || epoch
          || NUL || global_batch_index
          || NUL || normalized_image_path
condition_index = int.from_bytes(SHA256(payload)[0:8], "big") mod 8
```

Gaussian noise seed 固定为：

```text
noise_seed = int.from_bytes(SHA256(payload || NUL || condition_id || NUL || "noise")[0:8], "big") mod 2^63.
```

同一 seed/epoch/batch/image 在 A/B/C 必须得到逐位相同的 clean 与 perturbed Tensor。每个 run 必须输出逐 batch 的
`(image IDs, condition IDs, noise seeds, clean tensor SHA-256, perturbed tensor SHA-256)` rolling manifest；同 seed 三臂 manifest 不一致
则整个 paired comparison fail-closed。不得在结果后删除某个 condition 或只报告有利 family。

## 6. alpha 的 train-only signal calibration

alpha 禁止根据 validation mAP、P2-03 response128 结果或任何 formal run 调节。候选集一次冻结为：

```text
alpha in {0.25, 0.5, 1.0, 2.0, 4.0}.
```

Calibration 只使用三个冻结 P1 ON64 `epoch49.pt` EMA checkpoint 和
`diagnostic_train64.txt`（SHA-256
`1ad936698234fd07651993dbdefe7a98ffaf74861432a103b6e397bb45b9b676`），不得读取 validation 或
`diagnostic_response128.txt`。在 64 图 x 8 conditions 上一次性缓存 forward；alpha 的候选比较只做解析缩放，不为每个 alpha 重新采样。

在相同样本上缓存 `L_task`、三个未加权 Foundation loss 及它们对 clean/perturbed Student P4 的梯度。令拼接梯度空间为
`F_pair=(F_S(x),F_S(tau x))`，task gradient 的 perturbed 分量固定为 0。定义：

```text
q_loss_B = median[lambda * (L_static(x)+L_static(tau x)) / L_task]
q_loss_C(alpha) = median[lambda * (L_static(x)+alpha*L_response) / L_task]
r_loss(alpha) = q_loss_C(alpha) / q_loss_B

q_grad_B = median[||grad_F_pair lambda*(L_static(x)+L_static(tau x))|| / ||grad_F_pair L_task||]
q_grad_C(alpha) = median[||grad_F_pair lambda*(L_static(x)+alpha*L_response)|| / ||grad_F_pair L_task||]
r_grad(alpha) = q_grad_C(alpha) / q_grad_B.
```

同时描述性记录第二项的裸 loss/gradient 比例，但它们不参与 alpha 选择。候选可接受条件：pooled `r_loss` 与 `r_grad` 均在
`[0.5, 2.0]`，且至少 2/3 seed 的两项比例也同时在该区间。可接受候选中最小化：

```text
score(alpha) = |log r_loss(alpha)| + |log r_grad(alpha)|.
```

若并列，选择离 1.0 最近者；仍并列选择较小 alpha。若没有候选可接受，calibration 判为 **failed**，P2-04 formal training 不得启动；
不得扩展候选集、看 mAP 或临时修改区间。Calibration 必须输出 raw/summary、配置与代码 hash，并在任何 formal run 前独立 commit。

## 7. 机制评测

使用 P2-03 冻结的 `diagnostic_response128.txt`（SHA-256
`c0e00ba5e15de01d21afb35ae50937681186905d717498d4549565a3613dd582`）、同一 8 conditions、projection-free
spatial-relation embedding、image-cluster bootstrap 和数值保护，在 9 个 epoch49 EMA checkpoint 上评测。

主要机制 estimand：

```text
Delta_response_i = mean_(image,condition)[G_response(C_i) - G_response(B_i)].
```

- **Mechanism Support**：pooled image-cluster 95% CI 上界 `<0`，且至少 2/3 seed 的 CI 上界 `<0`；
- **Mechanism No support**：pooled CI 下界 `>=0`；
- 其它：**Mechanism Inconclusive**。

同时报告 `G_static(C)-G_static(B)`，检查 clean static anchor 是否发生结果层面的 trade-off；它不能替代主要 ResponseGap 判读。所有
seed、condition 和 family breakdown 都报告，但只作 secondary evidence。

## 8. 任务与 robustness 评测

### 8.1 主要 clean task endpoint

每个 arm/seed 使用 epoch 41--50 clean validation mAP50-95 的中位数：

```text
d_i = late10_median_mAP50-95(C_i) - late10_median_mAP50-95(B_i).
```

在三个 paired seeds 上使用双侧 95% paired t CI：

- **Task Support**：mean `>= +0.003` 且 CI 下界 `>0`；
- **Task Harm**：mean `<= -0.003` 且 CI 上界 `<0`；
- **No detectable task change**：`|mean| <0.003` 且 CI 包含 0；
- 其它：**Task Inconclusive**。

禁止使用 best epoch、只选有利 seed 或在 50 epoch 后延长训练。

### 8.2 冻结 corruption robustness

对 epoch49 EMA 在完整 512 validation 上分别评测 clean 与 8 个冻结 corruptions，报告每 condition mAP50-95、macro corrupted mAP 和
`clean - corrupted` robustness drop。C-vs-B 的 macro corrupted mAP 使用与 clean endpoint 相同的 paired 规则，但它是 secondary endpoint，
不能在 clean task 不支持时单独把最终方法标为成功，也不能挑某个 corruption 覆盖 macro 结果。

### 8.3 次级 reference comparisons

完整报告 B-vs-A 与 C-vs-A 的 clean、mechanism 和 robustness 指标。它们回答“双视图 static KD 是否有价值”和“Response-Field 整体
是否优于无 KD”，但不能替代主要 C-vs-B 结论。A/B/C 均比旧 P1 多一次 paired-view forward，因此不得把新 arm 与旧 P1 直接作为
matched-compute 因果比较。

## 9. 四类预注册结果与措辞

最终主标签只由 C-vs-B 的 Mechanism 与 clean Task 判定产生：

| 类别 | 机制判定 | clean task 判定 | 允许的结论 |
| --- | --- | --- | --- |
| A 最强支持 | Support | Support | Response-Field 改变预期机制，并伴随 clean detection 收益；仍不使用因果措辞 |
| B 机制成功、任务不涨 | Support | No detectable change | response alignment 可被操纵，但当前 mismatch 不是 clean detection 收益的充分机制 |
| C 方法未操纵机制 | No support | 任意 | 当前有限差分 response loss 未成功降低目标 ResponseGap；不得用 mAP 绕过机制失败 |
| D 动态模仿伤害任务 | Support | Harm | 更接近 Teacher 的动态响应伴随 Student 检测特化受损 |

Mechanism 或 Task 为 Inconclusive、或组合未落入 A--D 时，主标签就是 **Inconclusive**，不得事后改阈值。Robustness 作为独立
secondary modifier 报告，不改变 A--D 主标签。

## 10. 训练前 fail-closed 审计

Formal training 前必须自动验证：

1. 本协议、P2-03 protocol/result、数据清单和 calibration 证据 hash；
2. A/B/C 除冻结 arm 差异与输出 identity 外无其他配置差异；
3. 同 seed Student 初始化相同，B/C projector 初始化相同；
4. Teacher 本地资产、Student 初始化、数据图像/标签 inventory 与 P1 冻结值一致；
5. task loss 只接收 clean view；perturbed view 的 detection-loss gradient 必须为零；
6. B/C 的 Teacher/Student forward count、loss-term count和输入 tensor digest 一致；
7. alpha calibration 没有读取 validation、response128 或 formal mAP，且选择规则逐项通过；
8. synthetic smoke 覆盖 response formula、detach、shared projector、zero norm、finite、deterministic perturbation 和 paired digest；
9. 训练实现、runner、validator、summarizer、测试均为 clean tracked commit；
10. 无 CLI override 能修改 alpha、lambda、loss、扰动、seed、epoch、data 或 Teacher。

任一项失败则禁止 formal run。审计失败不能通过手工改 manifest、跳过样本或关闭测试绕开。

## 11. 停止规则与禁止事项

- 协议审阅前不实现；实现审计与 synthetic/P0 smoke 通过前不 calibration；calibration 独立冻结前不 formal training；
- formal seed24 启动后，不因任何中间结果修改 alpha 或 loss；三 seed 完成前不判定；
- P2-04 完成前禁止 ViT-L、256/512 align dimension、multi-stage、magnitude/relational response、gating、adaptive weight、更多 corruption；
- Mechanism No support 时停止当前有限差分 loss，不通过加复杂项追结果；
- Task Inconclusive 只允许按新协议补 seed，不得调 alpha、挑 epoch 或换数据；
- P1 No-Go、P2-01 Inconclusive、P2-02 No-Support、P2-03 H1/H2 Support 与 H3 Inconclusive 永久保留原义；
- 所有实现、calibration、formal result 的 commits 与 protocol commit 分开；未经新的明确授权不 push 结果、不开始后续方法扩展。

## 12. 必须归档的证据

- protocol/config/code/test hashes 与 Git SHA；
- calibration 64 图 raw/summary、alpha decision、无 validation 访问审计；
- 9 个 run 的 resolved args、完整 log、50 epoch results、epoch49 checkpoint hash、resume history；
- paired-view rolling manifests 与 A/B/C 同 seed 一致性报告；
- Teacher/Student forward counts、backward counts、wall time、峰值显存；
- 128 图 mechanism raw/summary 与 10,000 次 image-cluster bootstrap；
- clean late10 paired table/CI；完整 8-corruption validation raw/macro summary；
- A--D 主标签、robustness secondary modifier 与所有边界声明。

本协议的完成标准是产生可否证的 C-vs-B matched-compute 干预结论，不是“必须涨点”。
