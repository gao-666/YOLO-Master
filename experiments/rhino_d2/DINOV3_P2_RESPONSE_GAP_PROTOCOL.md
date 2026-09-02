# D2 DINOv3-S P2-03 Static–Response Gap / Counterfactual Response Probe

## 1. 研究位置与冻结边界

已有结论保持不变：

- P1：DINOv3-S、P4、static cosine KD 在正式三 seed 协议下为 **No-Go**；
- P2-01：稳定负向梯度冲突假设为 **Inconclusive**，更明确的观测是 KD 梯度弱且近乎正交；
- P2-02：`align_dim 64 -> 128` 为 **No support**，且 late KD/task 梯度比下降。

P2-03 不训练新模型，不修改 checkpoint，不选择 best epoch，不重新标定 KD weight，也不引入 response loss。它只使用正式 P1 的
三个 OFF 与三个 ON64 `epoch49.pt` EMA checkpoint，回答：

1. static KD 是否让 clean Student 表征更接近 Teacher；
2. clean static gap 的变化是否同步转化为 perturbation response gap 的变化；
3. response gap 是否比 static gap 更能解释扰动导致的检测退化。

P2-03 的结果不能追溯改写 P1、P2-01 或 P2-02，也不能直接证明 Response-Field Distillation 有效。

## 2. 冻结输入

### 2.1 模型与 checkpoint

固定 seed：`20260824/20260825/20260826`。每个 seed 使用：

```text
runs/rhino_d2/v3_p1/v3-p1-off-s{seed}/weights/epoch49.pt
runs/rhino_d2/v3_p1/v3-p1-on-s{seed}/weights/epoch49.pt
```

只加载 EMA 权重。特征与预测提取保持 eval mode。逐图像 detection loss 使用 Ultralytics loss 所需的 raw-head 模式：每次只送入
一张图像（`loss_batch_size=1`），临时调用 `student.train()` 后立即把所有 BatchNorm 模块切回 eval，关闭梯度并调用与 P1 相同的
`student.loss(batch, predictions)`；完成后恢复整个 Student 的 eval mode。全过程不得更新 BatchNorm running statistics、参数、buffer 或
optimizer state，optimizer steps 必须为 0，并以运行前后 `state_dict` SHA-256 完全一致作为 fail-closed 审计。OFF 与 ON64 必须来自同
seed，并自动核验 P1 配置、数据、Student 初始化、训练源码及 checkpoint 中的训练参数。

### 2.2 新 diagnostic split

只使用 COCO mini 的 training split，绝不读取 validation 指标调节协议。冻结文件：

```text
experiments/rhino_d2/results/p2_response_gap/diagnostic_response128.txt
```

- 数量：128 张；
- 选择：从 2048 张训练图像中排除 P2-01 的 `diagnostic_train64.txt` 后，按
  `SHA256("dinov3-p2-response-gap-v1-20260902" || NUL || normalized_path)` 排序取前 128；
- 与 P2-01 子集交集：0；
- 文件 SHA-256：`c0e00ba5e15de01d21afb35ae50937681186905d717498d4549565a3613dd582`。

该列表在任何特征、检测结果或扰动结果产生前冻结。

## 3. 冻结扰动

图像先经过与 P1 相同的确定性 letterbox/resize 到 256，再在 `[0,1]` RGB 张量上施加扰动；Student 与 Teacher 接收逐位完全相同的
clean/perturbed 张量。所有输出裁剪至 `[0,1]`，不使用随机增强。

| family | severity 1 | severity 2 | 精确定义 |
| --- | ---: | ---: | --- |
| brightness | `0.8` | `0.6` | `clip(a*x,0,1)` |
| contrast | `0.75` | `0.50` | `clip(mu+a*(x-mu),0,1)`，`mu` 为逐图像逐通道空间均值 |
| Gaussian blur | `sigma=1.0` | `sigma=2.0` | separable Gaussian，radius=`ceil(3*sigma)`，reflect padding |
| Gaussian noise | `std=0.03` | `std=0.06` | 加性零均值噪声；使用下述冻结 seed 公式 |

共 8 个冻结 perturbation conditions。不得在看到结果后删除某个 family、severity 或图像；失败实例只能标记并触发 fail-closed，
不能静默跳过。

Gaussian noise 的规范化扰动 ID 固定为 `gaussian_noise:0.03` 或 `gaussian_noise:0.06`，路径使用 diagnostic list 中的 LF
规范化字符串。定义：

```text
payload = "dinov3-p2-response-gap-v1" || NUL || normalized_path || NUL || perturbation_id
noise_seed = int.from_bytes(SHA256(payload).digest()[0:8], "big") mod 2^63
```

使用 CPU `torch.Generator`、`manual_seed(noise_seed)` 和 FP32 `torch.randn` 生成一次噪声张量。对同一
`image + perturbation_id`，OFF/ON64、三个 seed 和 Teacher 必须复用逐位完全相同的 perturbed tensor；不得为不同模型重新采样。

## 4. 表征与 gap 定义

Student 和 Teacher 通道数不同，因此主要指标不得依赖事后训练的新映射。对 P4 特征 `F in R^(C x H x W)` 定义冻结的
projection-free spatial-relation embedding：

1. 若空间尺寸不同，adaptive-average-pool 到 `16 x 16`；
2. 展平为 256 个 token，逐通道减去空间均值；
3. 每个 token 沿通道 L2 normalize，epsilon=`1e-12`；
4. 计算 `256 x 256` token cosine Gram matrix；
5. 取不含对角线的严格上三角并展平，记为 `Phi(F)`。

Student P4 与 Teacher P4 在进入 `Phi` 前一律转换为 FP32；pool、center、normalize、Gram、response subtraction、cosine 和
image-level gap 全部以 FP32 计算并在每步检查 finite。统计汇总、Spearman、Fisher-z 和 bootstrap 累积转换为 FP64。不得因
数值结果更稳定或更显著而切换 embedding、dtype、pool size、center 或 normalization 定义。

### 4.1 Static gap

对 arm `a in {OFF,ON64}`：

```text
G_static(a,x) = 1 - cos(Phi(S_a(x)), Phi(T(x))).
```

另记录 ON64 projector objective-space gap 作为辅助审计。该辅助量不能替代 projection-free 主指标，也不能单独支持 H1；它只用于区分
“Student 主表征更接近”与“projector 自身吸收了 static matching”。

### 4.2 Response gap

对每个冻结扰动 `tau`：

```text
R_S(a,x,tau) = Phi(S_a(tau(x))) - Phi(S_a(x))
R_T(x,tau)   = Phi(T(tau(x)))   - Phi(T(x))
G_response(a,x,tau) = 1 - cos(R_S(a,x,tau), R_T(x,tau)).
```

若任一 response norm 小于 `1e-12` 或出现 NaN/Inf，该观测必须报告并使正式分析 fail-closed；不得用任意常数替代。

## 5. 检测退化指标

primary image-level degradation：

```text
Delta L_detect(a,x,tau) = L_detect(a,tau(x),y) - L_detect(a,x,y).
```

使用冻结标签与 P1 相同的 loss 实现，按上一节冻结的 `loss_batch_size=1 + raw-head + BatchNorm eval` 路径逐图像记录 scalar task
loss；不得把 batch aggregate loss 拆分、广播或伪装成 image-level loss。辅助指标：

- post-NMS top-10 confidence mean，少于 10 个检测时以 0 补齐；记录 clean-minus-perturbed confidence；
- 每个 arm/seed/condition 在完整 128 图像上的 diagnostic mAP50-95 drop。

confidence 与 diagnostic mAP 只作稳健性描述；H3 的正式判读只使用 `Delta L_detect`，避免在 128 图像上把小样本 mAP 当作主证据。

## 6. 预注册假设与统计判据

所有 bootstrap 固定为 10,000 次、seed `20260903`、95% percentile CI。图像是重采样 cluster；同一图像的所有 seed、arm 和
perturbation observations 必须一起重采样。报告所有单 seed、单 perturbation family 结果，但不据此挑选结论。

### H1：clean static alignment

对每张图像和 seed 计算：

```text
Delta_static = G_static(ON64) - G_static(OFF).
```

- **Support**：pooled clustered CI 上界 `<0`，且至少 2/3 seed 的 CI 上界 `<0`；
- **Not supported**：pooled clustered CI 下界 `>=0`；
- 其余：**Inconclusive**。

### H2：static–response transfer gap

先对每张图像/seed 将 8 个条件的 response gap 取等权均值，再计算：

```text
Delta_response = mean_tau[G_response(ON64)-G_response(OFF)]
C_gap = Delta_response - Delta_static.
```

由于两者均为 cosine distance，`C_gap>0` 表示 response gap 的改善弱于 static gap。

- **Support**：H1 为 Support，且 pooled `C_gap` CI 下界 `>0`，并至少 2/3 seed 的 `C_gap>0` CI 下界 `>0`；
- **Not supported**：H1 为 Support，且 pooled `C_gap` CI 上界 `<=0`；
- H1 非 Support 时：**Not evaluable**；
- 其余：**Inconclusive**。

### H3：response gap 对检测退化的解释力

在每个 `seed x arm x perturbation-condition` cell 内，对 128 张图像计算 Spearman correlation：

```text
rho_response = corr_rank(G_response, Delta L_detect)
rho_static   = corr_rank(G_static,   Delta L_detect)
```

对 48 个 cells 的 correlation 做 Fisher-z 后等权平均。cluster bootstrap 每次以图像为单位重采样并重算全部相关系数。正式比较：

```text
Delta_rho = mean_z(rho_response) - mean_z(rho_static).
```

- **Support**：`mean_z(rho_response)` 的 CI 下界 `>0`，`Delta_rho` 的 CI 下界 `>0`，且 6 个 seed-arm 聚合中至少 4 个
  `Delta_rho` 点估计为正；
- **Not supported**：`Delta_rho` CI 上界 `<=0`；
- 其余：**Inconclusive**。

跨 severity 的 pooled correlation、confidence drop 和 diagnostic mAP drop 只能作为 secondary descriptive evidence，不得覆盖上述判据。

## 7. 路线决策与停止条件

- 只有 **H1=Support** 且 **H2/H3 至少一个=Support**，才允许提交 Response-Field Distillation 的新训练建议书；
- 若 H1=Not supported，停止“static matching 成功但 response 未迁移”这条解释；
- 若 H1=Support 但 H2/H3 均为 Not supported 或 Inconclusive，不训练 response loss；
- 任一核心假设 Inconclusive 只如实报告，不通过增加 severity、换 checkpoint、换 Teacher 或挑选子集追结果；
- P2-03 完成前禁止 ViT-L、multi-stage、256/512 align dimension 或增加 KD weight。

任何后续 response-loss 训练都必须另立协议并获得明确授权；本协议本身不授权训练。

正式运行前的 smoke test 只能使用 synthetic tensors 检查 shape、determinism、finite、zero-norm fail-closed 与配对逻辑；不得使用
`diagnostic_response128.txt` 中的任何图像，也不得输出任何正式 static/response gap 或 detection degradation 数字。

## 8. 必须归档的证据

- 协议、配置和 diagnostic list 的 SHA-256；
- 六个 checkpoint 的路径、大小和 SHA-256；
- clean/perturbed 图像张量生成参数与噪声 seed；
- 每图像 raw static/response gap 与 detection degradation CSV；
- H1/H2/H3 clustered bootstrap raw/summary；
- 每 arm/seed/condition 的 diagnostic mAP 与 confidence summary；
- 运行环境、源码 commit、GPU、运行日志和无 optimizer-step 证明；
- 明确写出 `supported / not_supported / inconclusive / not_evaluable`，不把相关性写成因果结论。
