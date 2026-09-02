# DINOv3-S P2-03 Static–Response Gap 结果

正式判定：**H1=Support，H2=Support，H3=Inconclusive**。预注册的 Response-Field 机制准入 gate 已通过，
但本 probe 不授权任何新训练；后续训练仍需另立协议并获得明确批准。

## H1：clean static alignment

- pooled `ON64−OFF` static gap：`-0.051855`，95% image-cluster CI `[-0.059306, -0.044175]`；判定 **Support**。
- 三个 seed 的点估计依次为 `-0.071757` / `-0.080654` / `-0.003153`；前两个 seed 的 CI 全低于 0，第三个跨 0。

这说明原 P1 static KD 确实使 projection-free clean P4 spatial relation 更接近 Teacher；并非只有 projector objective 下降。

## H2：static 改善是否迁移到 response

- pooled `C_gap=Δresponse−Δstatic`：`+0.060675`，95% image-cluster CI `[0.052382, 0.069141]`；判定 **Support**。
- 三个 seed 的 `C_gap` 点估计依次为 `+0.083696` / `+0.087669` / `+0.010661`；前两个 seed 的 CI 全高于 0，第三个跨 0。
- 仅作描述的 pooled `Δresponse=ON64−OFF` 点估计为 `+0.008821`；三个 seed 分别为 `+0.011939` / `+0.007015` / `+0.007508`。该点估计表示 response gap 没有随 static gap 一起下降，反而略增；它没有替代预注册的 `C_gap` 判据。

因此当前证据支持更严格的表述：**Student 在 clean 点上更接近 Teacher，但这种改善没有迁移到冻结扰动下的局部响应。**

## H3：response gap 与检测退化的关联

- `mean_z(rho_response)`：`-0.006015`，95% CI `[-0.056409, 0.043735]`。
- `Delta_rho`：`+0.032081`，95% CI `[-0.062940, 0.128568]`；6 个 seed-arm 中 4 个点估计为正。
- 两个正式 CI 均跨 0，因此判定 **Inconclusive**。不能声称 response gap 导致检测退化，也不能声称它比 static gap 具有稳定更强的解释力。

## 完整性与边界

- 3 seeds × 2 arms × 128 images × 8 conditions = **6,144** 条唯一 paired rows；48 个 cell 均为 128 图。
- Bootstrap 单位是 image；同图像的 seed/arm/condition 观测一起重采样。
- 六个 EMA checkpoint 的 `state_dict` SHA-256 运行前后完全一致；训练次数和 optimizer steps 均为 0。
- 24 个 seed-condition 配对结果全部保存在 `d2_v3_p2_response_gap_condition_summary.csv`；单条件结果只作 secondary breakdown。
- P1 No-Go、P2-01 Inconclusive 与 P2-02 No-Support 均保持不变。
- Gate 通过只允许提出新的 Response-Field 训练协议，不等于该训练方法已有效，更不构成因果证明。
