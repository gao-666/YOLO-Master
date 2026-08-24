# D2 统计蒸馏探索方案（P0 后续，不冒充已实现结果）

## 1. 范式假设

逐元素特征匹配隐含了“学生应复现教师每个空间位置和通道数值”的强假设。D2 的探索假设是：Foundation 教师更值得迁移的可能是通道共激活、均值、方差与空间相关性等统计结构，而不是每个元素的绝对值。

这是待证伪假设，不预先声称 Gram/协方差一定代表纹理、遮挡、前景背景或多尺度上下文。尤其是空间维展平后的 Gram 矩阵会弱化位置排列信息，因此必须通过严格对照验证它对检测任务是否有帮助。

## 2. 为什么不能复制一个 GramLoss 就结束

直接复制风格迁移的 GramLoss 会留下四个问题：

1. 学生和教师通道数、空间分辨率不同，必须先复用现有 projector 对齐。
2. 未中心化 Gram 同时混合均值和二阶矩，数值尺度受通道数、空间 token 数和激活幅值影响。
3. `C×C` Gram 的显存和计算随对齐维度平方增长，需要固定 `align_dim` 并记录峰值显存。
4. 多一个非零 loss 不等于对检测有效；必须证明它进入 total loss、产生正确梯度并在同预算实验中优于基线。

## 3. 最小统计目标

对已经对齐且按通道标准化的特征 `F ∈ R^(B×C×H×W)`，令 `N=H×W`，展平为 `X ∈ R^(B×C×N)`：

```text
mu(F)  = mean_N(X)
Xc     = X - mu(F)
Cov(F) = Xc @ Xc^T / max(N - 1, 1)
```

候选统计损失：

```text
L_mean = mean(abs(mu(S) - mu(T)))
L_cov  = mean((Cov(S) - Cov(T))^2)
L_stat = lambda_mean * L_mean + lambda_cov * L_cov
```

P0 的 cosine 不删除。第一阶段只比较下列互斥实验臂：

| 实验臂 | 特征监督 | 目的 |
| --- | --- | --- |
| B0 | 关闭 Foundation | 检测基线 |
| D0 | cosine | 当前 P0 基线 |
| D1 | statistics only | 检验统计匹配能否独立工作 |
| D2 | cosine + statistics | 检验互补性 |

一次实验只改变 loss 类型和对应权重，学生、教师、数据、epoch、batch、imgsz、seed、增强和优化器全部保持一致。

## 4. 为什么暂不加入 logit 蒸馏

当前 Foundation 教师 DINOv2/DINOv3 是图像编码器，不是 COCO 检测器，不提供与 YOLO 检测头同语义的分类/框 logits。此时声称“logit 蒸馏辅助”并不成立。

若未来加入检测 logit 蒸馏，需要额外的检测教师、类别映射、框匹配和置信度校准，这会引入第二教师和新的混杂变量。因此它不进入当前 P0/P1，必须作为独立后续课题。

标签监督始终由原 YOLO detection task loss 保证，不被统计蒸馏替代。

## 5. 单测与生效证据

新增统计损失后至少满足：

1. 输入 shape 不一致时失败关闭。
2. 相同特征的统计损失接近 0。
3. 空间 token 同步置换后，统计损失保持不变；明确记录这是设计性质而不是精度优势。
4. 教师分支 `detach`，教师参数无梯度且不进入 optimizer。
5. 学生和 student projector 梯度非零、有限。
6. `total = task + weighted_stat (+ weighted_cosine)` 数值可核对。
7. 日志记录 raw/weighted mean、cov、cosine、task ratio、峰值显存和耗时。

## 6. 判读线与负结果

- 先用固定单 batch 验证 `L_stat` 有限、可下降、梯度方向正确，只作链路证明。
- 再做相同预算、至少 3 seed 的 B0/D0/D1/D2 对照。
- 主要结论仍由 `mAP50-95` 成对差值和置信区间决定，不用最低训练 loss 或最好单 seed 代替。
- 若 D1/D2 无效，依次排查：统计量归一化、align_dim、loss/task 比例、空间信息损失、学生容量和数据规模。

## 7. 与平滑种子选择思想的关系

统计对齐回答“迁移什么结构”，平滑 token 权重回答“在哪些位置迁移”。两者不能在第一轮同时改变。正确顺序是先完成统计 loss 的独立对照，再固定最优 loss，单独研究连续权重

```text
w_i(s) = (q_i + epsilon)^s / sum_j (q_j + epsilon)^s,  s in [-1, 1]
```

其中 `q_i > 0` 是教师显著性。负 `s` 只改变非负 token 权重分布，不能直接作为 KD loss 的负系数。
