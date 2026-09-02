# D2 DINOv3-S P2-01 梯度冲突诊断协议

## 研究边界

P1 冻结节点为 `931f9dbcf1e6880bc2832102e127b08f34881f11`。本协议只诊断 P1 已训练模型，不训练新模型，不修改
`DINOV3_P1_GO_NO_GO.md`，也不把后续结果追溯改写为 P1 的新结论。

主假设为：DINOv3 P4 cosine 特征匹配可以被优化，但其局部梯度可能与检测任务梯度冲突。对同一 P4 特征张量
`F_P4` 计算：

```text
g_task = d L_detect / d F_P4
g_KD   = d (0.15 * batch_size * L_cosine) / d F_P4
```

主判别量为逐图像 `cos(g_task, g_KD)`；辅助判别量为 `||g_KD|| / ||g_task||`。余弦小于 0 表示两个训练目标的
局部下降方向相互竞争，余弦接近 0 表示基本无关，余弦大于 0 表示局部方向一致。范数比只解释强度，不能替代方向判定。

## 冻结输入

- 只使用 P1 ON 的 seed `20260824 / 20260825 / 20260826`。
- 固定 EMA checkpoint：early=`epoch9.pt`、middle=`epoch24.pt`、late=`epoch49.pt`，分别对应完成 10、25、50 个 epoch。
- 数据只来自 P1 的 2048 张训练集。按字符串种子 `dinov3-p2-gradient-conflict-v1-20260902` 对规范化相对路径做
  SHA-256 排序，无放回取前 64 张；在计算梯度前保存图像清单和哈希。
- 不读取 P1 validation 图像，不使用 validation AP 调节诊断设置。
- 输入为 256 像素、batch 4、无随机增强的确定性 letterbox 预处理。
- Student 检测头保持训练前向语义；所有 BatchNorm 层切换到 eval，防止 64 张诊断图改变 checkpoint 的运行统计。
- Teacher 仍为本地 DINOv3-S/16 BF16 资产；P4 projector 使用各 checkpoint 内保存的参数。

## 统计与否证条件

每个 seed、每个时点报告：有效图像数、余弦中位数、均值、负余弦比例、余弦中位数的 95% bootstrap CI、范数比中位数及
95% bootstrap CI。late 时点还按图像聚类：先对同一图像的三个 seed 求平均，再对 64 张图像 bootstrap。

预先冻结的判定为：

- **支持梯度冲突解释**：pooled-late 余弦中位数 CI 上界小于 0，且至少两个 seed 的 late 中位数 CI 上界小于 0。
- **不支持梯度冲突解释**：pooled-late 余弦中位数 CI 下界大于或等于 0。
- 其余结果为 **inconclusive**，不据此直接进入新 loss、ViT-L、multi-stage 或 `align_dim=128`。

三个 seed 的 late 余弦中位数与 P1 `ON-OFF` 检测差值只做描述性 Spearman 对照；`n=3` 不作显著性或因果声明。

## 结论限制

该探针衡量固定 checkpoint、固定数据和固定 P4 张量处的一阶局部关系。它不能证明全训练轨迹的因果机制，也不能替代新协议下的
干预实验。无论结果为支持、不支持或不确定，P1 原协议仍保持 No-Go。
