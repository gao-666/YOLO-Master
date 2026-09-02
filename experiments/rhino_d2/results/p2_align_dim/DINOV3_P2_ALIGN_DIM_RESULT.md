# DINOv3-S P2-02 align_dim 64→128 结果

正式结论：**No support for the align-dimension bottleneck hypothesis**。

## 预注册性能判读

- 主比较 ON128−ON64：mean `-0.001320`，paired 95% CI `[-0.020892, 0.018252]`，判定 **no_support**。
- 次级 ON128−OFF：mean `-0.003490`，paired 95% CI `[-0.027199, 0.020219]`，判定 **inconclusive**。

三个 seed 的 ON128−ON64 分别为 `+0.006105 / -0.009585 / -0.000480`。按协议不继续搜索 256/512。

## 同图像梯度复测

- late cosine 配对中位变化：`0.001638`，95% CI `[-0.001329, 0.004544]`：没有可检测方向变化。
- late weighted norm-ratio 配对中位变化：`-0.001206`，95% CI `[-0.001609, -0.001059]`：ON128 的相对 KD 梯度更弱。

## 冻结边界

P1 的 No-Go 与 P2-01 的 Inconclusive 均保持不变。P2-02 只排除当前协议下的 64→128 单变量解释；它不证明投影维度在其他数据、Teacher 或训练协议下永远无关。梯度复测是一阶局部机制证据，不证明信息保真或因果关系。下一研究假设可转向 Static–Response Gap / Response-Field。
