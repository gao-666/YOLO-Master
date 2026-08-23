# D2 P0 接手手册

这份手册只解决一个目标：你能够解释、复现和调试“YOLO-Master 学生从冻结教师学习 P4 特征”的最小闭环。暂时不要从头学习所有 YOLO 版本，也不要先碰 P1 多 seed。

## 1. 先建立一张最小心智图

```text
图像 batch
  ├─ 学生 YOLO-Master ─→ 检测预测 ─→ task loss（框、类别等）
  │                  └→ P4 feature ─┐
  └─ 冻结 DINO 教师 ─────────→ dense feature
                                      ↓
                         projector：通道和空间对齐
                                      ↓
                         KD loss：比较学生与教师
                                      ↓
             total loss = task loss + weight × KD loss
                                      ↓
                   backward：只更新学生/学生投影，不更新教师
```

你只需先掌握五个词：

1. **学生**：最终要部署的较小检测模型，本项目是 YOLO-Master-N。
2. **教师**：只在训练时提供特征目标的较大模型；参数冻结，不参加部署。
3. **P4**：YOLO 检测头前的一张中等分辨率特征图。它不是原图，也不是最终框。
4. **Projector**：通常是 1×1 卷积，把不同通道数的特征变到共同维度；同时把教师空间尺寸缩放到学生尺寸。
5. **KD loss**：衡量两份对齐特征的差异。当前 P0 用 cosine，方向越相似，loss 越小。

## 2. 源码按这个顺序读

| 顺序 | 文件 | 只需先看 | 作用 |
| --- | --- | --- | --- |
| 1 | `experiments/rhino_d2/configs/d2_p0.yaml` | 全部 | 锁定学生、教师、P4、维度、loss、权重和 seed |
| 2 | `experiments/rhino_d2/scripts/d2_p0_train_smoke.py` | `main()` | 把真实学生、教师、检测 batch、总 loss 和优化器串起来 |
| 3 | `ultralytics/nn/foundation/taps.py` | `StudentFeatureTap` | 在 YOLO forward 时抓住 Detect head 使用的 P4，保留梯度 |
| 4 | `ultralytics/nn/foundation/projectors.py` | `P4AlignmentProjector.forward` | 对齐通道、空间尺寸、设备和 dtype；教师分支 detach/frozen |
| 5 | `ultralytics/nn/foundation/losses.py` | `cosine_kd_loss` | 计算 `1 - cosine_similarity`，梯度只流向学生侧 |
| 6 | `ultralytics/nn/foundation_distill_model.py` | `loss()` | 真实集成点：抓 P4、跑教师、对齐、计算 KD，并加入检测 loss |
| 7 | `ultralytics/engine/trainer.py` | `_setup_train()` 中 Foundation 部分 | 正式 `yolo train` 如何自动包装学生模型 |

当前 P0 脚本显式注入 DINOv2 教师，是因为正式构造器当前只接受 DINOv3、SigLIP2 或 multi。这个降级必须明确声明，不能把 DINOv2 smoke 冒充成正式 DINOv3 实验。

## 3. 你需要补到什么程度

不需要先会推导整篇 YOLO 论文。按依赖顺序补下面内容即可：

### Python/PyTorch 最低线

- 会读 `shape`，知道图像/特征常用 `B,C,H,W`。
- 会区分 `model.train()`、`model.eval()`、`torch.no_grad()`、`torch.inference_mode()`。
- 知道 `loss.backward()` 生成梯度，`optimizer.step()` 更新参数。
- 知道 `detach()` 切断梯度，`requires_grad=False` 冻结参数。
- 会识别 CPU/CUDA、float32/float16 不一致报错。

### YOLO 最低线

- Backbone 提取视觉信息，Neck/FPN 融合不同尺度，Detect head 输出检测结果。
- P3/P4/P5 是不同空间尺度的特征：P3 更细，P5 更粗；P0 只做 P4 控制复杂度。
- 检测训练 batch 至少含 `img`、`cls`、`bboxes`、`batch_idx`。
- `mAP50-95` 是验证指标，不等于训练 loss；P0 跑通不代表精度提升。

### 蒸馏最低线

- 教师和学生输出维度通常不同，所以先投影/缩放，再比较。
- 教师必须冻结且不能进入 optimizer/checkpoint。
- KD loss 非零不等于生效；还必须看到它进入 total loss，并对学生/投影层产生非零梯度。
- 单 batch 下降只证明链路可优化，不证明泛化和 mAP 提升。

## 4. 一条命令复现 P0

在仓库根目录执行：

```powershell
conda activate yolo-master-d2
python experiments/rhino_d2/scripts/d2_p0_train_smoke.py --config experiments/rhino_d2/configs/d2_p0.yaml --offline
```

结果写入 `experiments/rhino_d2/results/d2_p0_train_smoke.json`。只有以下检查全部为 `true` 才算 P0 技术闭环：

- `finite_losses`
- `foundation_loss_nonzero`
- `foundation_in_total_loss`
- `student_has_gradient`
- `projector_has_gradient`
- `teacher_frozen`
- `teacher_not_in_optimizer`
- `fixed_batch_foundation_loss_decreased`
- `fixed_batch_total_loss_saw_decrease`

检测目标动态分配、BatchNorm 和路由可能让 total loss 在固定 batch 上波动。P0 不要求它逐步单调，只要求 KD loss 从初值到末值下降、KD 确实进入 total loss，并且参数更新后至少观察到一次 total loss 低于初值。

## 5. 最基础的调试顺序

不要一次盯着整段训练。每次只回答一个问题：

1. **环境对吗**：`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
2. **输入对吗**：在 `_batch()` 后检查 `batch["img"].shape` 和四个 key。
3. **P4 抓到了吗**：断点停在 `foundation_distill_model.py` 的 `student_features`，检查是四维 Tensor。
4. **教师输出对吗**：检查 `teacher_feature.shape`，并确认 `requires_grad=False`。
5. **对齐了吗**：检查 `student_aligned.shape == teacher_aligned.shape`。
6. **loss 有限且非零吗**：排除 NaN/Inf、权重为 0、分支没启用。
7. **梯度流向对吗**：学生/投影梯度应非零，教师梯度必须为空。
8. **总 loss 真包含 KD 吗**：检查 `total = task + foundation`，而不是只打印了一个未参与反传的数。

### VS Code 断点建议

按顺序打四个断点：

1. `d2_p0_train_smoke.py` 的 `loss_components, loss_items = wrapper(batch)`。
2. `foundation_distill_model.py` 的 `student_features = ...`。
3. `foundation_distill_model.py` 的 `student_aligned, teacher_aligned = projector(...)`。
4. `foundation_distill_model.py` 的 `foundation_loss = ...`。

遇到 CUDA 报错时，先临时设置环境变量 `CUDA_LAUNCH_BLOCKING=1` 再复现一次，它会把真正失败位置暴露得更准确；只用于调试，不用于正式测速。

## 6. 常见错误如何判断

| 现象 | 第一检查点 | 常见原因 |
| --- | --- | --- |
| CUDA unavailable | PyTorch 版本和驱动 | 装成 CPU 版 torch，或环境没激活 |
| Expected all tensors on same device | 两份 feature 的 `.device` | 教师在 CPU、学生在 CUDA |
| 通道/shape 不同 | projector 前后 shape | align_dim 配错，教师 token 没还原成网格 |
| No feature captured | `StudentFeatureTap` 源层 | forward 尚未执行，或选错 Detect head 来源 |
| KD loss 一直为 0 | 开关和 loss weight | `foundation_enabled=False` 或 weight=0 |
| 教师出现梯度 | detach/freeze/optimizer 参数 | 教师未冻结，属于红线失败 |
| loss 是 NaN/Inf | 输入、学习率、归一化 | 学习率过大、半精度不稳、数据异常 |
| smoke 通过但 mAP 不涨 | 不是 P0 故障 | 进入 P1 后分析容量、stage、loss 比例和数据 |

## 7. 你在会上应该能回答的五句话

1. “P0 只蒸馏 P4，因为单 stage 最容易控制变量和定位问题。”
2. “教师是冻结的，且不在 optimizer；训练和部署都只保留学生。”
3. “Projector 解决通道数和空间网格不一致，当前共同维度是 64。”
4. “总损失由检测 task loss 与加权 KD loss 组成，证据包含非零 loss 和非零学生梯度。”
5. “单 batch loss 下降只证明链路跑通；是否涨 mAP 必须由 P1 同预算多 seed 对照回答。”
