# 已知局限与降级方案

## 当前局限

- 准入 smoke 使用一张仓库示例图和单 batch，只证明 stage、projector、loss 与梯度链路，不证明检测精度。
- DINOv2 已由正式 Foundation 构造器原生支持，并锁定模型 revision 与显式 dense 预处理契约；P0/P1 共用同一教师实现。
- DINOv3 受控访问申请已被拒绝，因此不进入 P1；不得使用非官方镜像规避门禁。
- `foundation_cache_teacher_features` 在当前默认配置中仍标记为 future phase。P0 不承诺缓存训练；如后续实现，缓存键至少包含教师 repo、revision、权重 SHA-256、预处理版本、数据样本 ID 和目标 level。
- `coco8`/`coco128` 方差较大，不能据此作论文级涨点结论。
- 单卡 16 GB 需先用 P4、batch 1–4；显存不足时先减 batch/imgsz，不改变 on/off 的相对预算。
- `environment.json` 会诚实记录整个仓库的 dirty 状态；本次抓取时因生成中的结果/文档及无关未跟踪文件而为 `true`，但单独审计的 `experiment_inputs_state.dirty=false`，且配置、脚本、测试、实现均有 SHA-256。评审实验身份时应以 experiment commit、输入 dirty 状态和哈希三者共同判断。

## 风险触发与降级

| 风险 | 触发条件 | 降级动作 |
| --- | --- | --- |
| DINOv3 不可访问 | 受控访问申请已被拒绝 | 正式切换并锁定 DINOv2 Apache-2.0 主路线；不再阻塞 P1 |
| 显存不足 | CUDA OOM | 单 stage P4；batch 同时降到 1；on/off 保持相同配置 |
| loss 不下降 | 20 step 后最终 loss 不低于初始 | 先检查梯度/shape/投影；降为 cosine-only；保留失败 JSON |
| 指标噪声 | 三 seed 区间包含 0 | 判定 inconclusive/no-go；增加 seed，不移动 0.3pp 判读线 |
| 进度不足 | 8.31 前多 stage 未闭环 | 保留单 stage P4，不扩 P3/P5、router 或 semantic KD |
| 缓存失效 | 教师 revision/hash、预处理或数据版本变化 | 禁止复用旧缓存，重新生成并记录 manifest |

## 安全边界

- 脚本不接收或执行任意 shell 字符串。
- 结果、缓存和环境记录限定在本实验目录与仓库 `runs/` 下。
- 环境记录不写出 token、密码、Authorization header 或完整环境变量。
- 教师权重和数据不提交 Git，只提交 revision、许可来源、文件 hash 和生成命令。
