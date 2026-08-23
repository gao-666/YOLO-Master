# 已知局限与降级方案

## 当前局限

- 准入 smoke 使用一张仓库示例图和单 batch，只证明 stage、projector、loss 与梯度链路，不证明检测精度。
- DINOv2 是许可明确的准入降级教师；当前仓库正式 Foundation 训练构造器只原生支持 DINOv3/SigLIP2/multi，因此 DINOv2 smoke 通过显式协议适配器运行，不冒充正式 DINOv3 实验。
- DINOv3 官方权重需要接受专用许可并登录 Hugging Face；未授权时不得使用非官方镜像规避门禁。
- `foundation_cache_teacher_features` 在当前默认配置中仍标记为 future phase。P0 不承诺缓存训练；如后续实现，缓存键至少包含教师 repo、revision、权重 SHA-256、预处理版本、数据样本 ID 和目标 level。
- `coco8`/`coco128` 方差较大，不能据此作论文级涨点结论。
- 单卡 16 GB 需先用 P4、batch 1–4；显存不足时先减 batch/imgsz，不改变 on/off 的相对预算。

## 风险触发与降级

| 风险 | 触发条件 | 降级动作 |
| --- | --- | --- |
| DINOv3 不可访问 | 未接受许可、无 HF token 或下载失败 | 使用已声明的 DINOv2 Apache-2.0 smoke；正式 P1 暂不启动 |
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
