# P2-05｜Response Gradient Geometry Probe

【1. 当前阶段】

独立 P2 机制诊断。P2-04 终点为 `769d1b66c96d5ff444fa4f9e4d4ec9961dd989b5`，Calibration Failed，selected_alpha=null。
本协议依据用户 2026-09-08 指示执行 CPU invariant → 三 checkpoint train64 probe，0 training runs；不校准任何新 alpha。
协议先独立提交，然后实现、CPU 检查，再执行 GPU 测量。新结果仅本地归档，远程发布另行决定。

【2. 已知事实 / Repository Audit】

Current repo already has:
- `ultralytics/engine/trainer.py` 的 `_setup_train` 调用 `build_foundation_distillation_wrapper`；训练 loop 调用 wrapper.loss、loss.sum、scaler.scale(...).backward，并收集 Foundation metrics。
- `FoundationDistillationModel.loss`：student → tap → teacher.encode → projector → static KD；`kd * effective_weight * batch_size` 拼接进真实 task loss。
- `StudentFeatureTap._find_detect_sources` 按 Detect.f=[16,19,22] 选择 P4：实际第19层 neck/FPN，非 backbone 第6层。不得改 tap 来让术语成立。
- `P4AlignmentProjector.forward` 为 student Conv+BN、frozen teacher projection、teacher-only bilinear resize；aligned Z 在 projector 之后。
- `response.py` 已有八扰动、epoch-major-v1、train BN rollback、FP32 strict cosine、detach 与 zero-norm fail-closed。
- `response_field_kd_loss` 目前由 smoke/calibration 调用；正式 Foundation wrapper 未调用它。A/B/C formal runner 并未因配置或文档存在而自动实现。
- 旧 calibration `alpha=.25` pooled r_loss=.848179、r_grad=4.259234，三 seed r_grad>4。只证明冻结信号预算匹配失败。

This experiment reuses: `calibrate_response_field_alpha.py` 的输入哈希、checkpoint 校验、teacher cache、完整 state/buffer 恢复和 CSV writer；现有 teacher、tap、projector、loss、paired view。
This experiment adds/modifies: 一个只测量的脚本（含 summarizer）、独立协议/配置、CPU tests 和本次结果；对现有模块不作实验性修改。
This experiment explicitly does NOT reimplement: teacher、projector、KD wrapper、training runner、perturbation、optimizer、loss。

真实 probe 链：冻结 EMA → train64/八条件 → student neck P4 → 共享 projector → static/response loss → autograd.grad(Z与F) → 几何分解 → buffer恢复 → CSV/CI。
本次不进入 trainer/backward/optimizer/validation 链。

【3. 当前未知问题】

Response 相对于 perturbed-static 的 P4 梯度放大，主要出现在 loss 的几何中，还是 projector 的梯度传递中？

【4. 主假设】

H1：控制夹角、token平均和双支计数后，小 response norm 的几何贡献在冻结数据上占主要正向放大，且不由单个 seed/family 独占。
数学公式吻合本身是实现恒等式检查，不能单独作为 H1 的经验证据。

令 N=B*H*W，r=Zpert-Zclean，c=cos(r,t)，u=r/||r||，v=t/||t||。
`dL/dr=(c*u-v)/(N*||r||)`；每 token norm=`sin(theta)/(N*||r||)`。
完整 response 梯度在 Z_pair 上为 `(-dL/dr,+dL/dr)`，norm 恰为单支的 sqrt(2) 倍。
Static 比较对象明确为 **第二项 perturbed-static**：在 Z_pair 上 `(0,dL_static_pert/dZpert)`。
另存 clean-static 和两项 Static-2V 的 norm/cross term 作与 P2-04 的解释桥梁，不混用分母。

定义 R=||dLresp/dr||，S=||dLstatic_pert/dZpert||，U=sqrt(sum_token(sin_resp^2/||Zpert||^2))/N。
G_norm=R/U（固定 response 角度，只代换分母 norm）；G_angle=U/S。
A_Z=||grad_Zpair Lresp||/S = sqrt(2)*G_norm*G_angle。
A_F=||grad_Fpair Lresp||/||grad_Fpair Lstatic_pert||；J=A_F/A_Z。
所以 `A_F=sqrt(2)*G_norm*G_angle*J`；必须逐 observation 验证。
J 只测 projector 的相对梯度传递，含 train-BN batch coupling，不是 backbone Jacobian/参数更新。
G_norm 与 G_angle 是代数分解，顺序已冻结；不能把它当成对真实数据/参数的因果干预。

【5. 竞争性解释】

H2：J 贡献主导；H3：sqrt(2) 双支计数解释；H4：family/seed 集中；H5：mean、分母、cross term、数值/路径问题；此外 G_angle 是与 norm 独立列出的竞争因素。
P2-01 Inconclusive 不等于排除冲突；P2-02 No support 只削弱本次64→128解释；P2-03 不证明 mismatch 导致检测退化。

【6. 最小实验】

沿用旧 calibration 全部模型/数据/扰动设置（配置 SHA 在本协议配套 YAML 固定），不做参数干预。
measurement_space={aligned-Z,Student-neck-P4}，loss_family={perturbed-static,response}；384 batch-condition observations。
保留每 token norm、cosine、sin、pred/observed norm 的分位摘要；梯度观测是完整 batch，不伪称逐图独立梯度。
同一 forward 同时取 Z 和 F 梯度。task loss 可按旧路径只算 clean 一次（确保调用状态一致），不作为选择信号；所有 buffers 在 finally 恢复。

【7. 最便宜前置 Probe】

CPU 固定角度、norm={1,.5,.25,.125}：loss不变、梯度反比；验证 mean N、双支sqrt(2)、teacher detach、非有限/zero-norm拒绝、J与分解。
CPU 不通过即止；真实 probe 无 optimizer/scheduler/model EMA update。task-loss EMA临时状态必须回滚，不能报告从未发生过该临时写入。

【8. 关键观测量】

记录 norm(clean/pert/r/teacher-r)、cos/sin/static/response、pred/obs gradient、A_Z、A_Z/sqrt(2)、A_F、J、G_norm、G_angle、clean-static cross term。
原始向量逐点 allclose（rtol=2e-4,atol=2e-7），整体 relative-L2<=2e-4；公式/双支/分解失败属于 technical_invalid，不给机制标签。
主要统计量 `D=log(G_norm)-[max(log(G_angle),0)+log(sqrt(2))+max(log(J),0)]`：norm贡献是否大于其他正向贡献之和。
这是本研究预先定义的“主要”操作性标准，不是学界通用阈值。
报告 mean、sample SD、median、IQR、正负比例、95% percentile bootstrap CI；pooled、每seed、每family、每condition、leave-one-family-out。
train BN跨图耦合：按16个固定 batch block bootstrap，重采样时保留该block的全部seed/condition；10000次，seed=20260908。
CI仅条件于这三个checkpoint、固定batch组成与已使用过的train64，不能当未见图像泛化或训练seed总体CI。

【9. 反证条件】

公式不符先技术排错；Z空间去双支后不放大、D<0或只在一个family成立，削弱“小norm广泛主导”。小norm并不充分：sin(theta)可趋近0；不把恒等式当相关性发现。

【10. Go / No-Go / Ambiguous】

所有技术恒等式通过后：
- H1 Support：pooled D的CI下界>0，至少2/3 seed且至少3/4 family的D下界>0，并且四个leave-one-family-out的A_Z/sqrt(2) CI下界均>1。
- H1 No-Support：pooled D的CI上界<0，或去双支A_Z的CI上界<=1。
- 其余 Ambiguous。No-Support/Support只是这一“广泛主导”命题的操作性标签，不否定恒等式/方法。
本次任意结果都不授权 smaller-alpha、normalization 或正式训练。

【11. No-confound 审计】

Planned：配置继承P2-04同一3seed/epoch49 EMA、train64 SHA、8条件、BF16教师、FP32学生/loss、batch4、imgsz256、P4(19)、align64、epoch49/16batch。
Resolved：核验checkpoint/file/state前后hash、实际tap index、dtype、BN flags、teacher frozen、输入tensor digest、数据图像/标签hash、.grad=None、forward计数。
Axis：无alpha选择；同一raw图取Z/F和两种第二项梯度；不新增loss。文件读取守卫拒绝validation/response128/formal结果目录，并记录所有本地数据读取路径；对原生库读取用显式图像清单/digest约束，不能声称Python审计钩子覆盖OS全部I/O。
旧 calibration 的 no_validation_access 是声明+源代码证据，并非操作系统级访问追踪；旧逻辑对False与0混用的总check断言并不可靠。本次用逐项明确布尔/计数检查，旧结果保持原样。

【12. 成本】

0训练run；3冻结checkpoint，64*8*3 image-conditions。旧完整calibration约168秒（log），增加Z梯度/统计后估计3–10分钟GPU墙钟（0.05–0.17 GPU小时）。
先CPU测试，再以首个batch记录真实耗时；预算超过15分钟技术中止并归档，不换seed/减少条件。

【13. Claim Boundary】

最多说明这三个checkpoint/固定train64八扰动下的局部梯度放大分解及其一致性。
不声明causal detection failure、方法有效、alpha<.25可行、normalization有效、未知corruption泛化或backbone参数更新。

【14. 证据文件】

`results/p2_response_gradient_geometry/`：raw.csv、summary.csv、result.json、manifest.json、PNG图、probe.log、CPU invariant结果。
raw/summary/result文件使用`response_gradient_geometry_`前缀；保留协议commit、执行commit、证据commit的区别，产物hash、完整input manifest、环境、脚本hash、声明与实际执行身份。
脚本名称`probe_response_gradient_geometry.py`回答“放大出现在哪个空间、由哪个因素贡献”；summary函数回答“这种贡献是否跨seed/family一致”。

【15. 下一刀】

读本次分解后只选择一个后续机制问题。本次不生成/执行更小alpha或新方法；P2-04始终停止。
