# D2 DINOv3 P0 准入证据

## 结论

DINOv3 ViT-S/16 与 ViT-L/16 的 BF16 单 stage P4 蒸馏 P0 均通过。该结论只证明本地 Teacher 资产、dtype、shape、投影、检测任务损失与 KD 梯度链可运行，不产生 mAP 或 Teacher capacity 结论。

| Gate | ViT-S/16 | ViT-L/16 |
| --- | --- | --- |
| Teacher BF16 finite | `[2,384,14,14]`，通过 | `[2,1024,14,14]`，通过 |
| Student P4 | `[2,128,14,14]` | `[2,128,14,14]` |
| 对齐后 | 双方 `[2,64,14,14]` | 双方 `[2,64,14,14]` |
| 空间 resize | 无 | 无 |
| Teacher frozen / optimizer 隔离 | 通过 | 通过 |
| Student / projector gradient | 非零且有限 | 非零且有限 |
| KD 进入 total loss | 通过 | 通过 |
| Fixed-batch KD | `0.09785→0.09156` | `0.09852→0.09682` |

S 的 5 步 total task loss 没有低于起点，作为非阻断观察保留。真实 YOLO detection assignment、BatchNorm 和路由会造成短步波动；冻结协议要求的是 KD finite、非零、进入总损失并在 fixed batch 下降，没有事后增加“总 task loss 五步必须下降”的红线。

## 证据

- Teacher 资产登记：[`env/dinov3_teacher_manifest.json`](env/dinov3_teacher_manifest.json)
- S dtype：[`results/d2_v3_teacher_vits16_bf16.json`](results/d2_v3_teacher_vits16_bf16.json)
- L dtype：[`results/d2_v3_teacher_vitl16_bf16.json`](results/d2_v3_teacher_vitl16_bf16.json)
- P0-S：[`results/d2_v3_p0_vits16.json`](results/d2_v3_p0_vits16.json)
- P0-L：[`results/d2_v3_p0_vitl16.json`](results/d2_v3_p0_vitl16.json)

## 复现命令

```powershell
$env:YOLO_CONFIG_DIR="E:\2026YOLO\YOLO-Master\runs\rhino_d2\config"
$env:HF_HUB_OFFLINE="1"
python experiments/rhino_d2/scripts/d2_v3_teacher_smoke.py --config experiments/rhino_d2/configs/d2_v3_p0_vits16.yaml
python experiments/rhino_d2/scripts/d2_v3_teacher_smoke.py --config experiments/rhino_d2/configs/d2_v3_p0_vitl16.yaml
python experiments/rhino_d2/scripts/d2_v3_p0_train_smoke.py --config experiments/rhino_d2/configs/d2_v3_p0_vits16.yaml
python experiments/rhino_d2/scripts/d2_v3_p0_train_smoke.py --config experiments/rhino_d2/configs/d2_v3_p0_vitl16.yaml
```
