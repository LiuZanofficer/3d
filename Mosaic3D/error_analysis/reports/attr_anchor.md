# TASK A — 判别性文本锚点 (attribute-enriched anchors)｜ScanNet200｜annotation-free

**一句话结论**：核心假设（用判别性属性描述替换/融合类名锚点 → 锚点去共线 → 打开命名死结）**被否**。
CLIP 文本编码器把「刻意写出可见区别」的近义类描述仍映射到近共线嵌入（簇内 |cos| 0.9194→0.9101，几乎不降），
cos(d_vis,d_txt) 也没抬起来（0.111→0.114）。属性锚点单独用**有害**；只有 `fuse0.7`（0.7 类名 + 0.3 属性均值）
带来一个**小的、非机制性**的可部署增益（`fuse0.7+MTV` fg-mIoU **0.18231 > 0.17573**），但增益**不落在同辈簇**（非同辈反而更大）
且伴随明显附带损伤，属于锚点集成/去噪，**不是命名修复**。→ 按预注册顺序转 TASK B。

## 口径自检（铁律）
- 在线 `name(重编码)+baseline` 注入 → fg-mIoU **0.15475207**，与 baseline 逐位一致 → 锚点注入端到端正确。
- 重编码类名锚点 vs `reports/text_embeddings.npz`：mean cos = **1.0000**（编码管线与 eval 一致）。
- 离线 `name` 锚点 instance-renaming fg-mIoU(full) = **0.2009** ≈ TASK2 baseline 实例读出 **0.1999** ✓。
- 部署 `pred_majority` sibling acc = **0.3822** ≈ TASK2 报告 **0.379** ✓。

## A1 锚点生成（annotation-free）
- 由 LLM 世界知识为 CLASS_LABELS_200 每类写 **K=5** 条判别性视觉描述（明确写出与近义类的可见区别，如
  `office chair: swivel base with wheels, mesh backrest, armrests` vs `chair: four fixed legs, no wheels`）。仅用类名，不碰任何测试数据/GT。
- 存 `reports/attr_descriptions.json`（1000 条）与 `reports/attr_anchors.npz`
  （`emb_all`[1000,768] 逐描述、`mean_emb`[200,768] 属性均值、`name_emb`[200,768] 类名锚点，同一 recap CLIP 编码）。

## A2 锚点构造扫描（离线，GT-实例池化 = oracle 上限诊断）
7769 实例；tight 近义簇 17 个（thr0.90/cap8）覆盖 44 fg 类、1099 实例。

| 变体 | full inst-acc | sibNaming acc(限簇) | miou_rename_full | miou_rename_incluster |
|---|---:|---:|---:|---:|
| name (baseline) | 0.4521 | 0.7916 | 0.2009 | 0.2489 |
| attr_mean | 0.3707 | 0.6888 | 0.1915 | 0.2280 |
| attr_single | 0.3343 | 0.7234 | 0.1767 | 0.2154 |
| attr_maxsim | 0.3506 | 0.7025 | 0.1808 | 0.2162 |
| fuse0.3 | 0.4042 | 0.7161 | 0.2067 | 0.2502 |
| fuse0.5 | 0.4244 | 0.7352 | 0.2101 | 0.2580 |
| fuse0.7 | 0.4423 | 0.7598 | 0.2114 | 0.2634 |

- 关键：任何属性变体的 sibNaming acc 都 ≤ name(0.7916)（纯属性 0.69，最好 fuse0.7 也只 0.76）。属性锚点没让命名变好。
- fuse0.7 的 miou 上界略升（+0.010~0.015），来自「主要是类名 + 少量属性」的轻微集成，非命名修复。

## A3 机制验证（直接呼应 Step0）
- 近义簇内锚点两两 |cos|：name 0.9194 → attr_mean 0.9101（Δ−0.009，不显著）。逐簇见 `task2/attr_offline.json`。
- cos(d_vis, d_txt)（同辈对，d_vis 用 GT 类均值仅分析）：name 0.1108 → attr_mean 0.1136（没抬升）。
- 判据① 不成立：锚点没去共线、方向没对齐。机制层面证否——CLIP 文本编码器的固有性质，判别性措辞无法改变近义嵌入的近共线。

## A4 在线可部署评测（qz baseline ckpt，MTV 用 masks_binary=Segment3D）
| 配置 | fg-mIoU | head | common | tail | mAP | mAcc | Δ vs baseline | Δ vs base+MTV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (name+base) | 0.15475 | 0.333 | 0.100 | 0.026 | 0.1149 | 0.281 | — | — |
| attr_mean+base | 0.14557 | 0.310 | 0.099 | 0.022 | 0.1214 | 0.292 | −0.0092 | — |
| fuse0.7+base | 0.15891 | 0.339 | 0.105 | 0.027 | 0.1250 | 0.297 | +0.0042 | — |
| name+MTV (= base+MTV) | 0.17573 | — | — | — | 0.1124 | — | +0.02098 | 0 |
| fuse0.5+MTV | 0.18005 | 0.369 | 0.126 | 0.038 | 0.1241 | 0.311 | +0.0253 | +0.0043 |
| fuse0.7+MTV | 0.18231 | 0.379 | 0.128 | 0.034 | 0.1225 | 0.309 | +0.0276 | +0.0066 |

### 逐类 / 同辈簇拆解（fuse0.7+MTV）
- vs name+MTV：同辈簇 dMeanIoU=+0.0033，非同辈 dMeanIoU=+0.0075 → 增益不集中在同辈簇，非同辈反而更大。
- vs baseline：同辈 +0.0214，非同辈 +0.0293。
- Top gainers（多为非同辈）：copier +0.40(sib)、cart +0.35、blackboard +0.29、bathroom vanity +0.28、whiteboard +0.24、stove +0.16。
- Top losers（含同辈受损）：washing machine −0.54、pillar −0.20(sib)、table −0.16、shower wall −0.16(sib)、bathroom cabinet −0.14。
- 结论：增益弥散、伴随明显附带损伤 → 锚点扰动的净正集成效应，不是同辈簇的命名修复。明细见 `attr_online/per_class_breakdown.json`。

## 预注册判据对号
- 成立（sibNaming 显著>0.379 且 fg-mIoU>0.17573 且增益落同辈簇）：否（命名 acc 未升、机制不成立、增益不在同辈簇）。
- 部分成立（cos 改善且命名 acc 升但 fg-mIoU 未超）：否（cos 未改善）。
- 失败（cos 降但命名/fg-mIoU 不动）：接近（cos 未降、命名 acc 未升）；例外是 fg-mIoU 因非机制集成小幅上移。
- 裁定：TASK A 核心假设被否（命名死结未被属性锚点打开）。fuse0.7 作为便宜的非机制 add-on 可保留（一致性轴上再 +0.0066），但不计作命名修复。→ 转 TASK B。

## 三问回答
- (a) 判别性锚点让近义锚点去共线？否（0.9194→0.9101）。
- (b) 把 cos(d_vis,d_txt) 拉起来？否（0.1108→0.1136）。
- (c) 在同辈簇上把命名 acc/fg-mIoU 抬到一致性轴之上？否（就命名而言）：fg-mIoU 虽 0.18231>0.17573，但增益不在同辈簇、不来自命名，且 sibNaming acc 未升。→ 命名死结未打开。

## 复现命令（容器内，PROJECT_ROOT=/root/Mosaic3D_work，seed 默认）
```bash
# A1 生成锚点
CUDA_VISIBLE_DEVICES=0 python scripts/build_attr_anchors.py
# A2/A3/A4-离线
python scripts/analyze_attr_anchors.py            # -> task2/attr_offline.json
# 变体锚点文件 -> error_analysis/reports/anchor_variants/anchors_{name,attr_mean,fuse0.3/0.5/0.7}.npz
# A4-在线（每档单卡；MOSAIC3D_ANCHOR_NPZ 注入 emb_target，仅当行数==该 postfix 类数才生效）
CKPT=/workspace/Mosaic3D/qz/sc+ar+sc++.ckpt
MOSAIC3D_ANCHOR_NPZ=.../anchors_fuse0.7.npz MOSAIC3D_READOUT_MODE=mask_text_vote \
  python src/eval.py experiment=train_spunet_multidata_ppt data=sc+ar+sc++ +ckpt_strict=false ckpt_path=$CKPT trainer.devices=1
# 逐类拆解（带 MOSAIC3D_DUMP_EVAL=1 重跑 name+MTV 与 fuse0.7+MTV 后）
python scripts/score_per_class.py                 # -> attr_online/per_class_breakdown.json
```
