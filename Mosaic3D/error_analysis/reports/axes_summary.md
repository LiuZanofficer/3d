# TASK3 - Axes accounting (ScanNet200, annotation-free)

## Mask provenance (Segment3D, not GT)
`batch["masks_binary"]` / dumped `pred_masks` are **Segment3D** class-agnostic proposals
(`/.../scannet_masks/segment3d/point_indices.npz`, fields `packed`/`lengths`/`scores`), NOT GT instances:
- mask count != GT-instance count in **302/312** val scenes (13165 proposal masks vs 10044 GT instances);
- each mask carries a class-agnostic `pred_mask_scores` (float32, e.g. [0.008, 0.66]), no GT labels.
So mask-based instance consistency is annotation-free.

## What each axis is
- **baseline**: per-point foreground argmax of `feat·text` → point fg-mIoU **0.15475** (self-checked口径).
- **+mask_text_vote (Method A)**: pool per-point prob within each Segment3D mask, argmax, paint mask-consistently.
  This is an **instance-consistency** axis (denoising scattered point noise), NOT a direction fix.
- **+direction-fix (inference)**: annotation-free re-orientation of the text readout at eval time
  (anchor_decorrelate / caption_proto / unsup text-naming). TASK2 result: **no gain** (all <= baseline).
- **+direction-fix (training = Method B)**: hard-negative caption loss retrains the visual<->text alignment. Pending Phase-4 eval.

## Combination-axis table (fg-mIoU; mAP)
| axis | fg-mIoU | Δ vs baseline | mAP | source |
|---|---:|---:|---:|---|
| baseline | 0.15475 | — | 0.11486 | live eval + offline self-check (OK) |
| + mask_text_vote (consistency) | 0.17573 | +0.02098 | 0.11236 (-0.00250) | prior live eval (established) |
| + direction-fix (inference) | ~0.15475 | +0.000 | — | TASK2 refuted (best annotation-free = baseline) |
| + mask_text_vote + direction-fix (inference) | ~0.17573 | +0.02098 | — | inference fix adds nothing on top of consistency |
| + direction-fix (training, Method B) | TBD | TBD | ~0.115 target | Phase-4 eval (B ckpt) |
| + mask_text_vote + Method B | TBD | TBD | TBD | Phase-4 eval |

## Gain decomposition
- The only realized gain so far is **instance-consistency** (+0.02098 from mask_text_vote), which is orthogonal
  to naming: it removes scattered per-point noise but keeps each mask's (possibly wrong) sibling name.
- **Inference-time direction-fix contributes 0** (TASK2): the text anchors are collinear and the separable
  structure cannot be named without labels.
- Remaining head-room is bounded by naming ceilings, all annotation-free-unreachable:
  Task0 real-labelspace oracle top25=0.1906 / top50=0.2118 / all=0.3257; within-tight-sibling naming ceiling
  (oracle_incluster) instance-renaming fg-mIoU=0.2311. Method B must convert part of this naming head-room.

## Reading
mask_text_vote is a baseline/consistency axis and must not be double-counted as a "fix". The scientific
question for Method B is whether *trained* direction correction adds fg-mIoU **on top of** the 0.17573
consistency axis, and whether the gain concentrates on the sibling pairs identified in TASK2.

---

# 交接任务追加：判别性文本锚点(A) + caption 命名(B)｜ScanNet200｜annotation-free

## 组合轴表（本任务新增，全部可部署 / qz baseline ckpt；口径自检 name+base=0.15475 逐位复现）
| axis | fg-mIoU | Δ vs baseline | Δ vs base+MTV | mAP | 备注 |
|---|---:|---:|---:|---:|---|
| baseline (name+base) | 0.15475 | — | — | 0.1149 | 自检 OK |
| + attr_mean 锚点 (baseline 读出) | 0.14557 | −0.0092 | — | 0.1214 | 纯属性**有害** |
| + fuse0.7 锚点 (baseline 读出) | 0.15891 | +0.0042 | — | 0.1250 | 非机制小增益 |
| + mask_text_vote (一致性轴) | 0.17573 | +0.02098 | 0 | 0.1124 | 既有 |
| + MTV + fuse0.5 锚点 | 0.18005 | +0.0253 | +0.0043 | 0.1241 | 非机制 |
| **+ MTV + fuse0.7 锚点** | **0.18231** | **+0.0276** | **+0.0066** | 0.1225 | 最优可部署；但增益**弥散、不落同辈簇** |

## 命名信号阶梯（sibling 命名 acc；tight 簇 44 类）
| 命名法 | sibling acc | 部署性 | 说明 |
|---|---:|---|---|
| baseline pred_majority（部署读出，val） | 0.3822 | 可部署 | 自检≈TASK2 0.379 |
| unsup_textname（文本锚点，val） | 0.377 | 可部署 | TASK2 负结果 |
| 属性锚点 attr/fuse（重读，val 上界） | 0.69–0.76 | 可部署 | **≤ 类名重读 0.792**，无增益 |
| **caption 类名众数词（train，上限）** | **0.7311** | **不可部署(val 无 caption)** | ≫ 文本锚点，信号在**词**里 |

## 增益分解（fuse0.7+MTV vs name+MTV，逐类）
- 同辈簇 dMeanIoU **+0.0033**，非同辈 **+0.0075** → 增益**不集中在同辈簇**（非同辈更大）。
- 属于锚点扰动的净正集成/去噪，**非命名修复**；伴随明显附带损伤（washing machine −0.54、table −0.16、pillar/shower wall 同辈受损）。

## 三问最终回答（命名死结是否被打开）
- **(a) 判别性锚点让近义锚点去共线？否。** 簇内 |cos| 0.9194→0.9101（不显著）。CLIP 文本编码器把「刻意写区别」的近义描述仍映射到近共线嵌入。
- **(b) 把 cos(d_vis,d_txt) 拉起来？否。** 0.1108→0.1136（未抬升）。读出方向仍与视觉判别方向近正交。
- **(c) 在同辈簇上把命名 acc/fg-mIoU 抬到一致性轴之上？**
  - 属性锚点：**命名 acc 否**（≤ 类名重读）；fg-mIoU 虽 0.18231>0.17573 但**增益不在同辈簇、不来自命名**（非机制）。
  - caption 词命名：**命名 acc 是**（0.7311 ≫ 0.379），但**只在 train 可得、val 不可部署**。
- **裁定：命名死结未被「属性文本锚点」打开（本任务核心假设被否）。** 但 TASK B 定位到真正的命名信号——
  **caption 的离散类名词**（≈2× 文本锚点）。Method B/caption_proto 之所以失败，是因为它们对齐/读出 caption 的 **CLIP 文本嵌入**（近义仍共线），
  而信号在**词**里被 CLIP 抹平。

## 下一步（TASK C 的正确形态，待定/需算力与确认）
- **不要**照搬 TASK C 原文（对齐 A 的共线属性锚点）——那等于 Method B 覆辙。
- 推荐：把「caption 类名众数词」当**离散命名目标**，frozen backbone 训一个轻量分类/命名头（annotation-free 词监督），
  或先做**零训练可部署验证**：用 caption 词把 train 实例分组→求各类**视觉原型**→val 用视觉特征最近原型命名（部署侧不需 caption）。
- 判据仍按四条：B' 或 B'+MTV fg-mIoU>0.17573、cos(d_vis,d_txt)↑、增益落同辈簇、mAP 维持、且不打坏 baseline 读出(<0.155)。

## TASK C 可行性说明（本次已探明，未跑训练）
- 试图做「零训练可部署原型」验证（caption 词→train 实例视觉原型→val 最近原型命名），但：
  - 实例特征 dump 路径 `_update_instance_segmentation_metrics` **依赖 `masks_binary`（Segment3D masks）**；
    `scannet200_masks` **只有 312 个 val 场景，无 train masks** → 无法在 train 上走 dump 拿到 GT-实例池化视觉特征。
  - `MOSAIC3D_FORCE_EVAL=1`（已加，env 门控 `is_train`）可让 train split 走 eval 行为（加载 GT segment/instance、跳过 caption），
    但仍受制于上面的 masks 依赖；只能 dump 到 `eval_visual_means`（按 **GT 类**池化=oracle，非 annotation-free 命名可用）。
- 结论：可部署地利用 caption 词信号，**必须**要么（a）自写轻量前向脚本在 train 的 **caption 点组**上池化 clip_feat 建原型（annotation-free），
  要么（b）走 TASK C 轻量训练（frozen backbone + 词监督头）。二者都超出「纯推理零成本」范围，需单独排期。
- 建议优先 (a)：最接近零训练、且直接检验 caption 词→视觉原型能否迁移到 val；(a) 有效再上 (b) 放大。
