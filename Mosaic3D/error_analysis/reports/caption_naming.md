# TASK B — caption 类名众数命名（无监督簇命名的正确 caption 用法）｜ScanNet200｜annotation-free

**一句话结论**：命名信号**存在，且在 caption 的离散词汇里**。用 gsam2+seem caption 的类名众数词命名近义簇内实例，
sibling 命名 acc = **0.7311**（已命名中 0.8094），是文本锚点基线（unsup_textname 0.377 / baseline pred_majority 0.379）的
**约 2 倍**，判据**满足**。这与 Method B / caption_proto 的失败并不矛盾：那两者对齐/读出的是 caption 的 **CLIP 文本嵌入**（近义间仍共线），
而命名信息在 **caption 的词**里，被 CLIP 文本编码器抹平了。**caveat：caption 只在有标注的 train 场景存在，val 无 caption，
故此法不可直接部署在 val**——这是命名信号的**上限诊断**，不是可部署读出（可部署需 TASK C 把词级信号蒸馏进视觉特征）。

## 方法（annotation-free；GT 仅用于打分）
- anno_source = **gsam2 + seem**（与训练默认一致）。
- caption 文件按「每条 caption ↔ 一个点组」组织；把每个点组按多数投票归到其 GT 实例（`instance.npy`），
  实例真类由 `segment200.npy` 多数决（**仅打分**）。
- 命名：对簇内每个候选成员类名，统计该实例的 caption 中「包含该类名全部词」的条数，取最具体（词更多者优先）的命中，
  对该实例的所有 caption 取**众数**得预测类名。**不使用任何 GT/文本锚点/CLIP 相似度**。
- 打分只在 tight 近义簇（thr0.90/cap8，17 簇 44 类）内、限候选为簇成员。

## 自检
- baseline `pred_majority` sibling acc = **0.3822** ≈ TASK2 报告 **0.379**（见 attr_anchor.md 自检）。
- 命名率（caption 含任一候选类名词）= **0.9033**；未命名一律记为错（保守）。

## 结果（全 1201 caption 场景，4083 个 tight-cluster 实例）
- sibling 命名 acc（all，未命名=错）= **0.7311**
- sibling 命名 acc（among named，0.9033）= **0.8094**
- 对照：unsup_textname **0.377**、baseline pred_majority **0.379** → **caption 词命名 ≫ 文本锚点命名**。

### 逐簇（acc_all）
| 簇 | acc | named/total |
|---|---:|---|
| pillow \| cushion | 0.956 | 677/697 |
| curtain \| shower curtain \| shower curtain rod | 0.909 | 412/429 |
| clock \| alarm clock | 0.875 | 43/48 |
| towel \| blanket | 0.863 | 455/503 |
| bottle \| water bottle \| case of water bottles | 0.850 | 192/206 |
| plate \| cup \| tray \| bowl | 0.807 | 151/176 |
| plant \| potted plant | 0.799 | 273/293 |
| refrigerator \| mini fridge | 0.647 | 159/184 |
| shelf \| bookshelf | 0.599 | 737/775 |
| printer \| copier | 0.580 | 131/138 |
| stairs \| stair rail | 0.578 | 64/71 |
| board \| sign \| poster | 0.575 | 79/113 |
| coffee maker \| water pitcher \| coffee kettle | 0.551 | 70/89 |
| light switch \| power outlet \| power strip | 0.413 | 20/46 |
| bathroom stall \| bathroom stall door | 0.265 | 42/98 |
| column \| pillar | 0.206 | 17/34 |
| shower \| shower wall \| shower door \| shower floor \| shower head | 0.175 | 166/183 |

### 系统性失败模式（诚实记录）：VLM 用「上位词/常见词」
- copier 0.000（caption 说 "printer"）、mini fridge 0.000（说 "refrigerator/fridge"）、bookshelf 0.094（说 "shelf"）、
  stair rail 0.000（说 "stairs/railing"）、shower wall 0.086 / shower 部件（统称 "shower"）、column 0.050（与 pillar 混）。
- 即 caption 能解决**语义主类**的同辈混淆（pillow/cushion、towel/blanket、plate 家族、curtain 家族、bottle 家族），
  但对**同物细分/部件**（mini vs full、book- vs 普通 shelf、shower 部件、stall 门 vs 体）会退化为上位词。

## 预注册判据对号
- 判据（sibling 命名 acc > unsup_textname 0.377 且 > baseline 0.379，逐簇报）：**满足**（0.7311 ≫ 两者；逐簇见上）。
- 结论：**caption 词级信号是真实、可用的命名来源**，显著强于文本锚点。但 val 无 caption → 需 TASK C 把词级命名蒸馏进模型才能部署。
  关键设计含义：**TASK C 不应对齐 TASK A 的（共线）属性锚点或 caption 的 CLIP 嵌入**（=Method B 覆辙），
  应把「caption 类名众数词」当作**离散命名目标**，训一个轻量头去预测它（frozen backbone）。

## 复现命令
```bash
cd /root/Mosaic3D_work && export PROJECT_ROOT=/root/Mosaic3D_work
python scripts/task_b_caption_naming.py --sources gsam2,seem   # -> task2/caption_naming.json
```
