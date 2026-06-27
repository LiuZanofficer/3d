# 错误分析假设日志（可证伪）

> 目的：把"看场景报现象"变成可证伪的科学流程。每条假设都要有**全验证集量化指标**，
> 不能只凭单场景印象下结论。流程见底部 Workflow。

## 评分口径（务必对齐）
- 预测：`pred_semantic`（只在前景类里取 argmax），不是裸 argmax。
- GT：背景类（`wall/floor/ceiling`）映射为 ignore；`ignore_label` 点丢弃。
- 指标：`IoU_c = tp/(tp+fp+fn)`（union=0 记 0），`fg-mIoU = mean(IoU over fg_class_idx)`。
- 单场景工具用 `--pred semantic`；analyzer 自检 fg-mIoU 必须≈本次 eval 报告值。

## 根因类型（填表用代号）
- **A 近义混淆**：FN 去向高度集中到 1-2 个语义近邻类（如 table↔kitchen cabinet）。
- **B 未学会 / dead**：该类 `pred_count≈0`，模型几乎从不预测它。
- **C 边界/上下文**：错误集中在物体边缘或与相邻大物体（wall/floor）交界。
- **D 小物体/召回低**：gt 点少、recall 低、FN 弥散，无单一去向。
- **E 标注/ignore**：错误其实落在 ignore 或标注噪声上，非模型问题。
- **F 上下文偏置**：同一类在不同场景表现差异大，依赖周围共现物体。

## 假设表

| # | 场景/范围 | 类 | 观察(现象) | 怀疑根因(A-F) | 量化指标(单场景) | 全val结果 | 结论(确认/推翻) | 下一步 |
|---|---|---|---|---|---|---|---|---|
| 1 | scene0011_00 | table | 电视下方整片 GT 漏检(绿) | A? C? | 该场景 IoU/FN去向top1 | analyzer FN去向top1占比 | 待定 | 多场景采样 |
| 2 |  |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |  |

### 示例（填写格式参考）
- **#1 table / scene0011_00**
  - 观察：电视柜上方那片 GT `table` 几乎全绿（FN），但旁边餐桌全黄（TP）。
  - 怀疑根因：A（近义混淆，疑似被判成 `kitchen cabinet`/`cabinet`）或 C（贴墙边界）。
  - 单场景量化：`python scripts/analyze_fn_predictions.py --pred-file <dump>/scene0011_00.npz --class-name table --pred semantic`
    → 记录 FN top-1 去向与占比、point-IoU。
  - 全val量化：`scripts/analyze_eval_dataset.py` 的 `fn_destinations.md` 里 `table` 的 FN top-1 去向是否同样集中。
  - 结论：____（若全val也集中到同一近义类→确认 A；若单场景集中但全val弥散→推翻，改判 D/F）。

## Workflow（闭环）
1. **排序选目标**：`analyze_eval_dataset.py` → `targets.md` 取 cost 最高的几类 + dead 类。
2. **多场景生成假设**：对每个目标类，挑 `targets.md` 给的"最差几个场景" + 随机 1-2 个，用
   `visualize_eval_query.py --pred semantic` 看，记录现象到上表（先别下结论）。
3. **全val量化确认**：回到 `fn_destinations.md` / `per_class_report.csv`，看该现象是否在全集成立
   （FN 去向是否集中、是否 dead、recall/precision 形态）。
4. **下结论**：确认或推翻，填"结论"列；确认的写成一句可检验命题（如"table 的 28% FN 被判为
   cabinet 类，属近义混淆"）。
5. **导出可改进项**：每个确认的根因对应一个可做的改法（如近义类文本提示工程 / 难负样本 /
   边界细化），进入实验队列。

## 备注
- 一切以全验证集数字为准；单场景只用来"生成假设"，不用来"证明假设"。
- 改了模型/推理后重跑 eval（带 `MOSAIC3D_DUMP_EVAL=1`）生成新 dump，再用同一套脚本对比。

---

# Phase 1 结果（全 val，dump=2026-06-26_02-22-42，预登记对号，2026-06-26）

> 纪律：以下每条的判据均在开跑前钉死，结果直接对号，未事后改阈值。

## V1a 文本名共线性（analyze_text_embeddings.py）
- 预登记：CONFIRM 需 `cos_abs ≥ p95_rand 且 ≥ μ_rand+0.10`；REFUTE 需 `≤ μ_rand+0.02`。
- 实测：吸点对 cos=0.9082；随机 μ=0.8135、p95=0.8652、σ=0.0321。0.9082≥p95 ✓ 但 <μ+0.10(0.9135) ✗。
- **结论：INCONCLUSIVE**（有迹象但差 0.0095 未达确认线，未挪线）。

## V4 合并同辈重算 mIoU（analyze_merge_superclass.py，不相交 top-1 配对 size≤2）
- 预登记：CONFIRM 需 `conf z>3 且 text z>3 且 Δ_conf≥+0.02`；PARTIAL 需 `conf z>3 但 text z<2`；REFUTE 需 `conf z<2`。
- 实测：baseline 0.1548 自检 OK。confusion 配对 mIoU 0.2456 vs 随机 0.1949±0.0064，Δ+0.0506，**z=7.9**；text 配对 0.2163 vs 0.2012±0.0070，Δ+0.0152，**z=2.2**。
- **结论：INCONCLUSIVE**（conf 极显著=点可恢复、FN高度集中在单一同辈、非纯特征问题；但 text z=2.2 落在[2,3]死区，文本名相似抓不住真实混淆）。
- 旧"阈值连通分量"版因链式巨组(101/197并一组)判为无效测量，弃用。

## V2 几何部分-整体（analyze_part_whole.py，5cm 表面）
- 预登记：CONFIRM 需 `frac_in≥0.70 且 比随机高≥0.30`；REFUTE 需 `frac_in≤0.30 或 diff≤0.10`。
- 实测：book→bookshelf frac_in=0.405(diff+0.405) → INCONCLUSIVE；pillow→bed 0.259 → REFUTE；doorframe→door 0.272 → REFUTE。
- **结论：包含机制弱**。误判点确实比随机家具更靠近真容器（diff大），但 5cm 内绝对包含仅 26–40%，未达"贴表面"。当 limitation。

## V1b winner-take-all 视觉方向（analyze_winner_take_all.py，Phase 2，重跑 2026-06-26_10-17-27）
- 预登记：CONFIRM 需 `r_winner≥0.70`；REFUTE 需 `r_winner≤0.50`；中间不确定。
- 实测：r_winner(top25)=**0.320**，平均 margin(sim_abs−sim_self)=−0.0027；随机类对照 r_winner_rand=0.011。
- **结论：REFUTE**。受害类的视觉均值在 68% 情况下仍最贴近自身文本（吸点文本平均并不更近）→ "整片点被文本决策系统性判给吸点"的 winner-take-all 故事不成立。
- 但吸点比随机类强约 29×（0.320 vs 0.011），说明混淆真实存在，只是发生在**逐点特征的方差/尾部**，不在类质心方向。

## V3 dead 类 caption 覆盖（analyze_caption_coverage.py，Phase 3，132万 caption）
- 预登记：CONFIRM 需 `ρ≥+0.40 且 median(freq[dead])≤0.25·median(freq[live])`；REFUTE 需 `ρ≤+0.10 或 dead≥live`。
- 实测：ρ(log freq, IoU)=**0.406**(p=3.4e-9)；median 频次 dead=560 vs live=1301(≈43%)；64 个 dead 类仅 5 个零覆盖。
- **结论：INCONCLUSIVE**。覆盖度与 IoU 正相关但非 dead 主因——92% 的 dead 类在 caption 里被提及数百次仍 IoU≈0。真零覆盖（监督缺口）只有 5 类：toilet seat cover dispenser / handicap bar / music stand / dumbbell / storage organizer。

# 最终综合裁决（Phase 1 + Phase 2 + Phase 3，2026-06-26）

> 原假设：*"开放词汇3D分割的主要失效 = 前景同辈/上位类的文本决策混淆（winner-take-all），而非特征能力不足"*。

| 验证 | 预登记结论 | 对原假设 |
|---|---|---|
| V4-conf z=7.9 | 点可恢复、FN 集中单一同辈 | 支持"可恢复"，但合并增益也可由特征重叠解释 |
| V1a 0.908 (差0.0095) | INCONCLUSIVE | 文本名共线**未坐实** |
| V4-text z=2.2 | 死区 | 文本名相似**抓不住**真实混淆 |
| V2 frac_in 0.26–0.41 | 弱/推翻 | 几何包含非主因 |
| V1b r_winner=0.32 | REFUTE | winner-take-all 文本决策方向**被推翻** |

- **判定：原"文本决策/winner-take-all"机制 NOT CONFIRMED（被 V1a/V4-text/V1b 联合证伪）。**
- **数据指向**：混淆真实且高度集中在 top-1 同辈，但发生在**逐点视觉特征的可分性**层面（受害类相当一部分点的特征落入吸点区，尽管类质心仍正确）——更像表示/边界/子群问题，不是全局文本锚点决策伪影。
- **纪律价值**：这套预登记判据**阻止了**围绕"文本决策/winner-take-all"写一个数据不支持的卖点（即"用数据编故事"的坑）。下一步若继续该方向，应转向"逐点特征判别性/难子群"切口，而非改文本锚点。
- **V3 补充**：caption 覆盖与 IoU 弱-中正相关(ρ=0.41)但非 dead 主因（dead 类多被提及数百次仍失败）→ 同样指向表示/可分性，而非监督覆盖缺口；仅 5 个真零覆盖类是干净的标注缺口。

---

# 新方向：受害类内子群刻画（2026-06-26）

## S1 边界子群（analyze_subpop_boundary.py）
- 预登记：CONFIRM 需 `diff≥+0.15 且 frac_pos≥0.70`；REFUTE 需 `diff≤+0.03`。
- 实测：23 对，平均 per-pair diff(FN→J − TP)=**+0.0066**，frac_pos=0.61 → **REFUTE（非边界效应）**。
- 关键观察：两组边界度都极小(~0.02)，误判点和正确点一样深在物体内部 → 不是边缘点；chair 67.6万点整片翻成 office chair，提示**instance 级整体翻转**，下一步 S3 验证。

## S3 instance 级整体翻转（analyze_subpop_instance.py）
- 预登记：CONFIRM 需 `mean leak_in_flipped≥0.60`；REFUTE 需 `≤0.30`。
- 实测：25 对，**mean leak_in_flipped=0.697** → **CONFIRM**。每实例 frac_J 双峰（近0/1占0.80，中间仅0.09）。
- 典型：door→window 0.93 / book→bookshelf 0.985 / pillow→bed 0.984 / chair→office chair 0.88；少数（反例 bed→mattress 0.00、desk→dresser 0.22）。
- 关键：`flipped_inst_frac` 低(chair 0.27)但 `leak_in_flipped` 高(0.88) → **少数大实例被整体误判，贡献绝大多数损失**。

## 新方向确诊（S1+S3）
- 失效机制 = **mask/instance 级的同辈外观混淆**：受害类的一部分实例被整体（且自信地）判给 top-1 同辈，非边界点、非散点、非文本决策、非长尾覆盖。
- 与 Mosaic3D(堆数据)/几何修漂移/长尾重加权/文本锚点**全部正交**。可做切口：mask 级特征聚合 + 同辈判别（hard-instance mining / 实例级置信校准 / mask→text 的同辈对比损失）。Mosaic3D 本就是 mask-based，落点自然。
- 待验证（下一步）：扩 dump 存"受害类逐实例视觉特征"，看翻转实例 vs 正确实例在特征空间是否可分（决定切口是"特征不可分"还是"决策阈值"）。

---

# Task 0/1/2：实例可分性与 caption 粒度（2026-06-27）

## Task 0 真 200 类标签空间 oracle 天花板（analyze_oracle_real_labelspace.py）
- 预登记：`oracle_fgmiou≤0.18` → 上限低；`≥0.21` → 有冲 20+ 物理空间；中间 → 可继续但降预期。
- 自检：baseline fg-mIoU=**0.1548**，OK。
- 实测：top25 oracle=**0.1906**(+0.0358) → MODEST；top50 oracle=**0.2118**(+0.0570) → ROOM_FOR_20_PLUS；all valid pairs oracle=**0.3257**(+0.1709)。
- 结论：只修 top25 空间不够大；扩到 top50 才刚越过 0.21。说明该问题**值得继续**，但真实方法不能按 all-oracle 或“暴涨到30”预期设计。

## Task 1 实例视觉特征可分性（language_module.py + analyze_instance_separability.py）
- 改动：eval dump 新增 `eval_instance_features/scannet200/<scene>.npz`，每个 GT 前景实例保存 `pooled_feat=L2norm(mean(L2norm(clip_feat)))`、真实类、预测直方图、多数预测等；不存逐点特征。
- 重跑：`logs/eval/runs/2026-06-27_06-58-17`，scannet200 mIoU=**0.154751956**，instance feature dumps=312。
- 预登记：AUC≥0.80 → 可分（决策/readout/聚合/校准问题）；AUC≤0.60 → 不可分（表示问题）；中间不确定。
- 实测：翻转 vs 正确实例 mean AUC=**0.984**，median=0.986；随机标签 AUC=0.528；真 C vs 真 J sibling AUC=0.943。
- 结论：**SEPARABLE**。翻转实例和正确实例在视觉特征空间高度可分，根因不再是“表示完全不可分”，更偏实例级 readout / 聚合 / 置信校准 / reranking 问题。

## Task 2 训练 caption 粒度（analyze_caption_granularity.py）
- 前置：当前 `data=sc+ar+sc++` ScanNet 配置未显式写 `anno_sources`，按 `dataset_base.py` 默认使用 `["gsam2", "seem"]`；本次统计这两个 source。
- 自检：baseline fg-mIoU=**0.1548**，OK；ScanNet train scenes=1201，caption regions=1,312,860；point_indices/segment 对齐 smoke check 通过。
- 预登记：CONFIRM 需 >50% top pairs 满足 `P(J|C)>0` 且 J 是非 C 第一名，并且 mean Spearman≥0.4；REFUTE 需 median `P(C|C)≥0.5` 且 mean Spearman<0.1；否则 INCONCLUSIVE。
- 实测：pair-confirm fraction=**0.240**(6/25)，mean Spearman=**0.270**，median `P(C|C)=0.238`。
- 结论：**INCONCLUSIVE**。训练 caption 有一定 sibling 偏细信号，但不足以解释多数 eval 翻转；也不能说 caption 完全正常。

## 2x2 裁决
- Task 1：特征可分（AUC 0.984）。
- Task 2：caption 偏细不确定（pair-confirm 0.24、rho 0.27）。
- 当前落点：**优先按“决策/readout/实例级校准”推进**，同时保留 caption 粒度作为次要混杂因素，而不是主因。
- 最便宜的下一步方法方向：在 annotation-free 设定内做 mask/instance 级 reranking 或 calibration，利用实例 pooled feature 与文本候选的 margin/entropy/同辈 hard negatives，不使用 ScanNet200 GT 训练。
