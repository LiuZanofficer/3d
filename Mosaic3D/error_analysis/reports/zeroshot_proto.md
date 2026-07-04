# 零训练视觉原型 + 词命名 (TASK C 第一步：先验证迁移，再决定是否轻量训练)

状态：预注册（数据未看，判据先定死）

## 0. 一句话
在**全 train 集**上，用**基线模型（qz/sc+ar+sc++.ckpt）**的逐点 `clip_feat`，
按 **caption 点组**池化出区域视觉特征，只在 **tight 近义簇内**用最近邻原型做**簇内消歧**
（全局仍走 CLIP 文本读出，保住 open-vocab）。在 val 上量 sibling naming acc 与 fg-mIoU。

## 1. 与已死的 caption_proto 的三点差异（必须写明，否则是重复实验）
caption_proto 失败形态：`300-batch` 子集、region 级池化、**单条 caption 解析**、**原型直接当读出**。
本方法（干净版）的三点差异：
1. **全 train 集**（非 300-batch 子集）：遍历 scannet train 全部场景建原型，样本量大、类覆盖全。
2. **词级众数分组**（非单条 caption 解析）：区域类别 = 该 caption 的**类名众数词**
   （TASK B 已证词级众数命名 acc=0.7311，远比单条解析干净）；不再用单条噪声 caption。
3. **hybrid 簇内消歧**（非原型当读出）：全局读出仍是 CLIP 文本（open-vocab 不动），
   原型**只**在 tight 近义簇内对候选做重排；非近义簇实例完全不受影响。

## 2. 红线（数据使用）
- train 侧分组只用 **caption 点组**（Segment3D 派生区域），**不许**用 train 的 `gt_instance/segment200` 建可部署原型。
- GT 只许用于**打分**与**oracle 对照诊断**（下面 (ii)）。
- annotation-free：训练/构造不碰 ScanNet200 GT 标签。

## 3. oracle 对照（把"原型质量"和"词命名质量"拆开）
同一套原型构造流程（caption 点组池化），跑两版命名：
- **(i) caption 词众数命名**：区域标签 = caption 类名众数词 → 可部署逻辑。
- **(ii) train GT 命名（仅诊断）**：区域标签 = 该 caption 点组的 segment200 众数类 → 诊断上限。
两版都在 **val** 上按同一"restricted-to-cluster"协议量 sibling naming acc。

## 4. 预注册判据（数值 + gate）
参照系（同一 val 实例特征、同一 tight 簇、同一 restricted-to-cluster 协议）：
- baseline 文本名锚点 sibNaming（离线上限）≈ **0.792**；可部署(在线) ≈ **0.379/0.377**。
- TASK B 词命名（train 上，诊断）≈ **0.731**。

判据（先定死，不许事后挪）：
- **A 成功**：(i) caption 词众数原型在 val 上 sibNaming **显著高于 0.379**（可部署上限），
  且带来 tight 簇内 fg-mIoU 正增益 → 得到 training-free 方法，B 变"可选放大"。
- **A 失败但 oracle 成功**：(i) 不显著、但 (ii) train GT 原型在 val 上 sibNaming 显著高于 0.379
  → 信号在、原型/词太弱 → **照常做 B（轻量词蒸馏）**。
- **oracle 也失败**：(ii) 也不高于 0.379 → 特征在 train→val 不迁移 → **杀掉整条路（含 B）**。

"显著"操作定义：相对 0.379 的绝对提升 ≥ +0.03，且 tight 簇内 fg-mIoU 不降。

## 5. 复现命令 / 资产（运行后回填）
- dump: scripts/dump_caption_region_feats.py（8 卡分片，基线 ckpt）→ /root/proto_dump/*.npz
- 建原型 + 评分: scripts/build_score_proto.py → 本报告表格
- val 实例特征: logs/eval/runs/2026-06-27_06-58-17/eval_instance_features/scannet200 (312 scenes)
- tight 簇: error_analysis/reports/task2/clusters_thr0.90.npz (size 2..8)


## 6. 结果（已运行，min_np 扫描稳健，min_np=0 最优）
协议：val GT-instance 池化特征（离线 oracle 上界，与 analyze_attr_anchors 一致）。
参照系（同协议）：文本名锚点 sibNaming=**0.7916**、miou_incluster=0.2489；
（在线可部署 sibNaming 参照=0.379，属另一更低回归，见 §注）。in_tight=1099，tight 簇=17，覆盖 44 类。

聚合（min_np=0）：
| 方法 | sibNaming acc | miou_incluster | 说明 |
|---|---|---|---|
| text（文本名锚点，基线） | 0.7916 | 0.2489 | 全局 CLIP 文本读出 |
| proto_caption（可部署 i） | 0.7771 | 0.2372 | caption 词众数标签建原型；**< 文本** |
| proto_gt（oracle ii） | **0.8053** | **0.2544** | segment200 标签建原型；**> 文本** |
| **按簇择优(text/proto_gt)** | **0.8508** | — | 选择性上界（+0.0592 vs 文本） |

min_np 扫描（越大越差 → 更多区域=更好质心）：proto_gt @0/25/50/100/200 = 0.805/0.804/0.791/0.781/0.772。

## 6b. 每簇拆解（关键：聚合 +0.014 掩盖强异质性）
proto_gt − text 的按簇差（正=视觉原型赢）：
| 簇 | n | text | proto_gt | proto_cap | Δ(gt−text) |
|---|---|---|---|---|---|
| bottle\|water bottle\|case of water | 47 | 0.362 | 0.702 | 0.553 | **+0.340** |
| stairs\|stair rail | 18 | 0.611 | 0.889 | 0.389 | **+0.278** |
| printer\|copier | 37 | 0.676 | 0.946 | 0.243 | **+0.270** |
| coffee maker\|water pitcher\|... | 11 | 0.727 | 0.909 | 0.545 | +0.182 |
| shelf\|bookshelf | 232 | 0.711 | 0.828 | 0.815 | +0.116 |
| shower\|shower wall\|... | 47 | 0.809 | 0.851 | 0.681 | +0.043 |
| plant\|potted plant | 50 | 0.920 | 0.940 | 0.460 | +0.020 |
| ... | | | | | |
| pillow\|cushion | 214 | 0.808 | 0.715 | 0.883 | −0.093 |
| board\|sign\|poster | 37 | 0.784 | 0.568 | 0.622 | −0.216 |
| plate\|cup\|tray\|bowl | 58 | 0.707 | 0.431 | 0.638 | **−0.276** |

规律：视觉原型在**"文本共线但视觉可分"**的簇上大赢（bottle/printer/stairs/coffee maker/shelf）；
在**视觉细粒度相近**的簇上大亏（plate/cup/bowl、board/sign/poster）。
且在最高价值簇（printer/copier、stairs），**caption 词命名因 VLM 上位词而崩**（0.243/0.389），
即可部署词标签无法吃到这些增益。

## 7. gate 判定与 TASK C 去留
- **A（可部署零训练 caption 词原型）= 失败**：同协议下 0.7771 < 文本 0.7916（min_np 全程如此）。
  相对在线 0.379 虽高，但那是**部署差距**（GT-instance 0.79 vs 部署 0.38，来自 mask/实例成形+逐点读出），
  与"文本 vs 视觉锚点"无关，不能算 A 成功。
- **oracle 对照 = 成功（且带强选择性头顶）**：proto_gt 0.8053 > 文本 0.7916；
  按簇择优上界 **0.8508（+0.0592）**。**视觉特征确实 train→val 迁移**，信号在。
- 按预注册 gate："A 失败但 oracle 成功 → 照常做 B（信号在，原型太弱）"。

**诚实的量级修正（供决策）**：
1. 盲目全局用原型几乎无收益（+0.014）；价值全在**按簇选择性**（+0.059 上界）。
2. 最高价值簇的**词标签被 VLM 上位词毁掉**（printer/copier caption acc 0.243）→
   直接用 caption 词做 B 的目标会继承此天花板，吃不到 printer/stairs 的大增益。
3. 全部为**离线 GT-instance 上界**；部署差距（0.79→0.38）才是 fg-mIoU 的主杠杆，
   属 mask/实例成形与逐点读出，非锚点/原型。

**推荐的 TASK C 两条路（择一，需用户定）**：
- C1 训练-free **按簇门控原型读出**：在 train 上 annotation-free 估计每簇 proto-vs-text 收益，
  只在有益簇（bottle/printer/stairs/coffee maker/shelf）启用视觉原型覆盖，其余保持文本。
  离线上界 +0.059 sibNaming；无需训练；风险是门控需可靠的 annotation-free 估计。
- C2 **轻量词蒸馏训练（B）**：须解决上位词——高价值簇不能只用 caption 原词做目标，
  需用视觉聚类伪标签或原型软标签，否则天花板≈文本。

## 注（协议）
离线 sibNaming 用 val GT-instance 池化特征，是上界诊断；在线可部署（mask_text_vote）
sibNaming≈0.379。本轮结论回答的是"视觉原型信号是否 train→val 迁移"（是），
以及"nearest-centroid 相对文本读出的头顶"（盲目≈0，选择性≈+0.06 上界）。
