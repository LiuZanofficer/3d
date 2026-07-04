# 部署差距拆解（D 线）：为什么 GT-instance 的可分性落不到部署 fg-mIoU

状态：预注册（数据未看，判据先定死）

## 0. 背景与动机
sibling-naming 整条线的 fg-mIoU 天花板已被 C1 钉死：即使 oracle 标签+oracle 门控+GT-instance 池化，
簇内改名 fg-mIoU 顶只有 +0.0158（0.2489→0.2647），可部署侵蚀后≈0。
但同一 GT-instance 池化协议下 sibNaming 已达 0.79，而部署（mask_text_vote）只有 fg-mIoU 0.1757、
TASK2 部署 sibNaming≈0.379。**主损失不在锚点/命名，而在"从 GT-instance 理想态跌到 mask 部署态"这一段。**
本线拆解这段损失的来源，为真正能抬 fg-mIoU 的方向定位。

## 1. 预注册假设（互斥归因）
部署差距主要来自以下之一/组合：
- **H1 mask 质量**：Segment3D 提案 mask ≠ GT 实例（过/欠分割、边界错），IoU/purity 低。
- **H2 池化漂移**：mask 池化的视觉特征相对 GT-instance 池化特征发生方向漂移，命名退化。
- **H3 点级指派/覆盖**：点被多 mask 覆盖/无 mask 覆盖 + MTV 投票冲突，逐点 fg-mIoU 掉。

## 2. 控制变量阶梯（每步只换一个因子；GT 仅用于打分）
- **L0** GT-instance 池化 + 文本锚点 + 簇内限制 sibNaming = 0.7916（已知，理想态）。
- **L0b** GT-instance 池化 + 文本 + 全 200-way（无簇限制），隔离"簇限制"贡献。
- **L1** Segment3D-mask 池化 + 文本 + 簇内限制 sibNaming（mask 按多数 GT 类打分）。L0→L1 = H2 池化漂移。
- **L2** mask 对 GT 实例的对齐质量：best-IoU 分布、purity 分布、GT 实例被 mask 召回率。量 H1。
- **L3** 点级：每点取其 mask 的 MTV 预测类，在"GT 类∈tight 簇"的点上算 sibNaming + fg-mIoU。L1→L3 = H3 指派/覆盖。

## 3. 预注册判据（先定死）
- 若 **L0→L1 掉幅 ≥ L1→L3 掉幅** → 主损失是 H1/H2（mask 池化质量）→ 下一步做 mask/提案质量或 mask 特征提纯。
- 若 **L1→L3 掉幅占主导** → 主损失是 H3（点级指派/覆盖）→ 下一步做逐点读出一致性/覆盖补齐。
- purity/IoU 中位数：若 mask best-IoU 中位数 < 0.5 或 purity 中位数 < 0.7 → H1 mask 质量是硬瓶颈。
- 所有结论用全 val 312 场景统计，不看单场景下结论。

## 4. 资产 / 复现（运行后回填）
- val mask 级 dump：scripts/dump_val_mask_feats.py（基线 ckpt + scannet200 masks_binary）
- 拆解：scripts/decompose_deployment_gap.py → 本报告表格

## 5. 结果（全 val 312 场景）
### 5.1 阶梯（L0 GT-instance vs L1 Segment3D-mask）
| 层级 | 池化单元 | 簇内 sibNaming | full-200 acc |
|---|---|---|---|
| L0 | GT 实例 | 0.7916 (pts 0.804) | 0.3794 |
| L1 | Segment3D mask | **0.8030** (pts 0.810) | **0.4575** |

**L0→L1 命名不降反升**（sibNaming +0.011，full-200 +0.078）。→ **H1(mask 质量)/H2(池化漂移)被否定**：
mask 池化特征至少和 GT-instance 一样好用于命名。

### 5.2 关键重解："0.379" 不是部署退化，而是 full-200
L0 的 full-200 acc = **0.3794** ≈ 历史"部署 sibNaming 0.379"。
所以"0.79 vs 0.379"= **簇内限制 vs 全 200-way 开放决策** 之差，**不是** mask/部署造成的。

### 5.3 tight 类 mask 的 full-200 错误归因（n=1777）
| 类型 | 未加权 | 点加权 |
|---|---|---|
| 正确 | 0.4575 | 0.6334 |
| 同辈(簇内)错误 | 0.1120 | 0.1423 |
| **簇外错误** | **0.4305** | **0.2243** |
预测停在正确簇内 = 0.5695。→ **簇外混淆是同辈混淆的 ~4×（点权 ~1.6×）**。

### 5.4 mask 对齐质量（L2）
ALL masks（n=13165）：best-IoU 中位 0.531、purity 中位 0.924、IoU>0.5 占 0.517。
tight 类 mask：best-IoU 中位 0.464、purity 中位 0.994。mask GT 有效 0.873、in-fg 0.756、in-tight 0.135。
→ mask 较纯（purity 高），但 best-IoU 中位仅 ~0.5（存在过/欠分割），影响 IoU 而非命名。

### 5.5 fg-mIoU 的真实构成（来自 baseline+MTV eval）
miou_head=0.368、miou_common=0.120、**miou_tail=0.033**；fg-mIoU=0.1757。
→ fg-mIoU 被**长尾类近零 IoU + 逐类平均**主导。

## 6. 结论与下一方向
**否定**：部署差距不来自 mask 质量/池化漂移（L0→L1 命名反升）。"0.79→0.379"是簇内限制 vs 全 200-way，非部署。
**定位**：对歧义类，**簇外(全 200-way)混淆 >> 同辈混淆**（未加权 4×、点权 1.6×）；
sibling 整条线（TASK A/B/C）只覆盖同辈小头，fg-mIoU 天花板 +0.0158（已钉死）。
fg-mIoU 的主拖累是**长尾类近零 IoU**（tail 0.033 vs head 0.368）。

真正能抬 fg-mIoU 的 annotation-free 方向（按杠杆排序）：
1. **全 200-way 判别 / 簇外混淆**：文本提示工程/多模板集成、per-class 分数校准、开放词表 logit 温度/先验校正（尤其压制高频类对尾类的吞并）。
2. **长尾类 IoU**：尾类召回（tail-aware 校准、mask 级尾类先验），因逐类平均下尾类权重最大。
3. mask 提案 IoU（中位 ~0.5）可作次级项，但命名不受其害，优先级低于 1/2。

注：以上均 annotation-free（GT 仅打分）。sibling 命名线到此确定为**低杠杆**，不再投入。
