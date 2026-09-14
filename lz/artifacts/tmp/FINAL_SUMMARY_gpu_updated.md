# diag_rootcause_gpu_20260211_225210 FINAL SUMMARY

## 运行命令清单（可复制）

### C: 单场景过拟合（0-C）
```bash
/root/autodl-tmp/conda-envs/llz/bin/python src/train.py \
  data=sc_smoke \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  data.train_dataset.split=debug1 \
  data.val_datasets.0.split=debug1 \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  trainer.max_epochs=200 \
  +trainer.max_steps=2000
```

### D: 纯 ScanNet CE（从中断点续跑到 30 epoch）
```bash
/root/autodl-tmp/conda-envs/llz/bin/python src/train.py \
  data=sc \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  trainer.max_epochs=30 \
  ckpt_path=/root/lz_outputs/diag_rootcause_gpu_20260211_225210/resume_tomorrow.ckpt
```

## 关键指标表

| Case | exitcode | best_val_miou | last_val_miou | first_loss_step | last_loss_step |
|---|---:|---:|---:|---:|---:|
| C: overfit 单场景 (`sc_smoke` + `debug1`) | 0 | 0.05830 | 0.05830 | 5.30 | 0.95 |
| D: CE-only 纯 ScanNet（resume 后完整到 30 epoch） | 0 | 0.00694 | 0.00587 | 2.82 | 2.54 |

## 验收 PASS/FAIL

1. A-环境快照完整：**PASS**
2. B-映射诊断完成且有明确结论：**PASS**
3. C-单场景过拟合（要求 >=0.80）：**FAIL**（0.05830）
4. D-纯 CE 基线（要求 >=0.02 或 >0.01449）：**FAIL**（0.00694）

## 根因 Top3（按概率）

1. **训练/评估口径与目标错位**：`val/miou` 在当前定义下对该设置过于苛刻，掩盖了可学习性提升。
2. **损失与监督信号有效密度不足**：仅 CE 路线下学习缓慢，mIoU 增长受限。
3. **数据子集与样本难度配置不匹配**：单场景 overfit 仍远低于应有上限，说明 pipeline 仍有未定位约束。

## 下一步最小可行修复计划（<=3条）

1. 在不改训练权重前提下新增并行指标 `miou_present`（仅对出现类平均），与 `val/miou` 同时记录。
2. 固定 `data=sc` 路线做更长 step 的 CE-only 小实验（例如 60 epoch）验证是否持续上升。
3. 对 0-C 增加 per-step 预测分布统计（top1_ratio / unique_count / ignore_ratio）锁定是否存在预测塌缩。

## 当前总体结论

- **30 epoch 已完整跑完**（日志含 `Trainer.fit stopped: max_epochs=30 reached`）。
- 现阶段核心问题仍在“训练信号与口径/设定”层，不是 SSH、GPU 或进程中断问题。
