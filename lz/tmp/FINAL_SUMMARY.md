# diag_rootcause_20260211_221332 FINAL SUMMARY

## 1) 运行命令清单（可复制）

```bash
# 连通验证（password auth + 8s timeout）
ssh -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=8 -p 49932 root@connect.nmb1.seetacloud.com "hostname && whoami && pwd"

# Step A 快照
cd /root/lz
OUT=/root/lz_outputs/diag_rootcause_20260211_221332
PY=/root/autodl-tmp/conda-envs/llz/bin/python

# Step B
$PY /root/lz/label_mapper_check.py

# Step C（0-C）
echo scene0000_00 > /root/lz/src/data/metadata/split_files/scannet_debug1.txt
$PY src/train.py \
  data=sc_smoke \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.root_dir=/root/lz \
  paths.output_dir=${OUT}/stepC_outputs \
  hydra.run.dir=${OUT}/hydra_stepC \
  hydra.sweep.dir=${OUT}/hydra_sweep_stepC \
  data.train_dataset.split=debug1 \
  data.val_datasets.0.split=debug1 \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  trainer.max_epochs=200 \
  +trainer.max_steps=2000 \
  > ${OUT}/overfit_0C.log 2>&1

# Step D（0-D）
$PY src/train.py \
  data=sc \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.root_dir=/root/lz \
  paths.output_dir=${OUT}/stepD_outputs \
  hydra.run.dir=${OUT}/hydra_stepD \
  hydra.sweep.dir=${OUT}/hydra_sweep_stepD \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  trainer.max_epochs=30 \
  +trainer.max_steps=0 \
  > ${OUT}/ce_only_sc.log 2>&1

# Step E（加 debug 日志并重跑 C）
$PY /root/lz/apply_debug_patch.py
$PY src/train.py \
  data=sc_smoke \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.root_dir=/root/lz \
  paths.output_dir=${OUT}/stepC_debug_outputs \
  hydra.run.dir=${OUT}/hydra_stepC_debug \
  hydra.sweep.dir=${OUT}/hydra_sweep_stepC_debug \
  data.train_dataset.split=debug1 \
  data.val_datasets.0.split=debug1 \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  trainer.max_epochs=200 \
  +trainer.max_steps=800 \
  > ${OUT}/overfit_0C_debug.log 2>&1
```

## 2) 关键指标表（C 与 D）

| Case | exitcode | best_val_miou | last_val_miou | first_loss_step | last_loss_step | 结论 |
|---|---:|---:|---:|---:|---:|---|
| C: overfit_0C | 137 | N/A | N/A | N/A | N/A | 进程被系统 kill，未进入有效训练迭代 |
| D: ce_only_sc | 137 | N/A | N/A | N/A | N/A | 进程被系统 kill，未进入有效训练迭代 |

附：Step E 重跑 C（debug）同样 exitcode=137，指标仍为 N/A。

## 3) 验收项 PASS/FAIL

1. A-环境快照完整：**PASS**
2. B-映射诊断完成且有明确结论：**PASS**
   - `valid_class_mapper[-1] == ignore_label == -100`，负标签映射逻辑正常
3. C-单场景过拟合（best_val_miou >= 0.80）：**FAIL**
   - 未产出可用 mIoU（进程被 kill）
4. D-纯 CE 基线（best_val_miou >= 0.02 或 > 0.01449）：**FAIL**
   - 未产出可用 mIoU（进程被 kill）

## 4) 根因 Top3（概率排序）

1. **无 GPU 运行导致资源链路异常（高概率）**
   - `env.txt` 显示 `nvidia-smi: No devices were found`
   - 日志显示 `GPU available: False, used: False`
2. **CLIP 初始化（HF 模型）在当前环境触发资源/网络问题（高概率）**
   - 日志出现 `Cannot assign requested address`（HF 请求）
   - 进入 CLIP instantiation 后被系统 kill（exit 137）
3. **早期 split 写入错误（已修复，非当前主因）**
   - 曾写成 `scene0000_00n`，导致 scene 不存在；已修正为 `scene0000_00`

## 5) 下一步最小可行修复计划（<=3条）

1. 先恢复可见 GPU（`nvidia-smi` 必须能看到卡）后再重跑 C/D，优先验证是否仍出现 exit 137。
2. 为排除 CLIP 干扰，先加 `model.text_encoder_cfg.use_clip=false` 跑通 0-C/0-D；确认 loss/mIoU 可正常变化后再恢复 CLIP。
3. 若必须保留 CLIP：先在远端预热/缓存 HF 权重并验证网络可达，再执行正式训练诊断。
