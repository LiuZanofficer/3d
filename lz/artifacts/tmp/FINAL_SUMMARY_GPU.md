# diag_rootcause_gpu_20260211_225210 FINAL SUMMARY

## 状态
- 本次在 **有卡模式（RTX 4090）** 下完成了 0-C 与 0-D 复测，均成功结束（exitcode=0）。
- 关键输出目录：`/root/lz_outputs/diag_rootcause_gpu_20260211_225210`

## 关键指标

| Case | exitcode | best_val_miou | last_val_miou | first_loss_step | last_loss_step |
|---|---:|---:|---:|---:|---:|
| 0-C 单场景 overfit (`data=sc_smoke`, `debug1`) | 0 | 0.0583 | 0.0583 | 5.30 | 0.95 |
| 0-D 纯 ScanNet CE (`data=sc`, 本次 capped 到 `max_steps=3000`) | 0 | 0.00498 | 0.00498 | 5.31 | 3.29 |

## 验收判定（按你之前约定）
1. C pass 条件：`best_val_miou >= 0.80` → **FAIL**（0.0583）
2. D pass 条件：`best_val_miou >= 0.02` 或 `> 0.01449` → **FAIL**（0.00498）

## 结论
- 现在已经排除“无卡导致直接失败”的问题；训练流程在有卡下可稳定执行。
- 但指标仍低，问题核心转为“模型/损失/指标口径与训练设定”层面，不是 SSH 或 GPU 可用性问题。

## 下一步最小动作（建议按顺序）
1. 跑一次 **0-C 延长步数**（例如 `trainer.max_epochs=2000 +trainer.max_steps=2000`）验证是否仅是当前 overfit 步数不足。
2. 跑一次 **0-D 全量 30 epoch（不设 max_steps）** 获取完整基线，而不是 3000 steps 截断基线。
3. 新增并同时记录 `miou_present`（仅对 val 中出现类求平均）用于与当前 `val/miou` 并排解释，排除“全 200 类平均导致单场景上限偏低”的误判。
