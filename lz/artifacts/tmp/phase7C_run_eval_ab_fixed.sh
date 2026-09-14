set -e
source /etc/network_turbo >/dev/null 2>&1 || true
PY=/root/autodl-tmp/conda-envs/llz/bin/python
cd /root/lz

BASE=/root/lz_outputs/phase7C_bn_fix_eval_only
OUT_A=$BASE/A_eval_bn_off
OUT_B=$BASE/B_eval_bn_on
mkdir -p "$OUT_A" "$OUT_B"

CKPT=/root/lz_outputs/phase7A_recheck_debug1_in6/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=0 "$PY" src/eval.py \
  model=spunet34c_lz \
  model.net.backbone.in_channels=6 \
  data=sc_smoke \
  ckpt_path="$CKPT" \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="$OUT_A" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  data.train_dataset.split=debug1 \
  data.val_datasets.0.split=debug1 \
  data.batch_size=1 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.seg_loss=1.0 \
  +model.eval_cfg.eval_bn_batch_stats=false \
  > "$OUT_A/eval.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PY" src/eval.py \
  model=spunet34c_lz \
  model.net.backbone.in_channels=6 \
  data=sc_smoke \
  ckpt_path="$CKPT" \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="$OUT_B" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  data.train_dataset.split=debug1 \
  data.val_datasets.0.split=debug1 \
  data.batch_size=1 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.seg_loss=1.0 \
  +model.eval_cfg.eval_bn_batch_stats=true \
  > "$OUT_B/eval.log" 2>&1
