set -e
source /etc/network_turbo >/dev/null 2>&1 || true
PY=/root/autodl-tmp/conda-envs/llz/bin/python
cd /root/lz
OUT=/root/lz_outputs/phase7A_recheck_debug1_aligned
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 "$PY" src/train.py \
  model=spunet34c_lz \
  data=sc_smoke \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="$OUT" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=200 \
  +trainer.max_steps=2000 \
  +trainer.precision=16-mixed \
  data.train_dataset.split=debug1 \
  data.val_datasets.0.split=debug1 \
  data.batch_size=1 \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.seg_loss.lovasz_weight=0.0 \
  data.coord_norm_enable=false \
  data.color_drop_prob=0.0 \
  callbacks.model_checkpoint.save_top_k=0 \
  callbacks.model_checkpoint.save_last=true \
  > "$OUT/train_regen.log" 2>&1
