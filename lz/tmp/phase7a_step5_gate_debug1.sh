set -e
source /etc/network_turbo >/dev/null 2>&1 || true
PY=/root/autodl-tmp/conda-envs/llz/bin/python
REPO=/root/lz
OUT=/root/lz_outputs/phase7A_gate_debug1_60
mkdir -p "$OUT"
cd "$REPO"

CUDA_VISIBLE_DEVICES=0 "$PY" src/train.py \
  model=spunet34c_lz \
  data=sc \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="$OUT" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=60 \
  trainer.gradient_clip_val=1.0 \
  +trainer.precision=16-mixed \
  data.batch_size=1 \
  sampler.class_freq_path=/root/lz_outputs/class_freq_scannet200.json \
  optim.lr=0.001 \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.seg_loss.lovasz_weight=0.0 \
  data.coord_norm_enable=false \
  data.color_drop_prob=0.0 \
  data.train_dataset.split=debug1 \
  data.val_datasets.0.split=debug1 \
  > "$OUT/train.log" 2>&1


