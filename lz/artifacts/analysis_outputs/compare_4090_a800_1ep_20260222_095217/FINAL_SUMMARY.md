# FINAL SUMMARY

- FINAL_DECISION: `NOT_PROVEN_NO_EFFECT`
- speed_compare(samples_per_sec): 4090=5.823411, A800=6.477423, delta=0.654012
- metric_compare(train/loss_epoch): 4090=2.130939, A800=2.905680, delta=0.774741
- metric_compare(val/miou): 4090=0.006650544703006744, A800=None
- metric_compare(val/miou_present_gt): 4090=0.00703761400654912, A800=None
- metric_compare(val/miou_fg_mosaic): 4090=0.0023671872913837433, A800=None

## High impact differences
- `batch_size`
- `global_batch`
- `data_name`
- `split.val`
- `sampler`
- `train_dataset_target`
- `train_transforms`
- `backbone_target`
- `voxel_grid`
- `eval_bn_batch_stats`

## Low impact differences
- `persistent_workers`
- `prefetch_factor`
- `checkpoint_policy`
