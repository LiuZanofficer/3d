# Impact Report (4090 vs A800, 1-epoch)

## High Impact (can change effect)
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

## Medium Impact (may change effect)
- `precision`

## Low Impact (mainly speed)
- `persistent_workers`
- `prefetch_factor`
- `checkpoint_policy`

## Assessment
- Final decision by rule: `NOT_PROVEN_NO_EFFECT`
- Reason: High-impact differences exist (data protocol, model backbone, BN eval strategy, global batch, transforms).
