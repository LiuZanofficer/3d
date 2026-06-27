# Task 1B instance visual feature separability

## Self-check
- baseline fg-mIoU = **0.1548** (ref 0.1548; OK)
- top-k cost pairs = 25; flip_thr=0.5; correct_thr=0.5
- instances loaded = 7769 from `logs/eval/runs/2026-06-27_06-58-17/eval_instance_features/scannet200`

## PRE-REGISTERED criteria
- AUC>=0.80: separable (decision/readout/aggregation/calibration)
- AUC<=0.60: not separable (representation/feature issue)
- otherwise: inconclusive

## Summary
- mean AUC flipped-vs-correct = **0.984**; median=0.986
- random-label control mean AUC = **0.528**
- mean true-C-vs-true-J sibling AUC = **0.943**
- **Verdict: SEPARABLE (decision/readout/aggregation/calibration issue)**

## Per-pair probes

| C | absorber J | cost | n_correct | n_flipped | AUC flip-vs-correct | random AUC | AUC true C-vs-J | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| chair | office chair | 1237268 | 649 | 311 | 0.998 | 0.463 | 0.869 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| shelf | bookshelf | 810674 | 47 | 41 | 0.958 | 0.455 | 0.939 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| cabinet | kitchen cabinet | 624692 | 20 | 22 | 0.984 | 0.548 | 0.958 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| window | blinds | 561323 | 187 | 18 | 0.991 | 0.529 | 0.977 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| door | window | 451372 | 204 | 7 | 1.000 | 0.455 | 0.984 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| table | desk | 373151 | 172 | 25 | 0.994 | 0.475 | 0.953 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| book | bookshelf | 352139 | 0 | 364 | nan | nan | 0.994 | LOW_SAMPLE |
| desk | dresser | 266085 | 95 | 2 | nan | nan | 0.997 | LOW_SAMPLE |
| trash can | trash bin | 254491 | 34 | 83 | 0.987 | 0.469 | 0.916 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| mailbox | bookshelf | 217967 | 0 | 2 | nan | nan | 0.989 | LOW_SAMPLE |
| curtain | window | 205013 | 43 | 7 | 0.970 | 0.724 | 0.964 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| bookshelf | shelf | 204247 | 63 | 10 | 0.994 | 0.695 | 0.926 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| clothes dryer | oven | 190001 | 1 | 3 | nan | nan | 1.000 | LOW_SAMPLE |
| whiteboard | board | 183777 | 32 | 27 | 0.968 | 0.449 | 0.857 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| doorframe | door | 175002 | 0 | 73 | nan | nan | 0.917 | LOW_SAMPLE |
| armchair | sofa chair | 151202 | 37 | 13 | 0.994 | 0.538 | 0.916 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| pillow | bed | 137850 | 18 | 123 | 0.982 | 0.476 | 0.992 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| box | crate | 133254 | 53 | 2 | nan | nan | 0.799 | LOW_SAMPLE |
| bed | mattress | 127571 | 63 | 0 | nan | nan | 0.991 | LOW_SAMPLE |
| object | bicycle | 125030 | 0 | 3 | nan | nan | nan | LOW_SAMPLE |
| picture | poster | 122695 | 71 | 69 | 0.975 | 0.451 | 0.716 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| couch | bench | 113702 | 83 | 7 | 0.985 | 0.664 | 0.999 | SEPARABLE (decision/readout/aggregation/calibration issue) |
| blinds | closet wall | 107859 | 2 | 4 | nan | nan | 0.987 | LOW_SAMPLE |
| copier | printer | 95293 | 0 | 5 | nan | nan | 0.994 | LOW_SAMPLE |
| radiator | vent | 92965 | 0 | 24 | nan | nan | 1.000 | LOW_SAMPLE |
