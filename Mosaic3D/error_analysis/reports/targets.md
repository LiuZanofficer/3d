# Error-analysis targets

- scenes analyzed: **312**
- foreground classes: **197**
- **fg-mIoU self-check: 0.1548** (should match the eval-reported val miou; if not, the scoring convention is off)

## Most worth looking at (fg classes by lost GT points = cost)

| rank | class | gt | pred | IoU | recall | prec | cost |
|---|---|---|---|---|---|---|---|
| 1 | chair | 2491617 | 1314846 | 0.4915 | 0.5034 | 0.9540 | 1237265 |
| 2 | shelf | 1169879 | 626609 | 0.2499 | 0.3070 | 0.5733 | 810674 |
| 3 | cabinet | 758107 | 207686 | 0.1603 | 0.1760 | 0.6424 | 624692 |
| 4 | window | 1385131 | 1096729 | 0.4969 | 0.5948 | 0.7512 | 561321 |
| 5 | door | 1291634 | 1139657 | 0.5281 | 0.6505 | 0.7373 | 451372 |
| 6 | table | 1224698 | 1070620 | 0.5898 | 0.6953 | 0.7954 | 373151 |
| 7 | book | 352186 | 77 | 0.0001 | 0.0001 | 0.6104 | 352139 |
| 8 | desk | 655163 | 637674 | 0.4305 | 0.5939 | 0.6101 | 266087 |
| 9 | trash can | 298994 | 56975 | 0.1429 | 0.1488 | 0.7811 | 254491 |
| 10 | mailbox | 217967 | 0 | 0.0000 | 0.0000 | 0.0000 | 217967 |
| 11 | curtain | 657363 | 523346 | 0.6211 | 0.6881 | 0.8643 | 205013 |
| 12 | bookshelf | 1125319 | 1808296 | 0.4577 | 0.8185 | 0.5094 | 204248 |
| 13 | clothes dryer | 206786 | 92592 | 0.0594 | 0.0812 | 0.1813 | 190001 |
| 14 | whiteboard | 323582 | 219672 | 0.3465 | 0.4321 | 0.6364 | 183777 |
| 15 | doorframe | 185363 | 48464 | 0.0464 | 0.0559 | 0.2137 | 175005 |
| 16 | armchair | 320557 | 281916 | 0.3910 | 0.5283 | 0.6007 | 151202 |
| 17 | pillow | 153508 | 25721 | 0.0957 | 0.1020 | 0.6088 | 137850 |
| 18 | box | 178867 | 82276 | 0.2116 | 0.2550 | 0.5544 | 133254 |
| 19 | bed | 841283 | 854700 | 0.7266 | 0.8484 | 0.8350 | 127570 |
| 20 | object | 125030 | 1 | 0.0000 | 0.0000 | 0.0000 | 125030 |
| 21 | picture | 184537 | 82559 | 0.3013 | 0.3351 | 0.7491 | 122695 |
| 22 | couch | 789769 | 774737 | 0.7610 | 0.8560 | 0.8726 | 113702 |
| 23 | blinds | 150500 | 308447 | 0.1024 | 0.2833 | 0.1382 | 107859 |
| 24 | copier | 99703 | 7670 | 0.0428 | 0.0442 | 0.5750 | 95293 |
| 25 | radiator | 96438 | 4962 | 0.0355 | 0.0360 | 0.6999 | 92965 |

## Dead classes (have GT but ~never predicted)

| class | gt | pred | IoU |
|---|---|---|---|
| book | 352186 | 77 | 0.0001 |
| mailbox | 217967 | 0 | 0.0000 |
| object | 125030 | 1 | 0.0000 |
| headphones | 1002 | 0 | 0.0000 |
| dustpan | 762 | 0 | 0.0000 |
| power outlet | 711 | 0 | 0.0000 |
| plunger | 613 | 0 | 0.0000 |
| dumbbell | 507 | 0 | 0.0000 |
| light switch | 432 | 0 | 0.0000 |
| mouse | 350 | 0 | 0.0000 |

## Worst scenes per target (for visualization)

### chair (idx=1)
- `scene0474_02` IoU=0.000 gt=5996 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0474_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0474_02.npz --default-class "chair"`
- `scene0203_01` IoU=0.000 gt=3450 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0203_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0203_01.npz --default-class "chair"`
- `scene0207_00` IoU=0.000 gt=4505 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0207_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0207_00.npz --default-class "chair"`
- `scene0207_01` IoU=0.000 gt=2391 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0207_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0207_01.npz --default-class "chair"`
- `scene0207_02` IoU=0.000 gt=2687 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0207_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0207_02.npz --default-class "chair"`

### shelf (idx=7)
- `scene0203_01` IoU=0.000 gt=8901 pred=7760
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0203_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0203_01.npz --default-class "shelf"`
- `scene0664_02` IoU=0.000 gt=1005 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0664_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0664_02.npz --default-class "shelf"`
- `scene0664_01` IoU=0.000 gt=1017 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0664_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0664_01.npz --default-class "shelf"`
- `scene0207_00` IoU=0.000 gt=5489 pred=9158
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0207_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0207_00.npz --default-class "shelf"`
- `scene0329_01` IoU=0.000 gt=9130 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0329_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0329_01.npz --default-class "shelf"`

### cabinet (idx=6)
- `scene0474_03` IoU=0.000 gt=1246 pred=620
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0474_03 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0474_03.npz --default-class "cabinet"`
- `scene0608_02` IoU=0.000 gt=766 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0608_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0608_02.npz --default-class "cabinet"`
- `scene0307_01` IoU=0.000 gt=2341 pred=9
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0307_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0307_01.npz --default-class "cabinet"`
- `scene0307_02` IoU=0.000 gt=9782 pred=505
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0307_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0307_02.npz --default-class "cabinet"`
- `scene0351_00` IoU=0.000 gt=3746 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0351_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0351_00.npz --default-class "cabinet"`

### window (idx=14)
- `scene0494_00` IoU=0.000 gt=926 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0494_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0494_00.npz --default-class "window"`
- `scene0643_00` IoU=0.000 gt=1118 pred=10
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0643_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0643_00.npz --default-class "window"`
- `scene0693_00` IoU=0.000 gt=509 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0693_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0693_00.npz --default-class "window"`
- `scene0169_00` IoU=0.000 gt=8523 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0169_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0169_00.npz --default-class "window"`
- `scene0426_03` IoU=0.000 gt=3227 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0426_03 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0426_03.npz --default-class "window"`

### door (idx=4)
- `scene0702_02` IoU=0.000 gt=1757 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0702_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0702_02.npz --default-class "door"`
- `scene0702_01` IoU=0.000 gt=1720 pred=4
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0702_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0702_01.npz --default-class "door"`
- `scene0458_00` IoU=0.000 gt=178 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0458_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0458_00.npz --default-class "door"`
- `scene0653_00` IoU=0.000 gt=1094 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0653_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0653_00.npz --default-class "door"`
- `scene0100_01` IoU=0.000 gt=1774 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0100_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0100_01.npz --default-class "door"`

### table (idx=3)
- `scene0077_01` IoU=0.000 gt=1593 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0077_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0077_01.npz --default-class "table"`
- `scene0701_00` IoU=0.000 gt=4068 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0701_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0701_00.npz --default-class "table"`
- `scene0203_01` IoU=0.000 gt=1328 pred=140
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0203_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0203_01.npz --default-class "table"`
- `scene0580_01` IoU=0.000 gt=1088 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0580_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0580_01.npz --default-class "table"`
- `scene0580_00` IoU=0.000 gt=975 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0580_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0580_00.npz --default-class "table"`

### book (idx=19)
- `scene0025_00` IoU=0.000 gt=547 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0025_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0025_00.npz --default-class "book"`
- `scene0025_01` IoU=0.000 gt=1725 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0025_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0025_01.npz --default-class "book"`
- `scene0025_02` IoU=0.000 gt=929 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0025_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0025_02.npz --default-class "book"`
- `scene0030_00` IoU=0.000 gt=25051 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0030_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0030_00.npz --default-class "book"`
- `scene0030_01` IoU=0.000 gt=23723 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0030_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0030_01.npz --default-class "book"`

### desk (idx=8)
- `scene0629_02` IoU=0.000 gt=831 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0629_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0629_02.npz --default-class "desk"`
- `scene0629_01` IoU=0.000 gt=775 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0629_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0629_01.npz --default-class "desk"`
- `scene0645_00` IoU=0.001 gt=5376 pred=4
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0645_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0645_00.npz --default-class "desk"`
- `scene0665_01` IoU=0.005 gt=12878 pred=219
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0665_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0665_01.npz --default-class "desk"`
- `scene0665_00` IoU=0.016 gt=22236 pred=359
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0665_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0665_00.npz --default-class "desk"`

### trash can (idx=48)
- `scene0695_02` IoU=0.000 gt=946 pred=325
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0695_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0695_02.npz --default-class "trash can"`
- `scene0146_01` IoU=0.000 gt=4050 pred=19
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0146_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0146_01.npz --default-class "trash can"`
- `scene0146_00` IoU=0.000 gt=4061 pred=8
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0146_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0146_00.npz --default-class "trash can"`
- `scene0583_02` IoU=0.000 gt=106 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0583_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0583_02.npz --default-class "trash can"`
- `scene0583_00` IoU=0.000 gt=520 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0583_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0583_00.npz --default-class "trash can"`

### mailbox (idx=163)
- `scene0304_00` IoU=0.000 gt=40596 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0304_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0304_00.npz --default-class "mailbox"`
- `scene0338_00` IoU=0.000 gt=14979 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0338_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0338_00.npz --default-class "mailbox"`
- `scene0338_01` IoU=0.000 gt=12295 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0338_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0338_01.npz --default-class "mailbox"`
- `scene0338_02` IoU=0.000 gt=13924 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0338_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0338_02.npz --default-class "mailbox"`
- `scene0462_00` IoU=0.000 gt=33733 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0462_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0462_00.npz --default-class "mailbox"`

### curtain (idx=18)
- `scene0193_01` IoU=0.000 gt=3171 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0193_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0193_01.npz --default-class "curtain"`
- `scene0382_01` IoU=0.000 gt=4501 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0382_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0382_01.npz --default-class "curtain"`
- `scene0702_02` IoU=0.000 gt=2193 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0702_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0702_02.npz --default-class "curtain"`
- `scene0696_00` IoU=0.000 gt=5544 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0696_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0696_00.npz --default-class "curtain"`
- `scene0357_00` IoU=0.006 gt=3693 pred=810
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0357_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0357_00.npz --default-class "curtain"`

### bookshelf (idx=16)
- `scene0695_02` IoU=0.001 gt=4490 pred=4
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0695_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0695_02.npz --default-class "bookshelf"`
- `scene0648_00` IoU=0.002 gt=6596 pred=6025
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0648_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0648_00.npz --default-class "bookshelf"`
- `scene0695_00` IoU=0.006 gt=4843 pred=27
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0695_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0695_00.npz --default-class "bookshelf"`
- `scene0695_03` IoU=0.007 gt=4688 pred=34
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0695_03 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0695_03.npz --default-class "bookshelf"`
- `scene0700_01` IoU=0.025 gt=28279 pred=702
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0700_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0700_01.npz --default-class "bookshelf"`

### clothes dryer (idx=92)
- `scene0139_00` IoU=0.000 gt=7183 pred=1049
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0139_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0139_00.npz --default-class "clothes dryer"`
- `scene0595_00` IoU=0.014 gt=15272 pred=2581
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0595_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0595_00.npz --default-class "clothes dryer"`
- `scene0678_02` IoU=0.043 gt=54653 pred=16480
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0678_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0678_02.npz --default-class "clothes dryer"`
- `scene0678_01` IoU=0.054 gt=68176 pred=28239
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0678_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0678_01.npz --default-class "clothes dryer"`
- `scene0678_00` IoU=0.108 gt=61502 pred=27694
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0678_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0678_00.npz --default-class "clothes dryer"`

### whiteboard (idx=45)
- `scene0353_02` IoU=0.000 gt=885 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0353_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0353_02.npz --default-class "whiteboard"`
- `scene0353_00` IoU=0.000 gt=881 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0353_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0353_00.npz --default-class "whiteboard"`
- `scene0663_02` IoU=0.000 gt=325 pred=2
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0663_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0663_02.npz --default-class "whiteboard"`
- `scene0500_01` IoU=0.000 gt=2828 pred=8
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0500_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0500_01.npz --default-class "whiteboard"`
- `scene0414_00` IoU=0.001 gt=8310 pred=1063
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0414_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0414_00.npz --default-class "whiteboard"`

### doorframe (idx=118)
- `scene0702_01` IoU=0.000 gt=1586 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0702_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0702_01.npz --default-class "doorframe"`
- `scene0702_00` IoU=0.000 gt=1069 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0702_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0702_00.npz --default-class "doorframe"`
- `scene0050_02` IoU=0.000 gt=2633 pred=77
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0050_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0050_02.npz --default-class "doorframe"`
- `scene0077_01` IoU=0.000 gt=929 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0077_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0077_01.npz --default-class "doorframe"`
- `scene0084_00` IoU=0.000 gt=696 pred=204
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0084_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0084_00.npz --default-class "doorframe"`

### armchair (idx=20)
- `scene0011_00` IoU=0.000 gt=3243 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0011_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0011_00.npz --default-class "armchair"`
- `scene0019_01` IoU=0.000 gt=17214 pred=4
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0019_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0019_01.npz --default-class "armchair"`
- `scene0203_00` IoU=0.000 gt=3834 pred=17
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0203_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0203_00.npz --default-class "armchair"`
- `scene0423_00` IoU=0.000 gt=20720 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0423_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0423_00.npz --default-class "armchair"`
- `scene0423_01` IoU=0.000 gt=15415 pred=1
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0423_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0423_01.npz --default-class "armchair"`

### pillow (idx=11)
- `scene0019_01` IoU=0.000 gt=2078 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0019_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0019_01.npz --default-class "pillow"`
- `scene0699_00` IoU=0.000 gt=1452 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0699_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0699_00.npz --default-class "pillow"`
- `scene0697_03` IoU=0.000 gt=2595 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0697_03 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0697_03.npz --default-class "pillow"`
- `scene0222_01` IoU=0.000 gt=2571 pred=22
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0222_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0222_01.npz --default-class "pillow"`
- `scene0231_00` IoU=0.000 gt=302 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0231_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0231_00.npz --default-class "pillow"`

### box (idx=22)
- `scene0025_00` IoU=0.000 gt=1680 pred=426
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0025_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0025_00.npz --default-class "box"`
- `scene0139_00` IoU=0.000 gt=2233 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0139_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0139_00.npz --default-class "box"`
- `scene0164_00` IoU=0.000 gt=1119 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0164_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0164_00.npz --default-class "box"`
- `scene0144_00` IoU=0.000 gt=585 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0144_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0144_00.npz --default-class "box"`
- `scene0164_02` IoU=0.000 gt=945 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0164_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0164_02.npz --default-class "box"`

### bed (idx=10)
- `scene0353_01` IoU=0.153 gt=21655 pred=7471
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0353_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0353_01.npz --default-class "bed"`
- `scene0353_02` IoU=0.274 gt=26652 pred=8773
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0353_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0353_02.npz --default-class "bed"`
- `scene0426_01` IoU=0.329 gt=5030 pred=14862
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0426_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0426_01.npz --default-class "bed"`
- `scene0144_00` IoU=0.345 gt=10439 pred=4352
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0144_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0144_00.npz --default-class "bed"`
- `scene0426_02` IoU=0.391 gt=5287 pred=13500
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0426_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0426_02.npz --default-class "bed"`

### object (idx=172)
- `scene0686_00` IoU=0.000 gt=240 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0686_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0686_00.npz --default-class "object"`
- `scene0686_01` IoU=0.000 gt=359 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0686_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0686_01.npz --default-class "object"`
- `scene0690_00` IoU=0.000 gt=1708 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0690_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0690_00.npz --default-class "object"`
- `scene0693_00` IoU=0.000 gt=1093 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0693_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0693_00.npz --default-class "object"`
- `scene0695_00` IoU=0.000 gt=138 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0695_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0695_00.npz --default-class "object"`

### picture (idx=13)
- `scene0678_01` IoU=0.000 gt=2416 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0678_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0678_01.npz --default-class "picture"`
- `scene0685_00` IoU=0.000 gt=1694 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0685_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0685_00.npz --default-class "picture"`
- `scene0678_00` IoU=0.000 gt=2978 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0678_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0678_00.npz --default-class "picture"`
- `scene0685_01` IoU=0.000 gt=1921 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0685_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0685_01.npz --default-class "picture"`
- `scene0685_02` IoU=0.000 gt=1855 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0685_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0685_02.npz --default-class "picture"`

### couch (idx=5)
- `scene0558_00` IoU=0.000 gt=184 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0558_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0558_00.npz --default-class "couch"`
- `scene0131_00` IoU=0.074 gt=1329 pred=98
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0131_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0131_00.npz --default-class "couch"`
- `scene0647_00` IoU=0.105 gt=8671 pred=913
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0647_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0647_00.npz --default-class "couch"`
- `scene0647_01` IoU=0.166 gt=9169 pred=1522
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0647_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0647_01.npz --default-class "couch"`
- `scene0131_01` IoU=0.273 gt=6391 pred=1744
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0131_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0131_01.npz --default-class "couch"`

### blinds (idx=73)
- `scene0697_00` IoU=0.000 gt=7222 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0697_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0697_00.npz --default-class "blinds"`
- `scene0697_02` IoU=0.000 gt=16135 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0697_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0697_02.npz --default-class "blinds"`
- `scene0697_03` IoU=0.000 gt=11606 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0697_03 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0697_03.npz --default-class "blinds"`
- `scene0203_01` IoU=0.005 gt=14728 pred=158
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0203_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0203_01.npz --default-class "blinds"`
- `scene0697_01` IoU=0.006 gt=7741 pred=448
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0697_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0697_01.npz --default-class "blinds"`

### copier (idx=62)
- `scene0077_00` IoU=0.000 gt=5423 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0077_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0077_00.npz --default-class "copier"`
- `scene0552_01` IoU=0.000 gt=8335 pred=6
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0552_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0552_01.npz --default-class "copier"`
- `scene0462_00` IoU=0.001 gt=11379 pred=211
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0462_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0462_00.npz --default-class "copier"`
- `scene0552_00` IoU=0.006 gt=8701 pred=55
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0552_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0552_00.npz --default-class "copier"`
- `scene0704_01` IoU=0.009 gt=10202 pred=715
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0704_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0704_01.npz --default-class "copier"`

### radiator (idx=80)
- `scene0084_00` IoU=0.000 gt=1173 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0084_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0084_00.npz --default-class "radiator"`
- `scene0084_01` IoU=0.000 gt=1566 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0084_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0084_01.npz --default-class "radiator"`
- `scene0084_02` IoU=0.000 gt=1598 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0084_02 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0084_02.npz --default-class "radiator"`
- `scene0095_01` IoU=0.000 gt=1153 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0095_01 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0095_01.npz --default-class "radiator"`
- `scene0217_00` IoU=0.000 gt=1994 pred=0
    - `python scripts/visualize_eval_query.py --scene-dir /datasets/mosaic3d/data/scannet/scene0217_00 --pred-file logs/eval/runs/2026-06-26_02-22-42/eval_scene_dumps/scannet200/scene0217_00.npz --default-class "radiator"`

