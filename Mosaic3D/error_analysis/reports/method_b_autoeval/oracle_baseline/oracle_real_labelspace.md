# Task 0 real 200-class oracle ceiling

## Self-check
- baseline fg-mIoU = **0.1380** (ref 0.1548; MISMATCH)
- flip threshold = 0.5
- scoring: pred_semantic, background->ignore, invalid labels dropped, fg_class_idx mean

## PRE-REGISTERED criteria
- <=0.18: low ceiling / warn
- >=0.21: physical room for 20+ fg-mIoU
- otherwise: modest room

## Oracle results

| top-k victim classes | n_classes | oracle fg-mIoU | gain over baseline | verdict |
|---|---:|---:|---:|---|
| 25 | 25 | 0.1751 | +0.0372 | LOW_CEILING (<=0.18): warn, this line alone is unlikely to be enough |

## Per-class contribution (25)

| C | absorber J | cost | corrected_inst | corrected_points | base_iou | oracle_iou | delta_iou |
|---|---|---:|---:|---:|---:|---:|---:|
| shelf | bookshelf | 776206 | 42 | 407787 | 0.2638 | 0.5527 | +0.2889 |
| chair | office chair | 656559 | 90 | 224525 | 0.6770 | 0.7795 | +0.1025 |
| cabinet | dresser | 595442 | 11 | 51025 | 0.1874 | 0.2523 | +0.0649 |
| desk | table | 486764 | 7 | 35643 | 0.2218 | 0.2673 | +0.0455 |
| window | blinds | 456435 | 11 | 85864 | 0.5271 | 0.6252 | +0.0982 |
| book | bookshelf | 340946 | 354 | 315321 | 0.0312 | 0.8869 | +0.8557 |
| table | desk | 335830 | 4 | 17670 | 0.5454 | 0.5818 | +0.0364 |
| trash can | trash bin | 270844 | 51 | 77470 | 0.0931 | 0.3222 | +0.2291 |
| whiteboard | board | 257016 | 36 | 213846 | 0.1852 | 0.7141 | +0.5289 |
| door | window | 247541 | 9 | 113422 | 0.5541 | 0.6630 | +0.1090 |
| bookshelf | shelf | 220562 | 11 | 125440 | 0.4570 | 0.7955 | +0.3384 |
| mailbox | bookshelf | 217964 | 2 | 85597 | 0.0000 | 0.3927 | +0.3927 |
| clothes dryer | oven | 198211 | 7 | 154093 | 0.0354 | 0.6430 | +0.6075 |
| doorframe | door | 182889 | 109 | 163699 | 0.0127 | 0.8429 | +0.8302 |
| curtain | window | 172347 | 11 | 88235 | 0.5997 | 0.6901 | +0.0904 |
| shower wall | shower | 164019 | 12 | 90219 | 0.1939 | 0.5128 | +0.3189 |
| office chair | chair | 141794 | 52 | 136184 | 0.2619 | 0.7076 | +0.4457 |
| armchair | couch | 140423 | 5 | 20330 | 0.4003 | 0.4437 | +0.0434 |
| object | bicycle | 125026 | 2 | 13729 | 0.0000 | 0.1097 | +0.1097 |
| coffee table | table | 122407 | 19 | 77439 | 0.2044 | 0.5442 | +0.3398 |
| blinds | window | 120385 | 3 | 38489 | 0.1080 | 0.2921 | +0.1841 |
| kitchen cabinet | cabinet | 119069 | 5 | 19871 | 0.6313 | 0.6677 | +0.0363 |
| pillow | bed | 118296 | 108 | 79288 | 0.1818 | 0.5393 | +0.3575 |
| backpack | bag | 116376 | 22 | 23758 | 0.2191 | 0.3516 | +0.1325 |
| box | basket | 105707 | 5 | 6292 | 0.2508 | 0.2774 | +0.0265 |
