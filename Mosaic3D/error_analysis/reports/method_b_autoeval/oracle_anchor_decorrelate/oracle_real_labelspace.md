# Task 0 real 200-class oracle ceiling

## Self-check
- baseline fg-mIoU = **0.1549** (ref 0.1548; OK)
- flip threshold = 0.5
- scoring: pred_semantic, background->ignore, invalid labels dropped, fg_class_idx mean

## PRE-REGISTERED criteria
- <=0.18: low ceiling / warn
- >=0.21: physical room for 20+ fg-mIoU
- otherwise: modest room

## Oracle results

| top-k victim classes | n_classes | oracle fg-mIoU | gain over baseline | verdict |
|---|---:|---:|---:|---|
| 25 | 25 | 0.2045 | +0.0495 | MODEST_ROOM (0.18,0.21): continue, but lower expectations |

## Per-class contribution (25)

| C | absorber J | cost | corrected_inst | corrected_points | base_iou | oracle_iou | delta_iou |
|---|---|---:|---:|---:|---:|---:|---:|
| shelf | bookshelf | 721368 | 41 | 400701 | 0.2876 | 0.6363 | +0.3487 |
| cabinet | kitchen cabinet | 571970 | 17 | 66451 | 0.2180 | 0.2967 | +0.0787 |
| chair | office chair | 529514 | 95 | 249843 | 0.7266 | 0.8694 | +0.1428 |
| desk | table | 405453 | 31 | 159146 | 0.3136 | 0.5280 | +0.2143 |
| window | door | 388020 | 24 | 111484 | 0.5962 | 0.7322 | +0.1360 |
| book | bookshelf | 351710 | 372 | 330923 | 0.0013 | 0.9389 | +0.9376 |
| trash can | trash bin | 279318 | 68 | 102240 | 0.0654 | 0.4049 | +0.3395 |
| whiteboard | board | 262619 | 45 | 243637 | 0.1661 | 0.8090 | +0.6429 |
| table | desk | 223194 | 12 | 34997 | 0.6189 | 0.7395 | +0.1206 |
| mailbox | shelf | 217967 | 3 | 105945 | 0.0000 | 0.4861 | +0.4861 |
| door | window | 211049 | 10 | 114811 | 0.5582 | 0.7108 | +0.1527 |
| clothes dryer | oven | 202419 | 6 | 144152 | 0.0198 | 0.6566 | +0.6369 |
| shower wall | shower door | 200218 | 20 | 150591 | 0.0308 | 0.7381 | +0.7073 |
| doorframe | door | 183126 | 106 | 156946 | 0.0119 | 0.8433 | +0.8314 |
| bookshelf | shelf | 156321 | 12 | 131490 | 0.4894 | 0.8695 | +0.3801 |
| shower curtain | shower curtain rod | 147922 | 19 | 77508 | 0.0636 | 0.5222 | +0.4586 |
| armchair | chair | 135864 | 7 | 32050 | 0.4287 | 0.5032 | +0.0744 |
| office chair | chair | 129942 | 51 | 129688 | 0.2881 | 0.9592 | +0.6711 |
| coffee table | table | 125901 | 20 | 76849 | 0.1807 | 0.6098 | +0.4291 |
| object | bicycle | 125030 | 3 | 20633 | 0.0000 | 0.1649 | +0.1649 |
| pillow | bed | 124939 | 110 | 81314 | 0.1588 | 0.6056 | +0.4469 |
| backpack | bag | 115377 | 31 | 33629 | 0.2304 | 0.4417 | +0.2113 |
| box | crate | 103888 | 3 | 8304 | 0.2837 | 0.3169 | +0.0332 |
| curtain | window | 101905 | 8 | 72414 | 0.6835 | 0.8031 | +0.1197 |
| blinds | curtain | 98872 | 3 | 44306 | 0.1940 | 0.3605 | +0.1665 |
