# Step 0 readout direction: visual LDA vs text anchor

## Self-check
- baseline fg-mIoU = **0.1380** (ref 0.1548; MISMATCH)
- valid pairs = 23 / top-25
- GT labels are used for this analysis only, not for any method.

## PRE-REGISTERED criteria
- CONFIRM if majority pairs have |cos|<0.5 and mean is not > random by 0.05
- REFUTE if mean |cos|>=0.5
- otherwise INCONCLUSIVE

## Summary
- mean |cos(d_vis,d_txt)| = **0.031**; median=0.020
- fraction <0.5 = **1.000**
- random direction mean |cos| = **0.025**
- **Verdict: CONFIRM (text readout direction differs from visual discriminant)**

## Per-pair

| C | J | cost | n_C | n_J | abs_cos | random_abs_cos | status |
|---|---|---:|---:|---:|---:|---:|---|
| shelf | bookshelf | 776206 | 155 | 77 | 0.002 | 0.018 | OK |
| chair | office chair | 656559 | 1166 | 105 | 0.092 | 0.065 | OK |
| cabinet | dresser | 595442 | 159 | 41 | 0.018 | 0.032 | OK |
| desk | table | 486764 | 127 | 273 | 0.014 | 0.019 | OK |
| window | blinds | 456435 | 282 | 13 | 0.039 | 0.004 | OK |
| book | bookshelf | 340946 | 447 | 77 | 0.015 | 0.017 | OK |
| table | desk | 335830 | 273 | 127 | 0.014 | 0.017 | OK |
| trash can | trash bin | 270844 | 237 | 15 | 0.013 | 0.013 | OK |
| whiteboard | board | 257016 | 76 | 21 | 0.006 | 0.066 | OK |
| door | window | 247541 | 306 | 282 | 0.066 | 0.031 | OK |
| bookshelf | shelf | 220562 | 77 | 155 | 0.002 | 0.021 | OK |
| mailbox | bookshelf | 217964 | 17 | 77 | 0.023 | 0.046 | OK |
| clothes dryer | oven | 198211 | 12 | 8 | 0.069 | 0.020 | OK |
| doorframe | door | 182889 | 126 | 306 | 0.085 | 0.027 | OK |
| curtain | window | 172347 | 67 | 282 | 0.049 | 0.001 | OK |
| shower wall | shower | 164019 | 26 | 7 | 0.021 | 0.022 | OK |
| office chair | chair | 141794 | 105 | 1166 | 0.092 | 0.030 | OK |
| armchair | couch | 140423 | 74 | 97 | 0.020 | 0.022 | OK |
| object | bicycle | 125026 | 208 | 0 | nan | nan | LOW_SAMPLE |
| coffee table | table | 122407 | 43 | 273 | 0.005 | 0.028 | OK |
| blinds | window | 120385 | 13 | 282 | 0.039 | 0.053 | OK |
| kitchen cabinet | cabinet | 119069 | 135 | 159 | 0.011 | 0.010 | OK |
| pillow | bed | 118296 | 213 | 70 | 0.007 | 0.002 | OK |
| backpack | bag | 116376 | 147 | 58 | 0.022 | 0.015 | OK |
| box | basket | 105707 | 180 | 3 | nan | nan | LOW_SAMPLE |
