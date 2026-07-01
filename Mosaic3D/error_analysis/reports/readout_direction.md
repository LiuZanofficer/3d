# Step 0 readout direction: visual LDA vs text anchor

## Self-check
- baseline fg-mIoU = **0.1548** (ref 0.1548; OK)
- valid pairs = 22 / top-25
- GT labels are used for this analysis only, not for any method.

## PRE-REGISTERED criteria
- CONFIRM if majority pairs have |cos|<0.5 and mean is not > random by 0.05
- REFUTE if mean |cos|>=0.5
- otherwise INCONCLUSIVE

## Summary
- mean |cos(d_vis,d_txt)| = **0.034**; median=0.029
- fraction <0.5 = **1.000**
- random direction mean |cos| = **0.028**
- **Verdict: CONFIRM (text readout direction differs from visual discriminant)**

## Per-pair

| C | J | cost | n_C | n_J | abs_cos | random_abs_cos | status |
|---|---|---:|---:|---:|---:|---:|---|
| chair | office chair | 1237268 | 1166 | 105 | 0.128 | 0.033 | OK |
| shelf | bookshelf | 810674 | 155 | 77 | 0.040 | 0.018 | OK |
| cabinet | kitchen cabinet | 624692 | 159 | 135 | 0.010 | 0.002 | OK |
| window | blinds | 561323 | 282 | 13 | 0.006 | 0.024 | OK |
| door | window | 451372 | 306 | 282 | 0.069 | 0.055 | OK |
| table | desk | 373151 | 273 | 127 | 0.009 | 0.089 | OK |
| book | bookshelf | 352139 | 447 | 77 | 0.005 | 0.003 | OK |
| desk | dresser | 266085 | 127 | 41 | 0.043 | 0.034 | OK |
| trash can | trash bin | 254491 | 237 | 15 | 0.088 | 0.045 | OK |
| mailbox | bookshelf | 217967 | 17 | 77 | 0.003 | 0.003 | OK |
| curtain | window | 205013 | 67 | 282 | 0.053 | 0.005 | OK |
| bookshelf | shelf | 204247 | 77 | 155 | 0.040 | 0.041 | OK |
| clothes dryer | oven | 190001 | 12 | 8 | 0.041 | 0.020 | OK |
| whiteboard | board | 183777 | 76 | 21 | 0.007 | 0.061 | OK |
| doorframe | door | 175002 | 126 | 306 | 0.007 | 0.041 | OK |
| armchair | sofa chair | 151202 | 74 | 19 | 0.064 | 0.024 | OK |
| pillow | bed | 137850 | 213 | 70 | 0.002 | 0.007 | OK |
| box | crate | 133254 | 180 | 4 | nan | nan | LOW_SAMPLE |
| bed | mattress | 127571 | 70 | 10 | 0.041 | 0.028 | OK |
| object | bicycle | 125030 | 208 | 0 | nan | nan | LOW_SAMPLE |
| picture | poster | 122695 | 215 | 7 | 0.002 | 0.036 | OK |
| couch | bench | 113702 | 97 | 18 | 0.012 | 0.004 | OK |
| blinds | closet wall | 107859 | 13 | 6 | 0.018 | 0.019 | OK |
| copier | printer | 95293 | 15 | 22 | 0.069 | 0.027 | OK |
| radiator | vent | 92965 | 41 | 4 | nan | nan | LOW_SAMPLE |
