# S1 boundary subpopulation test

- k=16 neighbors; boundary_score = frac of 3D neighbors with GT != victim class
- pairs analyzed: 23 (top-25 cost)
- **mean per-pair diff (FN->J minus TP) = +0.0066**, frac pairs positive = 0.61

## PRE-REGISTERED verdict
- CONFIRM: diff>=+0.15 AND frac_pos>=0.70; REFUTE: diff<=+0.03
- **REFUTE (not a boundary effect)**

## Per-pair (boundary score)

| C | absorber J | TP bnd | FN->J bnd | diff | n_TP | n_FN->J | cost |
|---|---|---|---|---|---|---|---|
| chair | office chair | 0.018 | 0.024 | +0.006 | 1254351 | 676121 | 1237266 |
| shelf | bookshelf | 0.023 | 0.021 | -0.002 | 359205 | 368566 | 810674 |
| cabinet | kitchen cabinet | 0.040 | 0.030 | -0.009 | 133415 | 90547 | 624692 |
| window | blinds | 0.028 | 0.009 | -0.019 | 823809 | 207153 | 561321 |
| door | window | 0.038 | 0.014 | -0.025 | 840261 | 98980 | 451372 |
| table | desk | 0.028 | 0.059 | +0.032 | 851547 | 105609 | 373151 |
| book | bookshelf | 0.096 | 0.108 | +0.012 | 47 | 322560 | 352139 |
| desk | dresser | 0.062 | 0.070 | +0.008 | 389078 | 30133 | 266085 |
| trash can | trash bin | 0.062 | 0.032 | -0.030 | 44503 | 117475 | 254491 |
| curtain | window | 0.026 | 0.032 | +0.006 | 452350 | 62440 | 205013 |
| bookshelf | shelf | 0.050 | 0.025 | -0.025 | 921070 | 101747 | 204248 |
| clothes dryer | oven | 0.009 | 0.004 | -0.005 | 16785 | 88987 | 190001 |
| whiteboard | board | 0.038 | 0.047 | +0.008 | 139805 | 141390 | 183777 |
| doorframe | door | 0.109 | 0.177 | +0.068 | 10358 | 102090 | 175005 |
| armchair | sofa chair | 0.021 | 0.032 | +0.010 | 169355 | 65774 | 151202 |
| pillow | bed | 0.028 | 0.092 | +0.064 | 15658 | 85793 | 137850 |
| box | crate | 0.051 | 0.086 | +0.035 | 45613 | 10721 | 133254 |
| bed | mattress | 0.041 | 0.048 | +0.006 | 713712 | 20388 | 127571 |
| picture | poster | 0.072 | 0.073 | +0.001 | 61841 | 34058 | 122695 |
| couch | bench | 0.034 | 0.009 | -0.024 | 676067 | 58788 | 113702 |
| blinds | closet wall | 0.014 | 0.024 | +0.010 | 42641 | 23794 | 107859 |
| copier | printer | 0.022 | 0.006 | -0.016 | 4410 | 39344 | 95293 |
| radiator | vent | 0.028 | 0.068 | +0.040 | 3473 | 41820 | 92965 |
