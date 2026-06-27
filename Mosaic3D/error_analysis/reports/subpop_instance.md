# S3 instance-level wholesale-flip test

- flip threshold = 0.5; pairs analyzed = 25
- **mean leak_in_flipped = 0.697** (share of FN->J points living in flipped instances)
- per-instance frac_J bimodality: near 0/1 = 0.80, middle(0.3-0.7) = 0.09

## PRE-REGISTERED verdict
- CONFIRM: mean>=0.60; REFUTE: mean<=0.30
- **CONFIRM (instance-level wholesale flip)**

## Per-pair

| C | absorber J | leak_in_flipped | flipped_inst_frac | n_FN->J | n_inst | cost |
|---|---|---|---|---|---|---|
| chair | office chair | 0.880 | 0.267 | 676121 | 1166 | 1237266 |
| shelf | bookshelf | 0.825 | 0.265 | 368566 | 155 | 810674 |
| cabinet | kitchen cabinet | 0.665 | 0.138 | 90547 | 159 | 624692 |
| window | blinds | 0.643 | 0.064 | 207153 | 282 | 561321 |
| door | window | 0.931 | 0.023 | 98980 | 306 | 451372 |
| table | desk | 0.701 | 0.092 | 105609 | 273 | 373151 |
| book | bookshelf | 0.985 | 0.814 | 322560 | 447 | 352139 |
| desk | dresser | 0.218 | 0.016 | 30133 | 127 | 266085 |
| trash can | trash bin | 0.828 | 0.350 | 117475 | 237 | 254491 |
| mailbox | bookshelf | 0.802 | 0.118 | 77925 | 17 | 217967 |
| curtain | window | 0.391 | 0.104 | 62440 | 67 | 205013 |
| bookshelf | shelf | 0.732 | 0.130 | 101747 | 77 | 204248 |
| clothes dryer | oven | 0.555 | 0.250 | 88987 | 12 | 190001 |
| whiteboard | board | 0.792 | 0.355 | 141390 | 76 | 183777 |
| doorframe | door | 0.870 | 0.579 | 102090 | 126 | 175005 |
| armchair | sofa chair | 0.707 | 0.176 | 65774 | 74 | 151202 |
| pillow | bed | 0.984 | 0.577 | 85793 | 213 | 137850 |
| box | crate | 0.454 | 0.011 | 10721 | 180 | 133254 |
| bed | mattress | 0.000 | 0.000 | 20388 | 70 | 127571 |
| object | bicycle | 0.716 | 0.014 | 13436 | 208 | 125030 |
| picture | poster | 0.824 | 0.321 | 34058 | 215 | 122695 |
| couch | bench | 0.593 | 0.072 | 58788 | 97 | 113702 |
| blinds | closet wall | 0.931 | 0.308 | 23794 | 13 | 107859 |
| copier | printer | 0.515 | 0.333 | 39344 | 15 | 95293 |
| radiator | vent | 0.879 | 0.585 | 41820 | 41 | 92965 |
