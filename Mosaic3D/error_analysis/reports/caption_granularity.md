# Task 2 caption granularity vs eval FN destinations

## Self-check and source
- baseline fg-mIoU = **0.1548** (ref 0.1548; OK)
- anno_sources = `['gsam2', 'seem']` (ScanNet config default unless explicitly overridden)
- scanned scenes = 1201; caption regions counted = 1312860
- alignment smoke checks passed = 1
- parsing: longest class/alias regex; multiple terms each count once; no match -> `<none>`

## PRE-REGISTERED criteria
- CONFIRM: >50% top pairs have P(J|C)>0 and J rank #1 non-C, plus mean Spearman>=0.4
- REFUTE: median P(C|C)>=0.5 and mean Spearman<0.1
- otherwise: INCONCLUSIVE

## Summary
- top pairs analyzed = 25
- pair-confirm fraction = **0.240** (6/25)
- mean row-wise Spearman(caption terms, eval FN destinations) = **0.270**
- median P(caption=C | GT=C) = **0.238**
- **Verdict: INCONCLUSIVE**

## Per-pair comparison

| C | eval absorber J | cost | P(J|C) | P(C|C) | J rank among non-C | row Spearman | P(<none>) |
|---|---|---:|---:|---:|---:|---:|---:|
| chair | office chair | 1237268 | 0.0185 | 0.5872 | 5 | 0.392 | 0.0236 |
| shelf | bookshelf | 810674 | 0.0416 | 0.2378 | 2 | 0.202 | 0.1746 |
| cabinet | kitchen cabinet | 624692 | 0.0057 | 0.1285 | 22 | 0.226 | 0.2259 |
| window | blinds | 561323 | 0.0878 | 0.4275 | 1 | 0.188 | 0.1233 |
| door | window | 451372 | 0.0241 | 0.5860 | 2 | 0.309 | 0.1145 |
| table | desk | 373151 | 0.0859 | 0.4239 | 2 | 0.301 | 0.0735 |
| book | bookshelf | 352139 | 0.0646 | 0.4333 | 2 | 0.291 | 0.0840 |
| desk | dresser | 266085 | 0.0208 | 0.1985 | 8 | 0.435 | 0.1174 |
| trash can | trash bin | 254491 | 0.0564 | 0.3406 | 3 | 0.180 | 0.0847 |
| mailbox | bookshelf | 217967 | 0.0565 | 0.0000 | 5 | 0.068 | 0.2653 |
| curtain | window | 205013 | 0.2289 | 0.2902 | 1 | 0.315 | 0.1576 |
| bookshelf | shelf | 204247 | 0.2902 | 0.0957 | 1 | 0.299 | 0.1519 |
| clothes dryer | oven | 190001 | 0.0164 | 0.0009 | 11 | 0.154 | 0.3379 |
| whiteboard | board | 183777 | 0.0671 | 0.2324 | 2 | 0.398 | 0.0673 |
| doorframe | door | 175002 | 0.3944 | 0.0019 | 1 | 0.174 | 0.1305 |
| armchair | sofa chair | 151202 | 0.0072 | 0.0045 | 12 | 0.287 | 0.0359 |
| pillow | bed | 137850 | 0.2476 | 0.4335 | 1 | 0.440 | 0.0613 |
| box | crate | 133254 | 0.0075 | 0.3751 | 15 | 0.223 | 0.1364 |
| bed | mattress | 127571 | 0.0485 | 0.4823 | 3 | 0.353 | 0.1040 |
| object | bicycle | 125030 | 0.0001 | 0.0046 | 142 | 0.247 | 0.1368 |
| picture | poster | 122695 | 0.0382 | 0.2849 | 2 | 0.226 | 0.0974 |
| couch | bench | 113702 | 0.0090 | 0.5203 | 12 | 0.370 | 0.0648 |
| blinds | closet wall | 107859 | 0.0000 | 0.1113 | 94 | 0.118 | 0.2085 |
| copier | printer | 95293 | 0.2075 | 0.0003 | 1 | 0.313 | 0.2002 |
| radiator | vent | 92965 | 0.1393 | 0.0590 | 2 | 0.246 | 0.1718 |
