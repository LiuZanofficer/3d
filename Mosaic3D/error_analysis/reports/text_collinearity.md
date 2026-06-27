# V1a text-embedding collinearity

- model: `hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP`  use_prompt=True
- absorber pairs (top-25 cost): mean cosine **0.9082**
- absorber pairs (all fg, n=186): mean cosine **0.8769**
- random non-sibling baseline: mu=0.8135 sd=0.0321 p95=0.8652

## PRE-REGISTERED verdict
- rule: CONFIRM if cos_abs>=p95_rand(0.8652) AND >=mu+0.10(0.9135); REFUTE if cos_abs<=mu+0.02(0.8335)
- **cos_abs(top25)=0.9082 -> INCONCLUSIVE (escalate to V1b)**

## Absorber pairs (C -> top-1 FN destination J)

| C | J | cosine | FN-share | cost |
|---|---|---|---|---|
| chair | office chair | 0.9372 | 54.6% | 1237265 |
| shelf | bookshelf | 0.9523 | 45.5% | 810674 |
| cabinet | kitchen cabinet | 0.9514 | 14.5% | 624692 |
| window | blinds | 0.8848 | 36.9% | 561321 |
| door | window | 0.9072 | 21.9% | 451372 |
| table | desk | 0.9154 | 28.3% | 373151 |
| book | bookshelf | 0.8988 | 91.6% | 352139 |
| desk | dresser | 0.9013 | 11.3% | 266087 |
| trash can | trash bin | 0.9925 | 46.2% | 254491 |
| mailbox | bookshelf | 0.8346 | 35.8% | 217967 |
| curtain | window | 0.8905 | 30.5% | 205013 |
| bookshelf | shelf | 0.9523 | 49.8% | 204248 |
| clothes dryer | oven | 0.8554 | 46.8% | 190001 |
| whiteboard | board | 0.8908 | 76.9% | 183777 |
| doorframe | door | 0.9544 | 58.3% | 175005 |
| armchair | sofa chair | 0.9717 | 43.5% | 151202 |
| pillow | bed | 0.8897 | 62.2% | 137850 |
| box | crate | 0.9347 | 8.0% | 133254 |
| bed | mattress | 0.9305 | 16.0% | 127570 |
| object | bicycle | 0.8472 | 10.7% | 125030 |
| picture | poster | 0.8821 | 27.8% | 122695 |
| couch | bench | 0.9003 | 51.7% | 113702 |
| blinds | closet wall | 0.8169 | 22.1% | 107859 |
| copier | printer | 0.9561 | 41.3% | 95293 |
| radiator | vent | 0.8575 | 45.0% | 92965 |
