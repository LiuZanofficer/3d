# V1b winner-take-all direction

- absorber pairs from top-25 cost fg classes (n=25)
- **r_winner = 0.320** (absorber text closer to victim visual mean than victim's own text)
- mean margin (sim_abs - sim_self) = -0.0027
- random-class control r_winner_rand = 0.011

## PRE-REGISTERED verdict
- CONFIRM if r_winner>=0.70; REFUTE if r_winner<=0.50; else INCONCLUSIVE
- **REFUTE (no better than coin flip)**

## Per-pair (C -> absorber J)

| C | J | sim_self | sim_abs | absorber wins | cost |
|---|---|---|---|---|---|
| chair | office chair | 0.0215 | 0.0164 | no | 1237266 |
| shelf | bookshelf | 0.0208 | 0.0196 | no | 810674 |
| cabinet | kitchen cabinet | 0.0166 | 0.0137 | no | 624692 |
| window | blinds | 0.0097 | 0.0044 | no | 561321 |
| door | window | 0.0142 | -0.0048 | no | 451372 |
| table | desk | 0.0276 | 0.0164 | no | 373151 |
| book | bookshelf | 0.0220 | 0.0399 | YES | 352139 |
| desk | dresser | 0.0296 | 0.0132 | no | 266085 |
| trash can | trash bin | 0.0247 | 0.0255 | YES | 254491 |
| mailbox | bookshelf | 0.0040 | 0.0225 | YES | 217967 |
| curtain | window | 0.0230 | 0.0130 | no | 205013 |
| bookshelf | shelf | 0.0354 | 0.0297 | no | 204248 |
| clothes dryer | oven | 0.0035 | 0.0085 | YES | 190001 |
| whiteboard | board | 0.0252 | 0.0253 | YES | 183777 |
| doorframe | door | 0.0037 | 0.0061 | YES | 175005 |
| armchair | sofa chair | 0.0272 | 0.0265 | no | 151202 |
| pillow | bed | 0.0321 | 0.0315 | no | 137850 |
| box | crate | 0.0196 | 0.0181 | no | 133254 |
| bed | mattress | 0.0376 | 0.0300 | no | 127571 |
| object | bicycle | 0.0040 | 0.0001 | no | 125030 |
| picture | poster | -0.0011 | -0.0028 | no | 122695 |
| couch | bench | 0.0401 | 0.0261 | no | 113702 |
| blinds | closet wall | 0.0180 | 0.0077 | no | 107859 |
| copier | printer | 0.0261 | 0.0272 | YES | 95293 |
| radiator | vent | 0.0031 | 0.0064 | YES | 92965 |
