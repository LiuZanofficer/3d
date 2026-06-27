# V2 part-whole geometric containment
- surface KNN distance = 5 cm; frac is point-weighted over scenes

## book -> bookshelf
- scenes=17, target points (gt=book & pred=bookshelf)=322158
- frac within 5cm of true bookshelf: **0.405**
- random-container control: 0.000  (diff=+0.405)
- **INCONCLUSIVE**

## pillow -> bed
- scenes=35, target points (gt=pillow & pred=bed)=85768
- frac within 5cm of true bed: **0.259**
- random-container control: 0.003  (diff=+0.256)
- **REFUTE (not containment)**

## doorframe -> door
- scenes=88, target points (gt=doorframe & pred=door)=97283
- frac within 5cm of true door: **0.272**
- random-container control: 0.012  (diff=+0.261)
- **REFUTE (not containment)**

## PRE-REGISTERED verdict rule
- CONFIRM: frac_in>=0.70 AND (frac_in-frac_rand)>=0.30; REFUTE: frac_in<=0.30 OR diff<=0.10
