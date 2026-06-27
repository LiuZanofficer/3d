# V3 dead-class caption coverage vs IoU

- sources=['gsam2', 'seem'], caption files=2402, total captions=1331778
- Spearman(log(freq+1), IoU) over 196 fg classes (excl. 'object'): **rho=0.406** (p=3.44e-09)
- median freq: dead(IoU<0.01, n=64)=560  live(n=132)=1301
- dead classes with ZERO caption coverage: 5/64 -> ['toilet seat cover dispenser', 'handicap bar', 'music stand', 'dumbbell', 'storage organizer']

## PRE-REGISTERED verdict
- CONFIRM: rho>=+0.40 AND median(freq[dead])<=0.25*median(freq[live]); REFUTE: rho<=+0.10 OR median(freq[dead])>=median(freq[live])
- **INCONCLUSIVE**

## Lowest-coverage fg classes

| class | iou | caption_freq |
|---|---|---|
| toilet seat cover dispenser | 0.004 | 0 |
| handicap bar | 0.001 | 0 |
| music stand | 0.000 | 0 |
| dumbbell | 0.000 | 0 |
| closet rod | 0.039 | 0 |
| case of water bottles | 0.028 | 0 |
| storage organizer | 0.000 | 0 |
| paper cutter | 0.019 | 1 |
| mailbox | 0.000 | 2 |
| laundry detergent | 0.083 | 2 |
| copier | 0.043 | 3 |
| dustpan | 0.000 | 3 |
| clothes dryer | 0.059 | 4 |
| keyboard piano | 0.017 | 4 |
| coffee kettle | 0.024 | 5 |
| closet wall | 0.013 | 7 |
| stair rail | 0.196 | 7 |
| fire alarm | 0.000 | 8 |
| paper towel roll | 0.087 | 9 |
| cd case | 0.000 | 10 |
| laundry hamper | 0.022 | 20 |
| recycling bin | 0.162 | 25 |
| bathroom stall door | 0.119 | 25 |
| fire extinguisher | 0.005 | 28 |
| dish rack | 0.011 | 31 |
| structure | 0.000 | 33 |
| shower curtain rod | 0.060 | 34 |
| coat rack | 0.009 | 35 |
| range hood | 0.283 | 44 |
| water pitcher | 0.000 | 51 |
