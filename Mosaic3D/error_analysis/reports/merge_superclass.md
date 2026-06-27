# V4 sibling-merge recompute (disjoint top-1 matching, size<=2)

- baseline (singleton) fg-mIoU = **0.1548** (ref 0.1548; self-check OK)
- merging is disjoint top-1 pairing (no transitive chaining); random control = random disjoint pairing of the SAME number of pairs.

| matching | #pairs | merged mIoU | random mIoU (same #pairs) | Delta | z | gain-over-baseline |
|---|---|---|---|---|---|---|
| confusion | 81 | 0.2456 | 0.1949±0.0064 | +0.0506 | 7.9 | +0.0908 |
| text | 88 | 0.2163 | 0.2012±0.0070 | +0.0152 | 2.2 | +0.0616 |

## PRE-REGISTERED verdict
- CONFIRM: conf z>3 AND text z>3 AND Delta_conf>=+0.02; PARTIAL: conf z>3 but text z<2; REFUTE: conf z<2
- **INCONCLUSIVE**
