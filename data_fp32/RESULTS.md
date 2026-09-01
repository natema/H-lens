# H-lens versus J-lens on the J-space dataset

Layer 12, 3,344 items, operator fitted on pile-10k with 128 Hessian pairs and
2,947 moment pairs and no knowledge of these items.

Every cell is scored, not only the cell where the J-lens fails. That cell is
*defined* by J-lens failure, so J-lens is conditioned to be at its worst there
and regression to the mean flatters any alternative, including a worse one. The
cells where J-lens does well are the only place a correction can be caught doing
damage.

`j2_shuffled` applies the same operator with its diagonal rows rolled by one
coordinate. It destroys the correspondence between a coordinate and its curvature
while preserving the row-norm distribution, so it is what the correction must
beat to claim the gain comes from curvature.

## All items (n = 3344)

| method | geo-mean rank | median | top-10 | wins/losses vs J-lens | p |
|---|---:|---:|---:|---:|---:|
| logit lens | 5328.6 | 8732 | 2.5% | 194/3131 | ~0 |
| J-lens | 103.5 | 78 | 25.5% | — | — |
| J² real | 103.4 | 75 | 25.6% | 1375/1317 | 0.27 |
| **J² shuffled** | **102.4** | 76 | 25.6% | 1381/1242 | **0.007** |
| R-lens | 138.6 | 120 | 21.9% | 1245/1897 | 2e-31 |

The real diagonal is indistinguishable from J-lens (p = 0.27). The shuffled
control significantly beats it (p = 0.007). Whatever small benefit the correction
carries does not depend on which curvature vector belongs to which coordinate.

## By cell

| cell | n | J-lens | J² real | J² shuffled | R-lens |
|---|---:|---:|---:|---:|---:|
| `self_report_only` | 1169 | 252.6 | 248.6 | **244.2** | 321.3 |
| `both` | 640 | **4.8** | 4.9 | 4.8 | 7.3 |
| `lens_only` | 375 | **6.9** | 7.0 | 7.0 | 10.8 |
| `neither` | 1160 | **548.2** | 550.1 | 546.4 | 689.9 |

Geometric-mean rank; lower is better.

### The selection trap, demonstrated

Restricted to `self_report_only`, J² beats J-lens 601 to 510 with **p = 0.0069**.
Read alone, that is a significant improvement and would have been reported as
one. But on the same cell the shuffled control wins 630 to 463 at **p = 4.9e-07**
— a stronger effect from an operator whose coordinate structure has been
destroyed. The apparent gain is regression to the mean on a set selected for
J-lens failure, not recovered curvature.

Outside that cell the correction does nothing: 139/153 on `both`, exactly 88/88
on `lens_only`, 547/566 on `neither`, no result significant. It also costs a
little where the J-lens already works, with top-10 falling from 86.7% to 84.7% on
`both` and 78.9% to 77.1% on `lens_only`.

## Restricting to well-formed items does not change it

Quality `strong`, all cells (n = 826): J² wins 335 and loses 336, p = 1.00. The
shuffled control wins 353 and loses 316.

## R-lens

R-lens is clearly worse than J-lens on average at layer 12 (geo-mean 138.6 versus
103.5, p = 2e-31), but it is not uniformly worse. On the `self_report_only` cell
it puts the concept in the top 10 for **4.9%** of items against J-lens's **0.1%**
— it recovers roughly one in twenty of the failures the J-lens cannot reach,
while being worse elsewhere. J² reaches 1.3% there, and its shuffled control
1.2%, so J²'s recoveries are not curvature either.

That trade is the interesting structure in this data: the two lenses fail on
different items rather than one dominating.

## Conclusion

On 3,344 held-out items with an operator fitted on unrelated text, a
development-averaged diagonal Hessian correction to the J-lens at layer 12 is
statistically indistinguishable from no correction, and is not distinguishable
from a version of itself with the coordinate structure destroyed. This is
evidence against the primary hypothesis in `PROJECT_IDEA.md`, and consistent with
its alternative that apparent gains come from rescaling rather than curvature.
