# H-lens versus J-lens on the J-space dataset

> The tables here were computed with the earlier self-report prompt and with
> every method's *raw* top-10 list. Since 2026-09-03 the record's cells are
> defined on the J-lens top-50 collapsed to 10 distinct concepts; the current
> counts, the raw-versus-collapsed comparison, and the prompt-change comparison
> showing every ranking invariant are in `README.md`. The narrative and the
> selection-trap demonstration are unchanged in substance.

Layer 12, 3,344 items, operator fitted on pile-10k with 128 Hessian pairs and
2,947 moment pairs and no knowledge of these items.

Every cell is scored, not only the cell where the J-lens fails. That cell is
*defined* by J-lens failure, so J-lens is conditioned to be at its worst there
and regression to the mean flatters any alternative, including a worse one. The
cells where J-lens does well are the only place a correction can be caught doing
damage.

## What the numbers are

**Geometric-mean rank** is `exp(mean(log(rank)))` of the concept token in the
readout, out of a 248,077-token vocabulary; lower is better. Geometric rather
than arithmetic because ranks are heavy-tailed — on one set the arithmetic mean
was 27,758 against a geometric mean of 1,249, the former being a handful of
six-figure failures.

**wins/losses** is a per-item paired comparison: for each item, the rank of the
concept under the two methods, counting how often one is strictly lower. Ties are
excluded, so the two numbers need not sum to n. On the 1,169-item
`self_report_only` cell, J² versus J-lens is 601 wins, 510 losses, 58 ties. The
p-value is a two-sided exact sign test on the wins and losses.

**top-10** is the share of items whose concept token appears in the readout's
first ten entries.

`j2_shuffled` applies the same operator with its diagonal rows rolled by one
coordinate. It destroys the correspondence between a coordinate and its curvature
while preserving the row-norm distribution, so it is what the correction must
beat to claim the gain comes from curvature.

## All items (n = 3344)

| method | geo-mean rank | median rank | top-10 | wins/losses vs J-lens | p |
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

Geometric-mean rank of the concept token, lower is better:

| cell | n | J-lens | J² real | J² shuffled | R-lens |
|---|---:|---:|---:|---:|---:|
| `self_report_only` | 1169 | 252.6 | 248.6 | **244.2** | 321.3 |
| `both` | 640 | **4.8** | 4.9 | 4.8 | 7.3 |
| `lens_only` | 375 | **6.9** | 7.0 | 7.0 | 10.8 |
| `neither` | 1160 | **548.2** | 550.1 | 546.4 | 689.9 |

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

## Cell counts, the metric the dataset is built on

Comparing methods by the rank of one token answers a different question from the
one the cells ask. The cells are assigned by a judge deciding whether the top-k
readout *names* the concept, which credits a variant (` trail` for *path*). So
each method's own top-k list is put through the identical procedure: same judge,
same prompt, same vocabulary evidence, same batch size. The self-report side
cannot change, since it does not depend on any lens, so every change in the cells
is a change in what the lens surfaced.

`self_report_only` means the lens failed; lower is better.

### Layer 12, quality `strong` (n = 826)

| method | self_report_only | both | lens_only | neither |
|---|---:|---:|---:|---:|
| J-lens | 359 | 162 | 66 | 239 |
| J² real | 359 (+0) | 162 (+0) | 68 (+2) | 237 (−2) |
| J² shuffled | 353 (−6) | 168 (+6) | 72 (+6) | 233 (−6) |
| R-lens | 366 (+7) | 155 (−7) | 70 (+4) | 235 (−4) |

The correction moves **zero** items out of the failure cell, and leaves `both`
unchanged. Its shuffled control moves six out.

### Layer 6, quality `strong` (n = 826)

| method | self_report_only | both | lens_only | neither |
|---|---:|---:|---:|---:|
| J-lens | 503 | 18 | 5 | 300 |
| J² real | 506 (+3) | 15 (−3) | 4 (−1) | 301 (+1) |
| J² shuffled | 505 (+2) | 16 (−2) | 4 (−1) | 301 (+1) |
| **R-lens** | **463 (−40)** | **58 (+40)** | 24 (+19) | 281 (−19) |

### All items

| | J-lens | J² real | J² shuffled | R-lens |
|---|---:|---:|---:|---:|
| layer 6, failures | 1691 | 1689 (−2) | 1691 (+0) | **1550 (−141)** |
| layer 12, failures | 1169 | 1178 (+9) | 1158 (−11) | 1256 (+87) |

## The dataset validates itself

R-lens reverses across layers: it removes 141 failures at layer 6 and adds 87 at
layer 12 (40 removed and 7 added on `strong`). That independently reproduces the
R-lens post's central claim of improved *early-layer* faithfulness, on 3,344
items constructed without reference to it, and reproduces the direction of the
reversal rather than a uniform preference.

This matters for reading the negative result. On the same instrument, at the same
layer, with the same judge, R-lens moves 141 items and J² moves 2. The correction
is not failing because the measurement is blunt: the measurement demonstrably
detects a real lens improvement roughly seventy times larger than anything the
correction produces.

At layer 6 the J-lens gets only 18 of 826 strong items into `both`, against 162
at layer 12, which is a direct measure of how much harder early-layer readout is
— the regime this project set out to improve.
