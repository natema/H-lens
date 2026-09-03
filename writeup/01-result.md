# The result

## Claim

A development-averaged **diagonal Hessian correction** to the J-lens, applied at
layer 12 of Qwen3.5-4B, is statistically indistinguishable from applying no
correction — and indistinguishable from a version of itself whose coordinate
structure has been destroyed.

## Definitions used throughout

- **rank**: position of the concept token in a lens readout, out of 248,077.
- **geometric-mean rank**: `exp(mean(log rank))`. Ranks are heavy-tailed; on one
  set the arithmetic mean was 27,758 against a geometric mean of 1,249, the
  former driven entirely by a few six-figure failures.
- **wins/losses**: per-item paired comparison of the two ranks, ties excluded.
  Significance is a two-sided exact sign test.
- **shuffled control** (`j2_shuffled`): the same operator with its diagonal rows
  rolled by one coordinate. Preserves the row-norm distribution, destroys the
  coordinate-to-curvature correspondence. This is what the correction must beat
  for a gain to be attributable to curvature.

## Headline numbers (n = 3344, layer 12)

| method | geo-mean rank | top-10 | vs J-lens | p |
|---|---:|---:|---:|---:|
| logit lens | 5328.6 | 2.5% | 194/3131 | ~0 |
| J-lens | 103.5 | 25.5% | — | — |
| J² real | 103.4 | 25.6% | 1375/1317 | **0.27** |
| J² shuffled | 102.4 | 25.6% | 1381/1242 | **0.007** |
| R-lens | 138.6 | 21.9% | 1245/1897 | 2e-31 |

The real diagonal does not beat J-lens. Its own destroyed version does.

## The finding that matters most methodologically

Restricted to the cell where the J-lens fails, **J² beats J-lens 601/510 with
p = 0.0069**. Reported alone that is a significant improvement.

On the same cell the shuffled control wins **630/463 at p = 4.9e-07** — a
stronger effect from an operator carrying no coordinate information. The apparent
gain is regression to the mean on a set *defined* by J-lens failure.

This is the write-up's most transferable point: **an interpretability result
evaluated only on cases selected for a baseline's failure will show improvement
whether or not the method works.** The control and the unselected cells are what
distinguish the two.

## Where the correction is slightly harmful

| cell | n | J-lens | J² real | top-10 |
|---|---:|---:|---:|---|
| `both` | 640 | 4.8 | 4.9 | 86.7% → 84.7% |
| `lens_only` | 375 | 6.9 | 7.0 | 78.9% → 77.1% |
| `neither` | 1160 | 548.2 | 550.1 | — |

139/153, 88/88, 547/566 — none significant, all slightly negative. Only visible
because the unselected cells were scored.

## Robustness

- Restricting to quality-`strong` items: 335/336, p = 1.00.
- The correction also raises development-set normalized error rather than
  lowering it, at both layers, on the data it was estimated from: layer 6
  0.972865 → 0.973037, layer 12 0.935327 → 0.935511. **Caveat:** these
  differences are ~0.02% of the target norm and the metric barely beats
  predicting zero (J-lens 0.9729 against 1.0), so it is weak corroboration, not
  the main evidence. An earlier draft of this argument overstated it.

## Secondary finding worth reporting: the lenses are complementary

R-lens is clearly worse than J-lens on average at layer 12 (138.6 vs 103.5,
p = 2e-31) but **not uniformly**. On the failure cell it reaches top-10 for
**4.9%** of items against J-lens's **0.1%** — it recovers about one in twenty of
the failures J-lens cannot reach, while being worse elsewhere. J² manages 1.3%
there and its shuffled control 1.2%.

Separately, on the 33-case hand-built battery R-lens beats J-lens at layer 6
(844 vs 1507 geo-mean) but loses at layer 12 (242 vs 211). That independently
reproduces the R-lens post's own early-layer claim and is a useful check that the
harness measures something real.

The story is not a ranking. The lenses fail on different items.

## Robustness to the self-report prompt

Regenerating the self-reports with a materially different prompt (few-shot,
sentence last, no synonym ban; naming rate 54.1% → 60.2%) moved every cell
count but **no ranking**. J² vs J-lens failures at layer 12, all items: exactly
+0; shuffled control −18. R-lens at layer 6: −161 (was −141). The comparisons
are invariant to a six-point shift in the instrument, which is what one wants
from a measurement of the lenses rather than of the prompt. Numbers in
`data_v2/README.md`.
