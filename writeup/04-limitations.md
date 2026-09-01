# Limitations

## Scope

- **One model.** Qwen3.5-4B only. Nothing here shows the diagonal correction
  fails on larger models, or on the DeepSeek model the original figures used.
- **One layer for the dataset.** The J-space dataset adjudicates at layer 12.
  Layer 6 was evaluated on the smaller battery only. Layer 12 was chosen because
  a fitted operator exists there and it is the median layer at which J-lens first
  reaches top-10; it is not the peak (layer 25 has the most top-10 hits, 10/33).
- **Diagonal only.** Cross-coordinate curvature was never estimated. The
  curvature-energy diagnostic put coordinate-diagonal terms at <1% of Hessian
  Frobenius norm, so the negative result is consistent with curvature mattering
  while being predominantly off-diagonal — which is an explicitly listed
  alternative in `PROJECT_IDEA.md`, not a refutation of it.
- **128 Hessian pairs.** The averaged Hessian uses a seeded subsample of the
  2,947 available (moments use all of them), because it costs one
  forward-over-forward pass per (coordinate, pair). Using all of them would cost
  ~557 GPU-hours for two layers, exceeding the allocation. The 8→128 comparison
  gives a 16x stability check and the conclusion is unchanged, but entrywise
  convergence was not established.

## The measuring instruments

- **The judge is a model.** Judge and mechanical top-k membership agree on 95.3%
  of items. The 4.7% disagreement is mostly intended (the judge counts ` trail`
  for *path*), but it is a model-shaped instrument in the measurement path.
- **The same model generates and judges.** GLM-5.2 writes the items *and*
  adjudicates the readouts. The exposure is limited — the judge compares two
  Qwen-derived lists rather than anything GLM wrote — but a blind spot shared
  between the two roles would not surface.
- **Self-report is a proxy for "the model holds the concept".** It establishes
  the concept is accessible to the model's own output behaviour at that prefix.
  It does not establish the concept is linearly represented in the residual at
  the probe, which is what a lens would need to read.

## Statistical

- Sign tests on paired ranks ignore magnitude by construction. A method could win
  fewer comparisons while rescuing the failures that matter most. The R-lens
  finding (worse on average, 4.9% vs 0.1% top-10 on the failure cell) is exactly
  such a case and would be invisible in the win/loss column alone.
- Cells are not independent samples of a single population; they are defined by
  the measurement. Comparisons *within* a cell are sound; pooling across cells
  weights by however many items happened to land in each.
