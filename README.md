# J²-lens baseline reproduction

This directory compares logit lens, J-lens, and R-lens on published qualitative
failure cases using `Qwen/Qwen3.5-4B`.

The model and matched J/R artifacts are pinned by full Hugging Face revisions.
The evaluator checks the artifacts' published SHA-256 hashes, validates their
matched metadata, records exact tokenization and probe positions, and saves
per-layer target ranks plus top-token readouts.

```bash
uv sync
uv run j2-baselines
```

After the initial artifact download, `uv run j2-baselines --offline` requires a
fully cached run. Results go to `results/baselines_qwen3.5-4b.json` by default.

The exact source screenshots used DeepSeek-V4-Flash, whereas this replication
uses the published Qwen3.5-4B lens pair. We therefore test whether the reported
qualitative behavior transfers, not whether numerical ranks match across
different models.

## Historical randomized curvature diagnostic

`j2-curvature` estimates the fraction of downstream Hessian Frobenius energy
in coordinate-diagonal terms without materializing the Hessian. It also
compares the full and diagonal quadratic forms along the probe activation's
displacement from the prompt's mean activation.

This diagnostic uses randomized projections and is retained only as an earlier
exploration. It is not used to construct or evaluate the forward-mode J² lens
below.

```bash
uv run j2-curvature --offline
```

The total-energy estimator uses independent Rademacher input directions and a
normalized Rademacher output projection. Coordinate terms are importance
sampled from a 50/50 mixture of uniform and squared-displacement probability;
inverse-probability weights keep both estimates unbiased. Natural-direction
diagonal energy uses independent sample splits to avoid the upward bias from
squaring a noisy estimate.

The default command evaluates source layers 6, 12, and 20 against target layer
30 for the typo, sushi, and Jordan prompts. Probe counts are intentionally
small and the raw results show substantial Monte Carlo variation, so they
should be treated as a diagnostic rather than a precise layerwise estimate.

## Held-out J² evaluation

The evaluation split is frozen in `configs/evaluation_split.json`: typo and
sushi estimate the activation moments and average Hessian; Jordan, Verona, and
Einstein are held out. The primary estimator computes the full vector
`H[e_j,e_j]` for every one of the 2,560 residual coordinates using nested
forward-mode JVPs. It uses no output projection, random directions, importance
sampling, finite-difference step, or fitted Hessian multiplier.

The correction is exactly

```text
J @ x + 0.5 * D @ ((x - mean_x)^2 - variance_x).
```

Run the decisive layer-6 evaluation with:

```bash
uv run j2-evaluate --offline \
  --estimator forward --layer 6 --coordinate-batch-size 32 \
  --artifact results/hessian_lens_forward_l6_qwen3.5-4b.pt \
  --output results/evaluation_forward_l6_qwen3.5-4b.json
```

The complete forward-mode diagonal has shape 2560 by 2560 and contains only
finite values. As an implementation check, it agrees with the complete
finite-difference diagonal at cosine 0.99987; finite differences do not enter
the reported predictions.

On the three held-out probes, J² changes target ranks from 58 to 138 (Jordan),
267 to 188 (Verona), and 388 to 1116 (Einstein): one win and two losses. Mean
rank worsens from 237.7 to 480.7; KL divergence, residual cosine, and normalized
residual error also worsen. A coordinate-shuffled correction has a better mean
rank (244.0) than the real diagonal. This is evidence against this
development-averaged diagonal J² correction at layer 6, not evidence that the
downstream map lacks curvature or that every possible second-order lens must
fail.

Tensor artifacts are ignored because the complete layer-6 operator is 26 MB
and is deterministically regenerable. Compact JSON results and full provenance
are retained in `results/`.

## Enlarged evaluation battery

The original five cases are the J-lens/R-lens posts' *qualitative demonstrations*,
not their evaluation set. The R-lens post evaluates five categories — multihop,
multilingual, association, typo, and poetry — and reports mean pass@10 over
layers, filtering multihop and multilingual to questions the model answers
correctly.

`configs/candidate_cases.json` holds 35 candidates in those five categories: the
R-lens representative prompts, the five original cases, and new items written in
the same style. `j2-screen` applies a mechanical, lens-blind screen — single-token
target, uniquely resolvable probe span, target string absent from the prompt, and
the R-lens answerability filter for multihop and multilingual — and writes the
accepted cases to `configs/battery_cases.json`.

```bash
uv run j2-screen --offline --max-new-tokens 32
```

33 of 35 candidates pass. The screen uses no lens output, so it cannot select
cases on J-, R-, or H-lens performance. The two rejects are multilingual items
Qwen3.5-4B does not answer as intended (`vecchio`→`nuovo`, a valid antonym but
not the "young" intermediate; the `ciel` prompt is read as a statement).

The development set stays frozen at the two cases behind the existing layer-6
forward-mode fit, so the operator is reused verbatim and all 31 new and old cases
enter the held-out side only:

```bash
uv run j2-evaluate --offline --estimator forward --layer 6 \
  --cases configs/battery_cases.json \
  --split configs/evaluation_split_battery.json \
  --reuse-artifact results/hessian_lens_forward_l6_qwen3.5-4b.pt \
  --output results/evaluation_battery_l6_qwen3.5-4b.json
```

`--reuse-artifact` reloads a previously fitted operator instead of re-paying the
30-minute fit. It refuses to reuse unless the stored development split,
`skip_first`, target layer, estimator, and model revision all match, and unless
the freshly recomputed development moments agree with the stored ones.

### Result on 31 held-out cases at layer 6

| method | geometric-mean rank | median | pass@10 |
|---|---:|---:|---:|
| logit lens | 4831.5 | 13275 | 6.5% |
| J-lens | 1506.8 | 1100 | 9.7% |
| J² (real diagonal) | 1775.3 | 1711 | 9.7% |
| J² (shuffled control) | 1562.4 | 1279 | 6.5% |
| R-lens | 844.4 | 704 | 12.9% |

Paired against J-lens, the real diagonal wins 16 and loses 15 (sign test
p = 1.00), while the shuffled control wins 18 and loses 13. The n=3 result was
"one win, two losses"; at n=31 the correction is indistinguishable from noise and
still no better than a coordinate-shuffled version of itself.

### Caveats this battery exposes

`j2-baselines --cases configs/battery_cases.json` sweeps every layer for the
three reference lenses (`results/baselines_battery_qwen3.5-4b.json`). Two
limitations follow from it.

First, **layer 6 is not where this battery carries signal.** The median layer at
which J-lens first places the target in the top 10 is 12, and 15 of 33 cases
never reach the top 10 at any layer. Fitting the correction only at layer 6 tests
it where no lens works.

Second, **some published examples do not transfer to Qwen3.5-4B**, as Einstein
already did not. For `association_fame` the model's own final-layer readout ranks
` fame` at 27379 at the probe position, so no lens can recover a concept the
model does not represent. The four poetry items behave the same way. A further
lens-blind screen — requiring the model itself to name the concept when asked
directly — would remove these, and should be applied before the battery is used
for a quantitative claim.
