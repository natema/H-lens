# Stage-by-stage notes

Working notes written as the project progressed, kept because they record how
each stage was checked and what each intermediate result looked like. The
[README](../README.md) and the [write-up](../writeup/WRITEUP.md) describe the
final state; where a number here differs from them (the eight-sample fit, the
33-case battery), the later documents supersede these notes. The SLURM script
used on the cluster is site-specific and not included; the fit is the
`j2-evaluate` call shown below, run once per coordinate shard and merged with
`j2-merge`.

## Baseline reproduction

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

### Layer 12, where the battery actually carries signal

Layer 6 is early enough that no lens recovers much, so it discriminates poorly.
Refitting at layer 12 — the median layer at which J-lens first reaches the top
10 — gives a far healthier comparison (same 31 held-out cases, same eight-sample
development set):

| method | geometric-mean rank | median | pass@10 | pass@100 |
|---|---:|---:|---:|---:|
| logit lens | 14831.0 | 34004 | 0.0% | 6.5% |
| J-lens | 210.9 | 161 | 22.6% | 45.2% |
| J² (real diagonal) | 223.7 | 147 | 16.1% | 48.4% |
| J² (shuffled control) | 212.6 | 135 | 16.1% | 45.2% |
| R-lens | 241.8 | 231 | 16.1% | 41.9% |

Two independent checks that the measurement is sound. J-lens improves sharply
from layer 6 to layer 12 (geometric-mean rank 1506.8 to 210.9), as the
first-hit-layer analysis predicts. And R-lens beats J-lens at layer 6 but not at
layer 12, which is exactly the early-layer claim the R-lens post makes; that
qualitative result is reproduced here independently.

J² wins 14 and loses 14 (sign test p = 1.00) and the shuffled control also wins
14 and loses 14, with a better geometric mean. The sharpest number is on the
development set rather than the held-out set: the normalized residual error rises
from 0.914183 for J-lens to 0.914238 for J². With the Taylor coefficient fixed at
1/2 the correction has no freedom to overfit, so this is not overfitting — the
second-order term is harmful at these displacement magnitudes, which is the
"relevant displacement is too large" alternative in `PROJECT_IDEA.md`.

## Fitting on pretraining text instead of the evaluation prompts

Both results above share a serious limitation. The development set was two
9-token prompts; after `skip_first=4` and the final-position exclusion that is
**eight** (prompt, position) samples, from which mu, the per-coordinate variance,
and the whole 2560x2560 averaged Hessian were estimated. Since the correction is
`0.5 * D @ (delta^2 - Var[delta])`, a variance estimated from 8 samples in 2560
dimensions is close to noise and is subtracted from every prediction. "The
diagonal correction does not help" could not be separated from "the correction
was fitted on eight points".

Two changes address this.

**A per-row reduction endpoint.** The batched reduction summed each row up to
`input_ids.shape[1] - 1`, one batch-wide endpoint, so a padded short prompt would
have summed straight through its padding and silently produced the wrong
operator. Guards therefore required every development prompt to share a length.
The endpoint is now a per-row `ends` tensor (`pad_pair_batch`,
`reduce_target_sums`). Right padding is exact, not approximate: attention is
causal, so a real position never attends to a token that follows it, and each row
stops at its own penultimate token. Verified on Qwen3.5-4B at relative error
3.9e-6, and in tests against a toy model with a causal mixing layer plus a test
that garbage in the padding region cannot change the result.

**A pretraining development corpus.** `configs/evaluation_split_pile.json` takes
development data from `NeelNanda/pile-10k` — the corpus the published J-lens was
fitted on, at its recorded 25 documents, `t_max` 128, `skip_first` 4 — so the
first- and second-order operators are estimated on the same distribution. Every
one of the 33 battery cases is then held out, including `typo_aganst` and
`multihop_sushi`, and no evaluation prompt informs the fit.

The two estimation costs differ by orders of magnitude and are decoupled. The
activation moments are forward passes only and use every pair, 2947 instead of 8.
The averaged Hessian costs one forward-over-forward pass per (coordinate, pair),
so it is capped by `--hessian-pairs` with a seeded subsample.

```bash
uv run j2-evaluate --offline --estimator forward --layer 12 \
  --cases configs/battery_cases.json \
  --split configs/evaluation_split_pile.json \
  --hessian-pairs 32 --coordinate-batch-size 4 \
  --artifact results/hessian_pile_l12_qwen3.5-4b.pt \
  --output results/evaluation_pile_l12_qwen3.5-4b.json
```

Work scales as `n_coordinates x n_hessian_pairs x t_max`, so the full fit runs on
an H100 (`j2_fit.slurm`, Jean Zay `gpu_p6`) rather than locally. Batch rows are
`coordinate_batch_size x hessian_pairs` full-length sequences and
forward-over-forward roughly triples activation memory, so the coordinate batch
must stay small.

## Result with the pretraining-fitted correction

Fitted on Jean Zay (`gpu_p6`, H100): four coordinate shards per layer,
2,947 moment pairs and 128 Hessian pairs, about 28 GPU-hours total. All 33
battery cases held out.

**Layer 6**

| method | geo-mean rank | median | pass@10 | pass@100 |
|---|---:|---:|---:|---:|
| logit lens | 3730.6 | 9733 | 6.1% | 21.2% |
| J-lens | 1249.3 | 967 | 9.1% | 24.2% |
| J² (real diagonal) | 842.6 | 396 | 12.1% | 30.3% |
| J² (shuffled control) | 908.9 | 573 | 12.1% | 30.3% |
| R-lens | 629.9 | 480 | 15.2% | 33.3% |

**Layer 12**

| method | geo-mean rank | median | pass@10 | pass@100 |
|---|---:|---:|---:|---:|
| logit lens | 12871.2 | 26626 | 0.0% | 6.1% |
| J-lens | 177.6 | 154 | 24.2% | 48.5% |
| J² (real diagonal) | 159.9 | 84 | 21.2% | 51.5% |
| J² (shuffled control) | 151.6 | 103 | 24.2% | 48.5% |
| R-lens | 202.2 | 191 | 18.2% | 45.5% |

**The sample size did matter.** On the cases common to both runs, the
eight-sample fit gave J² a geometric-mean rank of 1775.3 at layer 6 and 223.7 at
layer 12; the properly fitted correction gives 1050.2 and 190.1. J-lens and
R-lens are unchanged to the digit, as they must be, which also confirms the
pipeline is deterministic. The earlier "J² is worse than J-lens" reading was an
artifact of estimating mu, the variance, and the Hessian from eight vectors.

**But the correction still shows no curvature signal.** Against J-lens, J² wins
21 and loses 12 at layer 6 (sign test p = 0.16) and wins 16 and loses 11 at
layer 12 (p = 0.44) — neither significant. The coordinate-shuffled control wins
*more*, and significantly: 25-7 (p = 0.0021) at layer 6 and 23-5 (p = 0.0009) at
layer 12. Head to head, the real diagonal does not beat its own shuffled version
at either layer (layer 6: 16-14 for real, p = 0.86; layer 12: 8-19 against real,
p = 0.052).

So a correction of this form and magnitude does improve the rank readout, but
scrambling which curvature vector belongs to which coordinate does not remove
the improvement — and at layer 12 slightly increases it. The gain is not
attributable to coordinate-specific curvature. This is the "apparent gains come
only from an intercept, rescaling, or extra capacity" alternative in
`PROJECT_IDEA.md`, not the primary hypothesis.

The development-set number points the same way. On the very data the operator
was estimated from, the normalized residual error rises rather than falls, at
both layers and for both the real and the shuffled correction:

| | J-lens | J² real | J² shuffled |
|---|---:|---:|---:|
| layer 6 | 0.972865 | 0.973037 | 0.973053 |
| layer 12 | 0.935327 | 0.935511 | 0.935559 |

With the Taylor coefficient fixed at 1/2 there is no freedom to overfit, so the
second-order term is genuinely harmful to the approximation it is supposed to
improve, while helping a rank readout for reasons unrelated to curvature.

### Limitation of the control

`j2_shuffled` is a single deterministic `torch.roll` of the diagonal rows by one
coordinate. It preserves the row-norm distribution and destroys the coordinate
correspondence, which is the right null, but it is one draw rather than a
reference distribution. The comparisons above should be repeated over many
random permutations before the sharpest claim — that shuffling is no worse than
the real diagonal — is treated as quantitative rather than directional. This
needs no new model passes, only repeated readouts.
