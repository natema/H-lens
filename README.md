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

## Curvature diagnostic

`j2-curvature` estimates the fraction of downstream Hessian Frobenius energy
in coordinate-diagonal terms without materializing the Hessian. It also
compares the full and diagonal quadratic forms along the probe activation's
displacement from the prompt's mean activation.

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
