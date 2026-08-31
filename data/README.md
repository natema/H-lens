# J-space dataset

3,302 items over concepts drawn from Qwen3.5-4B's own vocabulary. Each item is a
sentence fragment ending on a probe term, plus two independent readings of what
the model holds at that probe.

Built with `uv run j2-dataset`. See `manifest.json` for the exact settings.

## Files

| file | contents |
|---|---|
| `items.jsonl` | generated fragments: concept, probe term, fragment, rationale |
| `readouts.jsonl` | lens ranks at layer 12, the top-10 readout, and the model's self-report |
| `dataset.jsonl` | the above plus the adjudication and the outcome cell |
| `manifest.json` | layer, top-k, batch sizes, cell counts, reproducibility note |

## The four cells

| cell | n | share | meaning |
|---|---:|---:|---|
| `self_report_only` | 1168 | 35.4% | **J-lens misses it, the model has it** |
| `neither` | 1146 | 34.7% | the model does not hold the concept at the probe |
| `both` | 625 | 18.9% | positive control |
| `lens_only` | 363 | 11.0% | lens finds it, self-report does not |

Nothing is filtered on agreement. Keeping only `both` would retain exactly the
items the J-lens already handles, leaving no headroom for a correction to show
any effect — selection on the outcome variable. `self_report_only` is the useful
cell precisely because the self-report independently establishes the concept is
present, so a J-lens miss there is a real failure rather than an absent target.
`neither` is kept for the same reason it matters: without it there is no way to
distinguish "the lens failed" from "there was nothing to find".

## Layer-12 ranks on the 1,168-item target set

| lens | geometric mean | median | top-10 | top-100 |
|---|---:|---:|---:|---:|
| J-lens | 250 | 163 | 1.0% | 42.0% |
| R-lens | 320 | 248 | 5.2% | 34.5% |
| logit lens | 9,538 | 13,839 | 0.3% | 3.0% |

The logit lens being two orders of magnitude worse is the sanity check that the
readout means something. R-lens reaches the top 10 five times more often than
J-lens here while having a worse median, so the two disagree about which
failures are recoverable — which is the comparison the correction has to beat.

## Known measurement noise

- **Judge versus mechanical rank: 5.0% disagreement.** The judge is asked
  whether the top-10 *names the concept*, which is not the same as whether the
  concept's own token ranks in the top 10. It calls ` trail`/` trails` a match
  for *path* and ` findings`/` conclusions` a match for *report*, while the
  exact tokens rank 13 and 16. That is the judge working as intended, but it
  means the cell label and the rank column measure slightly different things.
- **2 fabricated matches** were caught and discarded by the verbatim-quote
  check.
- **Batched self-report** introduces roughly 1.8% variation in whether a concept
  appears, measured against sequential generation. Reproducing the file exactly
  requires the same `read_batch` and the same concept ordering.
- **42 concepts** produced no conforming fragment in three generation passes and
  are absent.
