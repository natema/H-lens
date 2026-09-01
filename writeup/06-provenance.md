# Provenance

## Artifacts

| what | where |
|---|---|
| concept list, 3,344 | `configs/concepts.json` |
| hand-built 33-case battery | `configs/battery_cases.json` |
| J-space dataset, bfloat16 draft | `data/` |
| J-space dataset, float32 | `data_fp32/` |
| item quality grades | `data/quality.jsonl` |
| all-lens evaluation | `data_fp32/hlens.jsonl` |
| results narrative | `data_fp32/RESULTS.md` |
| fitted layer-12 operator | `results/hessian_pile_l12_merged_qwen3.5-4b.pt` |
| API cost ledger | `pilot/spend.json` |

Model `Qwen/Qwen3.5-4B` and both lens artifacts are pinned by full revision and
verified by SHA-256 at load.

## The fit

The correction is estimated on **pile-10k**, the corpus the published J-lens was
fitted on (25 documents, `t_max` 128, `skip_first` 4), so first- and second-order
operators come from the same distribution. **No evaluation item informs the fit.**
Moments use all 2,947 (document, position) pairs; the averaged Hessian uses a
seeded subsample of 128.

Computed on Jean Zay `gpu_p6` (H100), billed to `myv@h100`: four coordinate
shards per layer, layer 6 at 3h56m per shard and layer 12 at 3h09m, **~28
GPU-hours**. Coordinate sharding was verified to be exact — two shards merge to a
result bit-identical to the equivalent unsharded fit.

## Cost

**API: $9.08 over 2,423 calls** (GLM-5.2 for generation, adjudication, concept
filtering, quality grading; mistral-large for an earlier judge). Rates from the
documented per-model pages, with cached input billed separately. The Mistral API
exposes no billing endpoint, so cost is derived from the exact token counts each
call returns; `console.mistral.ai` is authoritative.

## Reproducibility caveats

Recorded in `manifest.json` per dataset:

- Self-report generation is batched, so an item's output depends on which items
  share its batch. Exact reproduction requires the same `read_batch` *and* the
  same concept ordering in `items.jsonl`.
- Model dtype is recorded, and it matters at the top-k boundary.
- All six model-facing prompts are module constants (`GENERATOR_SYSTEM`,
  `JUDGE_SYSTEM`, `SELF_REPORT_TEMPLATE`, `FILTER_SYSTEM`,
  `EXPLICIT_SCREEN_SYSTEM`, `SCORER_SYSTEM`), so no filtering decision lives only
  in shell history.
- 20 words removed from the concept list as sexually explicit are listed in
  `configs/concepts.json` under `removed_adult_content`; four over-flagged
  clinical/historical terms are retained via `SCREEN_KEEP`.
