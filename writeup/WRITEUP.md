# Does a diagonal Hessian correction make the J-lens more faithful?

*Draft for the MATS 12.0 application (Neel Nanda's stream). Executive summary
first, main write-up after. Every number is traceable to a file under
`j^2-lens/data/`, `results/` or `configs/`.*

---

## Executive summary

**Question.** The Jacobian lens (J-lens) reads intermediate concepts out of a
transformer's residual stream by propagating the residual at a token through the
*average Jacobian* of the downstream computation, then unembedding. It fails on
some early-layer cases where the R-lens does not. I asked whether the simplest
scalable second-order fix — adding a development-averaged **diagonal Hessian**
term, `½·D·(δ² − Var[δ])`, analogous to Optimal Brain Damage — recovers any of
those failures. Model: Qwen3.5-4B, using the published J-lens and R-lens
artifacts.

**Answer: no.** On 3,344 held-out items with an operator fitted on unrelated
pretraining text, the correction is statistically indistinguishable from *no
correction*, and indistinguishable from a version of itself with the coordinate
structure destroyed. At layer 12 it moves **exactly zero** items out of the
J-lens-failure cell on all items, and it is slightly worse than J-lens at layer 6.
Its coordinate-shuffled control is at least as good as the real operator
everywhere.

**Three things I think are worth a reader's attention beyond the null.**

1. *A selection trap that would have produced a false positive.* Evaluated only
   on the cell where the J-lens fails — the natural thing to do — the
   correction beats J-lens 601 to 510 (sign test p = 0.007). On the same cell
   the shuffled control wins 630 to 463 (p = 5×10⁻⁷). That cell is *defined* by
   J-lens failing, so regression to the mean flatters anything. Scoring every
   cell, not just the failures, is what separates the two readings.
2. *The dataset validates itself against a known effect.* To evaluate the
   correction I had to build an evaluation set, since three published examples
   cannot support a claim. On that set the R-lens **reverses across layers**
   exactly as its authors claim: it removes 161 J-lens failures at layer 6 and
   adds 91 at layer 12. The same instrument that shows R-lens moving 161 items
   shows the Hessian correction moving 2. The null is not blunt measurement.
3. *Most of the project was measurement, and most of the measurement bugs
   changed a conclusion.* The first fit used **eight** development samples
   because of a batching bug; the original rank definition was optimistic under
   bfloat16 ties; the generator put the evocative content after the probe on
   64% of items; the self-report prompt suppressed the concepts it was meant to
   detect. Each was found by a check that did not depend on the result, and each
   is recorded with how it was caught.

**What the evaluation set is.** 3,344 items, each a sentence fragment ending on
a probe word with a target concept drawn from Qwen's own vocabulary. Two
independent readings per item: does the lens's top-10 at the probe *name* the
concept, and does the model itself name it when shown the same prefix and asked.
Every item is kept under one of four cells; nothing is filtered on agreement.
1,289 items are `self_report_only` — the model has the concept, the J-lens
misses it — which is the set a better lens has to win on. A lens-blind quality
grade marks 826 items `strong`.

**Why this matters for the stream.** Neel's list asks directly: "what is J-Lens
actually doing? How much better is it really than logit lens and why?" and
warns "remember to compare to baselines." This project is a negative answer to
one proposed improvement, a positive validation of a competing one, and a
reusable instrument for asking the question about the next one. It also has a
methodological result — the selection trap — that generalises to any evaluation
of a lens on its predecessor's failures.

**Cost.** ~15h of counted time (see `PROJECT_LOG.md`), ~28 H100 GPU-hours on
Jean Zay for the operator fit, ~$61 of API calls (GLM-5.2 via Mistral) for
generation, adjudication and grading.

**Biggest limitations** (§7): one model, one primary layer, diagonal only (the
diagonal carries <1% of Hessian Frobenius energy, so this is consistent with
curvature mattering off-diagonal); the judge is a model; and a lens's top-10
holds only ~7.8 distinct concepts against the self-report's 10. I corrected that
budget asymmetry on the strong items by collapsing each lens's top-50 to its
first 10 distinct concepts: every lens gains substantially and no conclusion
changes (§4), though the collapser itself has residual limitations (§7).

---

## 1. The question and why I chose it

The J-lens (Anthropic, 2026) linearises the map `F_ℓ` from the residual at layer
ℓ to the final residual by its average Jacobian `J_ℓ = E[∇F_ℓ(x)]`, and reads
concepts from `J_ℓ·x` through the unembedding. The R-lens post shows early-layer
failures — sushi→Japan, Verona→Italy — and fixes them with LRP-style relevance
propagation. My hypothesis was different: that some failures are *curvature*
that a linearisation cannot see, and that the cheapest second-order term would
recover some of them.

The candidate lens:

    F̂(x) = J·x + ½·D·(δ ⊙ δ − E[δ ⊙ δ]),   δ = x − μ,   D_{:,j} = E[∂²F/∂x_j²]

`D` is `d_model × d_model`, the same size as `J`. Everything is fixed on
development data: no scalar calibration, no fitted intercept, Taylor coefficient
½. The pre-registered alternatives (from `PROJECT_IDEA.md`) were: curvature is
mostly off-diagonal; curvature is prompt-local and averages away; the real
problem is attribution through RMSNorm and gated MLPs, which R-lens addresses;
apparent gains come from rescaling or an intercept; the displacement is too large
for a Taylor expansion.

I chose it because it is a clean, falsifiable hypothesis about a method Neel's
list singles out, it has an obvious control (destroy the coordinate structure),
and it forces the question "how would I know if a lens got better?" — which
turned out to be the harder problem.

## 2. Technical setup

**Model and artifacts.** `Qwen/Qwen3.5-4B`, pinned revision, float32 for all
lens readouts (bfloat16 for generation). J-lens and R-lens matrices from
`camilablank/workspace-lenses`, pinned revision, SHA-256 verified at load.

**Estimating D.** Forward-over-forward autodiff: for each residual coordinate
`e_j`, `H[e_j,e_j]` is the JVP of the JVP of the downstream map, exactly, with
no random projection, no importance sampling and no finite differences. (An
earlier estimator used all three; I removed them after the first negative result
looked like it could be estimator noise. The forward-mode diagonal agrees with the
finite-difference one at cosine 0.99987, so they were not the cause.) The map's
output is the sum of target-layer residuals from the probe through the
penultimate token, matching the J-lens fitting reduction.

**Development data.** `NeelNanda/pile-10k`, the corpus the published J-lens was
fitted on — 25 documents, `t_max` 128, `skip_first` 4 — so first- and
second-order operators come from the same distribution and **no evaluation item
informs the fit**. Moments (μ, Var) use all 2,947 (document, position) pairs.
The averaged Hessian is expensive — one forward-over-forward pass per
(coordinate, pair) — and uses a seeded subsample of 128. Fitted at layers 6 and
12 on Jean Zay H100s, four coordinate shards per layer (verified bit-identical to
an unsharded fit), ~28 GPU-hours.

**Baselines.** Logit lens, J-lens, R-lens, and the **coordinate-shuffled
control**: the same operator with its rows rolled by one coordinate. This
preserves the row-norm distribution and destroys the coordinate-to-curvature
correspondence, so it is what the correction must beat for a gain to be
attributable to curvature.

**Metric.** The four cells below, defined per item and per lens, plus rank of
the concept token (geometric mean, since ranks are heavy-tailed: on one set the
arithmetic mean was 27,758 against a geometric mean of 1,249).

## 3. The evaluation set

**Why build one.** The project began with the five qualitative examples from the
posts, split 2 development / 3 held-out. The first result was "one win, two
losses". Nothing can be concluded from three items, and the posts' *evaluation*
sets are larger than their illustrations. A 33-case hand-built battery across the
R-lens categories (multihop, multilingual, association, typo, poetry) was better
and still too small.

**Concepts from the vocabulary.** A concept must be a single token for a lens to
emit it. 248,077 tokens → 40,727 matching `^ [a-z]{4,}$` (the word-initial form
a readout emits) → 14,990 at pile-10k frequency ≥ 3 with inflections dropped →
**3,344 concrete nouns** after an LLM filter and a screen for explicit content
(20 removed, listed; four over-flagged medical terms restored). Frequency is
measured over the whole corpus because shape alone does not identify English in
a multilingual vocabulary.

**Items.** GLM-5.2 writes a fragment for each concept that **ends on the probe
word**. Attention is causal, so the residual at the probe depends only on the
text up to and including it; anything after is discarded. Verified: prefix vs
full-sentence readouts agree to 3.9×10⁻⁶ relative in float32.

**Two readings per item.**
- **Lens:** does the top-10 readout at the probe *name* the concept? Decided by
  GLM-5.2 as judge, so a variant counts (` Japon` for *japan*, ` matrim` for
  *wedding*) but association does not (` sushi` for *japan*, ` pawn` for
  *chess*). Fragment ambiguity is resolved from the tokenizer, not recalled:
  `matrim` is extended only by the matrimony family, `Vol` by fourteen unrelated
  words. The judge must quote its match verbatim from the list; fabricated
  matches are discarded (2 caught).
- **Self-report:** shown *exactly the same prefix*, asked which concepts it is
  thinking about at the final word, with two worked examples. Both demo concepts
  and all 20 demo answer words are absent from the concept set.

**Four cells, nothing filtered on agreement** (current counts; strong-only in
parentheses):

| cell | n | meaning |
|---|---:|---|
| `self_report_only` | 1289 (399) | model has it, J-lens misses it — **the target** |
| `both` | 725 (167) | positive control |
| `lens_only` | 277 (58) | lens finds it, self-report does not |
| `neither` | 1053 (202) | model does not hold the concept at the probe |

Keeping only agreement would retain exactly the items the J-lens already
handles, leaving no headroom — selection on the outcome. `neither` is what
separates "the lens failed" from "there was nothing to find": for the published
`association_fame` example Qwen's own final readout ranks ` fame` at 27,379, so
no lens could recover it.

**Quality grade.** A lens-blind grader (sees fragment, probe, concept; never a
readout) marks each item strong/medium/weak: 826 / 1590 / 928. It catches what a
regex cannot — `home` with probe `house`, `design` with probe `blueprint` — and
it predicts lens behaviour it never saw: weak items have J-lens geometric-mean
rank 46 against 102 for strong, because a weak item's concept sits as a synonym
of a word already present. Medium items are hardest (162), consistent with
"needs a step of reasoning".

## 4. Results

Failures = `self_report_only` count; lower is better. Deltas are relative to
J-lens on the same items.

| | J² real | J² shuffled | R-lens |
|---|---:|---:|---:|
| layer 12, all (n=3344) | **+0** | −18 | +91 |
| layer 12, strong (n=826) | −6 | −12 | +4 |
| layer 6, all | −3 | −2 | **−161** |
| layer 6, strong | +3 | +2 | **−48** |

Layer 6, strong, in full:

| method | self_report_only | both | lens_only | neither |
|---|---:|---:|---:|---:|
| J-lens | 547 | 19 | 3 | 257 |
| J² real | 550 (+3) | 16 (−3) | 2 (−1) | 258 (+1) |
| J² shuffled | 549 (+2) | 17 (−2) | 2 (−1) | 258 (+1) |
| R-lens | **499 (−48)** | **67 (+48)** | 14 (+11) | 246 (−11) |

**The correction does nothing**, at either layer, on all items or strong ones,
and its shuffled control is never worse than it. Paired ranks tell the same
story: across all items J² vs J-lens is 1375 wins / 1317 losses (p = 0.27) while
the shuffle is 1381 / 1242 (p = 0.007).

**R-lens reverses across layers**, removing 161 failures at layer 6 and adding
91 at layer 12, and more than doubling the `both` cell at layer 6 (142 → 303).
That is the R-lens post's central claim, reproduced on 3,344 items built without
reference to it. On the same instrument R-lens moves 161 items where J² moves 2.

**Equalising the concept budget.** A lens's raw top-10 holds ~7.8 distinct
concepts (` sushi`, ` Sushi`, `寿司` count three times) where the self-report
holds 10, and the lenses differ (R-lens 8.2). On the 826 strong items I
collapsed each lens's top-50 to its first 10 distinct concepts (GLM-5.2 merges
spelling-level variants; a mechanical pass splits back out any different word
it over-merged, e.g. *sneakers* from *shoe*) and re-judged. Failures fall for
every lens — J-lens 399 → 307 at layer 12, 547 → 502 at layer 6 — so the raw
tables understate all of them. The orderings do not move:

| collapsed, failures vs J-lens | J² real | J² shuffled | R-lens |
|---|---:|---:|---:|
| layer 12 (J-lens 307) | +8 | +3 | **+38** |
| layer 6 (J-lens 502) | +4 | +5 | **−49** |

R-lens's layer-6 advantage holds at −49 (raw −48) and its layer-12 deficit
*widens* to +38 (raw +4): it had less redundancy to begin with, so J-lens gained
more from the fair budget. J² is +4/+8 and never beats its shuffle.

**Robustness to the instrument.** Regenerating the self-reports with a
materially different prompt moved the naming rate from 54.1% to 60.2% and every
cell count — and **no lens ranking**. J² stayed at +0, R-lens went from −141 to
−161. A six-point shift in the measuring instrument that leaves every ordering
intact is the best evidence available that these numbers are about the lenses.

## 5. Strongest evidence against the hypothesis

- **The shuffled control.** Destroying which curvature vector belongs to which
  coordinate does not remove the (tiny) effect and sometimes improves it. Whatever
  the correction does, it is not using curvature.
- **The selection-trap demonstration** (§summary, point 1). The one place the
  correction looks significant is the place where any perturbation would.
- **The development set moves the wrong way.** Normalized residual error on the
  data the operator was fitted from rises: layer 6 0.972865 → 0.973037, layer 12
  0.935327 → 0.935511. With coefficient ½ there is no freedom to overfit. I
  overstated this early on — the differences are ~0.02% of the target norm on a
  metric that barely beats predicting zero — so it is corroboration, not the case.
- **Sample size did not rescue it.** Going from 8 to 128 Hessian pairs and 8 to
  2,947 moment pairs improved J²'s own ranks substantially (1775 → 1050 at
  layer 6) and changed the conclusion not at all.

## 6. Measurement pitfalls that changed a conclusion

These belong in the write-up rather than a footnote because a reader evaluating
a *different* correction would hit the same ones.

1. **Selection on the outcome, twice.** First as "keep only items where lens and
   self-report agree"; then as "evaluate only where J-lens fails". Both caught by
   asking what the selection conditions on.
2. **An eight-sample development set.** A batch-wide reduction endpoint
   (`input_ids.shape[1] − 1`) forced equal-length prompts, which forced the
   development set to two 9-token prompts × 4 positions. Fixed with per-row
   ends; right-padding is then exact because attention is causal.
3. **Optimistic rank under ties.** `rank = count(logits > target) + 1` ignores
   ties, and bfloat16 ties readily: ` Election` and ` electronically` both round
   to 10.875000. 38% of boundary items have a 10th-vs-11th gap below bfloat16
   resolution. But the median gap is a probability ratio of 1.064, so a hard
   top-10 cutoff is arbitrary *at any precision*; the honest fix was to record
   the margin to the k-th token, not only to use float32. Float32 moved 8
   labels; judge run-to-run variability moved 25.
4. **Instructions are not constraints.** Told "never rely on what follows the
   probe", the generator put the evocative words after it on 64% of items
   (`"The score"` survived; `"was tied at halftime"` was discarded). Fixed
   structurally: reject at write time, retry.
5. **The self-report prompt suppressed its own signal.** "Do not list synonyms"
   held the self-report flat at 53.5% on the 803 items whose probe restates the
   concept, while the lens rate there doubled to 48%. Removing it and adding two
   worked examples raised published-target recovery on the 33-case battery from
   16/33 to 19/33; a minimal prompt got 13/33.
6. **The judge was asymmetrically strict.** It rejected `aluminum` for
   *aluminium* and `sheets` for *sheet* on the self-report side while accepting
   variants on the lens side — 92 near-variant rejections against 25. Fixed by
   naming those cases and requiring equal tolerance on both lists; the seven
   recorded fixtures still behave.
7. **A free-form list is not reproducible at temperature 0.** The concept
   filter agrees with itself at Jaccard 0.74 across runs; the judge's structured
   per-item verdict was stable. `configs/concepts.json` is therefore the artifact
   of record.

## 7. Limitations

- **One model, one primary layer, one layer pair for the correction.** Nothing
  here speaks to larger models or to DeepSeek, where the original figures came
  from (the Einstein example did not transfer to Qwen at all).
- **Diagonal only, 128 pairs.** The randomized energy diagnostic put
  coordinate-diagonal terms at <1% of Hessian Frobenius norm. The null is fully
  consistent with curvature mattering *off-diagonal* — a pre-registered
  alternative, not a refutation of it. Using all 2,947 pairs for D would have
  cost ~557 GPU-hours, beyond the allocation.
- **The concept-budget correction is checked only on strong items, and its
  collapser is imperfect.** The collapse (§4) was run on the 826 strong items,
  not all 3,344. Its depth of 50 tokens is marginal: the 10th distinct concept
  sits at median position 30, p90 46, and 6.6% (layer 12) / 16.4% (layer 6) of
  lists still yield fewer than 10 concepts. GLM-5.2 over-merged different words
  in 71% of lists before a mechanical un-merge corrected it; that un-merge
  over-splits in known ways (*wear*/*wore*, Latin-script translations become
  separate concepts) and leaves 34/83 duplicate-name merges per layer. The
  orderings survived all of this, but the corrected absolute counts carry these
  caveats.
- **The judge is a model, and the same model wrote the items.** GLM-5.2
  generates fragments and adjudicates readouts. The judge compares two
  Qwen-derived lists, not GLM's own text, which limits the exposure, but a shared
  blind spot would not surface. Judge and mechanical top-10 membership agree on
  95.3% of items.
- **Self-report is a proxy.** It shows the concept is accessible to the model's
  output at that prefix, not that it is linearly present in the residual, which
  is what a lens reads.
- **Sign tests ignore magnitude.** The R-lens finding — worse on average at
  layer 12, yet putting the concept in its top-10 for 5.1% of the failure cell
  against J-lens's 0.2% (J² 1.7%) — is invisible in a win/loss column.

Could I have addressed them? The collapse step and a multi-permutation shuffle
control (the current control is one deterministic roll) are each an hour of
work I ran out of time for. The off-diagonal question is a different project.

## 8. How I used LLMs, and how I checked them

**Codex (GPT-5 family), first day.** Set up the baselines and the first Hessian
estimator. I did not check its estimator closely enough: it used finite
differences, random projections and importance sampling I had not asked for,
and — the serious one — a development set of eight samples, which I only found
when a later batching constraint made no sense. I would have been unsurprised to
find a major error there, and did. Lesson recorded in the log: "I trusted Codex
too naively."

**Claude (this session), everything after.** Wrote the pipeline, the dataset
build, the analysis. My checking strategy was to read the code it wrote for the
parts that determine a conclusion — `jspace.py`, `concepts.py`, the prompts —
and to demand that every claimed number come from a stored file. Most of the
pitfalls in §6 were found because I read something and it looked wrong: the
suffix bug from reading `items.jsonl`, the judge asymmetry from reading
`browse/lens_only.md`, the "any layer" and "all layers recorded" docstrings from
reading `jspace.py`. Claude also made and then corrected several wrong
diagnoses in the open (the 17 boundary disagreements were called judge errors,
then rank errors, before being traced to bfloat16); those reversals are in
`data/README.md`. I would be moderately surprised by a major error in the cell
counts, less surprised by one in a prompt's wording having a residual bias.

**GLM-5.2 (via Mistral API), as instrument.** Generator, concept filter, judge,
quality grader, and the unfinished collapser. Every prompt is a module constant;
every paid call is in a ledger with exact token counts. Checks: the judge is
scored against seven recorded fixtures with hand-assigned answers; matches must
be quoted verbatim; fragment evidence comes from the tokenizer, not the model;
batching effects were measured (flat, ~4% on borderline items). The grader's
validity is supported by predicting lens ranks it never saw. Where I could not
verify — the collapse step's merges — I did not use the output.

What I did not check: individual self-report answers beyond spot checks; the
judge's reasons on items outside the fixtures and the ~120 I read by hand.

## 9. Artifacts

| | |
|---|---|
| dataset, cells, grades, browse tables | `j^2-lens/data/` |
| fitted operators (layers 6, 12) | `results/hessian_pile_l{6,12}_merged_qwen3.5-4b.pt` |
| hand-built battery | `configs/battery_cases.json` |
| concept list (artifact of record) | `configs/concepts.json` |
| all prompts | `src/j2_lens/{jspace,concepts,scoring,collapse}.py` |
| API ledger | `pilot/spend.json` |
| time log | `PROJECT_LOG.md` |
