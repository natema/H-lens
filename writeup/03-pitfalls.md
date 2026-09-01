# Measurement pitfalls

Bugs and artifacts found during the work. Several changed a conclusion, so they
belong in the write-up rather than in a footnote. Each is stated with how it was
detected, because the detection method generalises better than the fix.

## 1. Selection on the outcome variable — twice

**First form.** The initial dataset design kept only items where the lens readout
and the model's self-report agreed. That retains exactly the items the J-lens
already handles, leaving no headroom for any correction.

**Second form.** Evaluating the correction only on the cell where the J-lens
fails. Restricted there, J² beats J-lens at p = 0.0069; the shuffled control
beats it at p = 4.9e-07. The apparent effect is regression to the mean.

Both were caught by asking what the selection rule conditions on. The fix in both
cases was to record every outcome and slice at analysis time.

## 2. The batch-wide reduction endpoint

The operator's target was "sum of target-layer residuals from the probe through
the penultimate token", implemented as `target[row, position : input_ids.shape[1] - 1]`
— one endpoint for the whole batch. Padding a shorter prompt would have summed
straight through the padding and silently produced a different operator. Guards
therefore required every development prompt to share a length, which is why the
development set was **eight (prompt, position) samples**: two 9-token prompts,
minus `skip_first=4` and the final position.

Fixed with a per-row `ends` tensor. Right padding is then exact, not approximate:
attention is causal, so a real position never attends to a later token. Verified
against a toy model with a causal mixing layer, and on Qwen at relative 3.9e-6.

**Consequence for the result:** the sample size mattered. On the same items, J²'s
geometric-mean rank improved from 1775.3 to 1050.2 at layer 6 and 223.7 to 190.1
at layer 12 once fitted properly. An earlier "J² is worse than J-lens" reading was
an artifact of the eight-sample fit.

## 3. `rank ≤ k` is optimistic under ties

`rank = count(logits > target) + 1` does not count ties, so it reports the *best*
position within a tied group. In bfloat16, ties are common.

Worked example. For one item ` Election` and ` electronically` both round to
logit 10.875000. ` Election` scores rank 10, `torch.topk` awards the tenth slot to
the other token, and the concept never appears in the list. In float32 the logits
are 10.844481 and 10.855117 — rank 11, correctly outside.

**This affects every rank number reported earlier in the project**, including the
battery tables and the layer-6/12 comparisons, since `baselines.py` uses the same
definition. It only bites at the top-k boundary, so aggregate geometric means are
essentially unaffected, but `pass@k` counts can be slightly inflated — more in
bfloat16 runs than float32 ones.

The mechanical test is now membership in the actual top-k list, which is
unambiguous and is what the judge is shown.

**Two wrong diagnoses were made before the right one**, which is itself worth
recording: the 17 disagreements were first called judge errors, then called rank
errors that the judge got right, and are finally understood as bfloat16
corrupting the list the judge was shown. The judge was never the faulty
component.

## 4. Precision was not the real problem; the hard cutoff was

Float32 is genuinely more faithful — the weights are bfloat16 either way and
float32 merely computes the same checkpoint's arithmetic with less rounding
error. And it is needed: **38% of boundary items have a 10th-vs-11th gap smaller
than bfloat16 can represent**, so bfloat16 orders them by rounding noise.

But the median boundary gap is 0.0617 logits, a probability ratio of **1.064**.
Declaring one token "in the top 10" and the next "out" on a 6% preference is
arbitrary at any precision.

Rebuilding all 3,344 readouts in float32 moved **8 items'** concept membership
(0.24%). By contrast, 25 items changed cell with a **byte-identical** top-10 list
— pure adjudication nondeterminism. **The judge's run-to-run variability moved
three times more labels than the precision fix did.**

The useful output was not the precision but `margin_to_kth`, recorded per item:
median −2.047, with only 2.5% of items within 0.05 logits of the boundary. The
cutoff is sharp for the bulk of the data and ambiguous for a few percent, which
can now be excluded by a stated threshold rather than assigned by rounding.

## 5. Instructions to a generator are not constraints

The generator was told "the evocation must come from the text BEFORE the probe;
never rely on what follows; it will be deleted." **64% of 3,343 items violated
it**: `"The score"` survived while `"was tied at halftime"` was discarded;
`"The membrane"` survived while `"surrounded the cytoplasm"` was discarded.

Rewriting the prompt to demand a probe-final fragment raised conformance to 74%.
Only rejecting non-conforming items at write time and retrying the concept
reached 100%. The general lesson: enforce structurally, and measure conformance
rather than assuming it.

## 6. A regex cannot see a synonym

The structural screen rejects items whose prefix contains the concept string. It
passed `concept "home", probe "house"` — *"She bought a suburban two-story with a
yard and turned it into her house"* — which sat in the target cell at rank 32
with nothing to infer. The lens-blind quality grader catches this class:
`program`/`code`, `design`/`blueprint`, `event`/`occasion`, `death`/`coffin`.

Telling that grader **where the lens actually reads** changed 32.3% of grades. An
earlier version withheld the probe word and described the fragment as stopping
"mid-sentence" — phrasing copied from the generator's prompt, where it does work,
and misleading for a grader, since nothing follows the text at all.

## 7. Nondeterminism that must be declared for reproducibility

- **Batched self-report generation.** Independent prompts through the GPU in one
  pass, not many items in one prompt. Outputs are not bit-identical (bfloat16
  batch nondeterminism): 8/24 lists differ verbatim at batch 8, but the
  decision-relevant concept-presence flips are ~1.8%. An item's output depends on
  which items share its batch, so exact reproduction requires the same batch size
  *and* the same item ordering; both are recorded in `manifest.json`.
- **The judge.** Two singleton passes agree perfectly at temperature 0, but
  batches of 4/8/16 each differ from singleton on the same single borderline item
  (~4%), with no growth in the rate. Malformed replies occur at every batch size
  including one, so `judge_batch` retries then bisects rather than silently
  returning fewer verdicts than items.
