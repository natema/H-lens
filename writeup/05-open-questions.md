# Open questions and what a follow-up should do

## Immediate, cheap, and would strengthen the claim

**Average the shuffled control over many permutations.** `j2_shuffled` is a
single deterministic `torch.roll` by one coordinate. It is the right null —
preserves the row-norm distribution, destroys the coordinate correspondence — but
it is one draw, not a reference distribution. Repeating over 20–50 random
permutations would turn "the correction does not beat its shuffle" from
directional into a quantitative p-value. Needs no new model passes, only repeated
readouts against the stored operator.

**Evaluate at layer 6 on the full dataset.** The operator exists, and layer 6 is
the early-layer regime the project is actually about. The dataset currently
adjudicates at layer 12 only.

**Check `medium` items specifically.** They are the hardest for the J-lens
(geo-mean 162 against `strong`'s 102), on the reasoning that they require an
inferential step. If a correction were going to help anywhere, that is the
plausible place, and it has not been tested as a separate hypothesis.

## Substantive

**Cross-coordinate curvature.** The diagonal carries <1% of Hessian Frobenius
energy. A negative result for the diagonal is weak evidence about curvature in
general. A low-rank or subspace-restricted second-order term is the natural next
model, and `PROJECT_IDEA.md` anticipates it.

**Why are the lenses complementary?** R-lens is worse on average at layer 12 yet
reaches top-10 on 4.9% of the failure cell against J-lens's 0.1%. Characterising
*which* items each recovers is more informative than another aggregate, and this
dataset is large enough to support that analysis.

**Does the correction help at all, anywhere?** Every aggregate says no. But the
shuffled control also beats J-lens slightly and significantly across all items
(p = 0.007), which means adding *some* perturbation of this scale helps a little,
for reasons unrelated to curvature. Identifying that mechanism — rescaling,
variance subtraction, an implicit intercept — would explain the one real effect
in the data.

## Dataset improvements

- **Move the synonym check to generation.** The quality grader catches
  probe-is-a-synonym-of-concept after the fact; the same check could reject at
  write time, as the probe-final rule now does.
- **Exclude the ambiguous band by policy.** `margin_to_kth` is recorded; a stated
  threshold (±0.05 logits removes 2.5% of items) should be applied in analysis
  rather than left to the reader.
- **A second self-report phrasing.** The current one asks what the situation is
  about. Whether the cells are stable under a different phrasing is untested, and
  it is the least-validated component of the pipeline.
