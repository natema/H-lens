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
| `self_report_only` | 1183 | 35.4% | **J-lens misses it, the model has it** |
| `neither` | 1160 | 34.7% | the model does not hold the concept at the probe |
| `both` | 635 | 19.0% | positive control |
| `lens_only` | 366 | 10.9% | lens finds it, self-report does not |

Of the 1,183, **1,171** are also a miss by token rank (`target_strict`).

Nothing is filtered on agreement. Keeping only `both` would retain exactly the
items the J-lens already handles, leaving no headroom for a correction to show
any effect — selection on the outcome variable. `self_report_only` is the useful
cell precisely because the self-report independently establishes the concept is
present, so a J-lens miss there is a real failure rather than an absent target.
`neither` is kept for the same reason it matters: without it there is no way to
distinguish "the lens failed" from "there was nothing to find".

## Two measures of "the J-lens surfaced it"

The cell labels and the rank columns answer different questions, and both are
recorded per item:

- `lens_hit_judge` — does the top-10 readout *name* the concept? This is what
  defines the cells. It counts a variant or synonym, so ` trail` counts for
  *path* and ` warranty` for *guarantee*.
- `lens_hit_topk` — does the concept's own token appear in the top-10 list?
- `lens_hit_rank_optimistic` — was its computed rank <= 10? Kept only for
  comparison; do not use it (see below).

They disagree on 158 of 3,344 items (4.7%), almost all in one direction: the
judge sees a variant or synonym that the token itself does not supply, counting
` trail` for *path* and ` warranty` for *guarantee*. Only 4 items now go the
other way. `target_strict` marks the 1,181 items both measures call a miss.

### Why rank <= 10 is the wrong test

`rank_and_topk` computes `rank = count(logits > target) + 1`, which is
**optimistic under ties**, and bfloat16 produces ties readily. For the *election*
item, ` Election` and ` electronically` both round to logit 10.875000, so
` Election` scores rank 10 while `topk` awards the tenth slot to the other token.
The judge, shown the list, correctly reports no word naming an election. In
float32 the tie dissolves: the logits are 10.844481 and 10.855117, and the rank
is 11, genuinely outside the top 10.

This accounted for every one of the 17 items previously described here as judge
errors. They were rank errors; the judge was right in all of them. Membership in
the actual top-k list is unambiguous and is what the judge is shown, so it is the
mechanical measure the dataset uses.

**The rank tables below are computed from `lens_hit_rank` data — the token's
rank — while the set they are computed over is defined by the judge.** That is
why J-lens shows a non-zero top-10 rate on a set defined as "the J-lens did not
name it"; by the judge's own measure it is zero by construction.

## Layer-12 ranks on the target set

| lens | geometric mean | median | top-10 | top-100 |
|---|---:|---:|---:|---:|
| J-lens | 252 | 163 | 1.0% | 41.9% |
| R-lens | 322 | 249 | 5.2% | 34.4% |
| logit lens | 9,535 | 13,799 | 0.3% | 3.0% |

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
- **All 3,344 concepts** now have an item. An earlier build reported 42 as
  having "produced no conforming fragment", which was wrong: the retry loop was
  still converging (3344 → 624 → 139 → 42 concepts left per pass) and simply hit
  a fixed three-pass limit. It now continues while passes keep recovering
  concepts. Separately, three chunks had been lost whole to a `KeyError` when one
  malformed entry in a reply discarded the other nine concepts with it.

## Prompts

Every model-facing prompt is a module constant, so the pipeline can be audited
and rerun from the repository alone:

| prompt | location | role |
|---|---|---|
| `GENERATOR_SYSTEM` | `jspace.py` | writes the probe-final fragments |
| `JUDGE_SYSTEM` | `jspace.py` | adjudicates whether each list names the concept |
| `SELF_REPORT_TEMPLATE` | `jspace.py` | what Qwen3.5-4B is asked |
| `FILTER_SYSTEM` | `concepts.py` | picks concrete nouns from the vocabulary |
| `EXPLICIT_SCREEN_SYSTEM` | `concepts.py` | removes sexually explicit words |
| `SCORER_SYSTEM` | `scoring.py` | grades evocation strength |

`pilot/` additionally stores a verbatim request and response pair
(`prompt_system.txt`, `prompt_user.txt`, `mistral_response.json`) so the exact
bytes sent over the wire can be inspected, not only the templates.

## Item quality grades (`quality.jsonl`)

The generator writes plausible-looking fragments of very uneven quality, so each
item is graded `strong` / `medium` / `weak` on one question: reading the fragment
alone, how strongly does a competent reader think of the target concept?

The grader is **lens-blind** — it sees the fragment and the concept, never any
readout. Showing it lens output would entangle item selection with the behaviour
of a system under comparison, and a set filtered that way could not support a
fair claim about which lens recovers more. The grades live in their own file
keyed by concept so they join onto any readout of the same items.

```bash
uv run python pilot/score_dataset.py
```

`strong 826, medium 1590, weak 928`.

The grader is told the probe word and that it is the final token, because the
readout happens at exactly that position: the concept has to be live *there*, not
merely somewhere earlier in the text. An earlier version of the prompt withheld
the probe and described the text as stopping "mid-sentence", which was copied
from the generator's framing and is wrong here — nothing follows the text, so
there is no continuation to reason about. Adding the probe changed **32.3%** of
grades, and the newly-weak items are exactly the pathology the structural screen
cannot see: concept `program` with probe `code`, `design` with `blueprint`,
`event` with `occasion`, `death` with `coffin`.

Typical `weak` verdicts catch what the structural screen cannot: *"The word
'house' is already in the fragment, and 'home' is essentially a synonym here."*
That item — `home`, probe `house`, in the target cell at rank 32 — is the case
the grade exists for. The screen only rejects the literal concept string, so a
synonym of the concept passes it.

### The grade predicts lens failure without seeing the lens

| grade | n | J-lens geo-mean rank | median |
|---|---:|---:|---:|
| strong | 826 | 102 | 79 |
| medium | 1590 | 162 | 132 |
| weak | 928 | **46** | **29** |

Items graded `weak` are roughly 2 to 3 times easier for the J-lens than the rest, and
they cluster in the `both` and `lens_only` cells. That is what the mechanism
predicts: a weak item has the concept sitting as a synonym of a word already in
the fragment, so the lens surfaces it trivially. The grader saw none of this, so
the agreement is evidence the grade measures something real rather than echoing
the prompt.

The grades are **not** ordered by lens difficulty: `medium` items are the hardest
(geo-mean 162), harder than `strong` (102). That is coherent rather than
contradictory. `strong` means the concept is nearly unavoidable at the probe, so
the lens also has a better chance of surfacing it; `medium` means the link needs
a step of reasoning, which is precisely what a linearised lens would struggle
with. The grade measures how clearly the concept is evoked, not how hard it is to
read out.

So `strong` is the right filter for an evaluation set — a miss there is
unambiguous — while `weak` is the set to discard.

### The restricted evaluation set

Target cell restricted to `strong`: **362 items**, J-lens geometric-mean rank
264. This is the set to use for a lens comparison — the concept is unambiguously
evoked, the model demonstrably holds it, and the J-lens does not surface it.
