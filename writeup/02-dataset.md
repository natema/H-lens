# The evaluation set

## Why a new one was needed

The project began with the five qualitative examples from the J-lens and R-lens
posts, split 2 development / 3 held-out. Three held-out cases cannot support any
claim: the first result read "one win, two losses", which is indistinguishable
from noise. The published posts' *evaluation* sets are larger than their
memorable illustrations, so the illustrations were never meant to carry this.

Intermediate step: a 33-case hand-built battery across the five R-lens categories
(multihop, multilingual, association, typo, poetry), screened lens-blind. Useful,
still far too small.

## Construction of the J-space dataset

**Concepts from the model's own vocabulary.** A concept must be a single token
for a lens to emit it, so the vocabulary is the natural source. 248,077 tokens →
40,727 matching `^ [a-z]{4,}$` (the word-initial form a readout emits) → 14,990
at pile-10k frequency ≥ 3 with plain inflections dropped → **3,344 concepts**
after an LLM filter to concrete nouns. Frequency is measured over the whole of
pile-10k (9 seconds), because shape alone does not identify English in a
multilingual vocabulary: ` abogados` matches as readily as ` volcano`.

**Items are fragments ending on the probe.** Attention is causal, so the residual
at the probe depends only on the text up to and including it. The fragment
therefore *ends* at the probe and there is no discarded suffix. Verified
numerically: reading the probe position from the prefix versus the full sentence
agrees to relative 3.9e-6 in float32.

**Two independent readings per item**, which must be about the same model state:
1. the J-lens top-10 at the probe;
2. the model's own answer when asked, showing it *exactly the same prefix*, which
   concepts the situation evokes.

**Four cells, nothing filtered on agreement:**

| cell | n (float32) | meaning |
|---|---:|---|
| `self_report_only` | 1169 | J-lens misses it, the model has it — the target |
| `neither` | 1160 | the model does not hold the concept |
| `both` | 640 | positive control |
| `lens_only` | 375 | lens finds it, self-report does not |

Keeping only agreement would retain exactly the items the J-lens already handles,
leaving no headroom — selection on the outcome variable. `neither` is kept
because without it there is no way to separate "the lens failed" from "there was
nothing to find". That distinction is real: for one published example
(`association_fame`) Qwen's own final-layer readout ranks ` fame` at 27,379, so
no lens could have recovered it.

**Item quality grades**, from a lens-blind grader: `strong` 826, `medium` 1590,
`weak` 928. The grader sees the fragment, the probe and the concept, and never
any readout — showing it lens output would entangle item selection with the
behaviour of a system under comparison.

## A grade result worth reporting on its own

The grades are **not** ordered by lens difficulty:

| grade | n | J-lens geo-mean rank |
|---|---:|---:|
| strong | 826 | 102 |
| medium | 1590 | **162** |
| weak | 928 | 46 |

`medium` items are the hardest. This is coherent: `strong` means the concept is
nearly unavoidable at the probe, so the lens has a good chance too; `medium`
means the link needs a step of reasoning, which is what a linearised lens should
struggle with. The grade measures how clearly the concept is evoked, not how hard
it is to read out, and those come apart.

A lens-blind grader predicting lens behaviour it never saw (weak items are 2–3x
easier for the J-lens, and cluster in `both`/`lens_only`) is evidence the grade
measures something real rather than echoing the prompt.
