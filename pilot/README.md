# J-space pilot

Generates candidate probe/concept items with an API model, then validates each
one twice against Qwen3.5-4B and keeps only items where the two checks agree.

```bash
uv run python pilot/run_pilot.py --concepts basketball,japan,chess
```

## The causal constraint

Attention is causal, so the residual at the probe position depends only on the
tokens up to and including the probe. Two consequences shape the whole design:

- Everything after the probe is invisible to the lens, so it is cut. The item
  is the **prefix**, and the rest of the generated sentence is discarded.
- The self-report question must show the model the *same* prefix and nothing
  more, or the two checks would not be asking about the same model state.

`verify_causal_equivalence` checks this numerically by reading the probe
position from the prefix and from the full sentence. It reports a *relative*
deviation, because the absolute figure is dtype-dependent: on this model the
same check gives ~7e-3 in bfloat16 and ~5e-7 in float32. Only the float32 number
reflects the mathematics; the bfloat16 residue is rounding, not leakage.

## The two checks

1. **J-lens** — is the concept in the top-k readout at the probe, at any layer?
2. **Self-report** — asked directly, does the model list the concept itself?

The self-report question asks what the *situation* is about, not what the probe
word evokes. Asking about the word alone produced dictionary associations that
ignored the context entirely: for a prefix ending "tossing ... she lay awake"
the model answered "consciousness, alertness, vigilance" while the J-lens had
insomnia at rank 2.

### The judge

Exact string matching is too brittle on both sides, because each readout may
name the concept with a different surface form. `judge_agreement` asks
GLM-5.2 (temperature 0) whether the concept is named in each list. It resolves the case this was built for: the lens surfaces `Japanese`
while the self-report says `japan`, and those are the same concept.

The judge is deliberately strict, because a liberal reading would match every
list to every concept and destroy the measurement. Three traps are ruled out
explicitly, each one observed in the pilot before it was fixed:

| trap | example | verdict |
|---|---|---|
| association | `pawn`, `board`, `king` for *chess* | absent |
| broader category | `ceremony`, `rites` for *wedding* | absent |
| subword fragment | `noct` for *insomnia*, `Vol` for *volcano* | absent |

The judge must also quote its match verbatim from the list it claims. Any
`matched` value absent from that list is treated as a fabrication, recorded in
`fabricated_matches`, and the claim discarded. This check was added after a
judge reported `noct` as a match for *insomnia* while justifying it as
"directly named in both lists".

Both generation and judging use GLM-5.2. `pilot/check_judge.py` pins the
judge's behaviour with fixtures whose answers are known, because a pilot run
only exercises whatever the generator happens to produce — in one run every
agreement was an exact string match, so the variant handling that motivates
having a judge at all went untested:

```bash
uv run python pilot/check_judge.py
```

It covers the accepting cases (`Japanese`/`japan`, `火山`/`volcano`) and each
rejecting trap (Tokyo for Japan, `ceremony`/`matrim` for wedding, `noct` for
insomnia, chess pieces for chess). All six behave as intended.

One coupling to keep in mind: the same model writes the items and judges the
readouts, so a blind spot shared between the two roles would not show up here.
The judge compares two Qwen-derived lists rather than anything GLM wrote, which
limits the exposure, but it is not zero.

A concept is a *token*, and which token depends on casing: the model holds Japan
as `" Japan"`, not `" japan"`. Every single-token surface form is scored and the
best-ranked one is used, with the per-variant ranks recorded.

Thinking is disabled for the self-report. Qwen3.5 prefills a reasoning block
that routinely runs past any sane token budget, and a truncated block yields no
answer at all — the parser returns `None` rather than scraping words out of the
scratchpad, which would silently manufacture a concept list.

## Files

| file | contents |
|---|---|
| `prompt_system.txt` | system prompt sent to the generator, verbatim |
| `prompt_user.txt` | user prompt sent to the generator, verbatim |
| `mistral_response.json` | request payload, raw reply, token usage |
| `items.json` | parsed items, derived prefixes, discarded suffixes, screen results |
| `results.json` | both checks per item, in full |
| `self_report_raw/` | the model's untouched answer per item |
| `spend.json` | running API cost ledger |

## Cost tracking

The Mistral API returns exact token counts per call but exposes **no billing
endpoint** (`/v1/usage`, `/v1/billing`, `/v1/credits` all 404). `spend.json`
therefore multiplies measured tokens by published rates. The Mistral Large rate
is from `mistral.ai/pricing`; the `glm-5-2` rate is not listed there and comes
from a third-party aggregator, so it is approximate. Treat the console at
`console.mistral.ai` as authoritative.
