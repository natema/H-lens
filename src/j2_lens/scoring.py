"""Score how strongly each item's fragment evokes its target concept.

The generator writes plausible-looking fragments that vary a lot in quality. A
dial moved "to the opposite position" does not really evoke a *switch*, and a
patron devouring a savory dish evokes *food* so obviously that a lens missing it
by a few ranks says little. Grading them lets the comparison run on the items
that carry evidence.

The grader is deliberately **lens-blind**: it sees the fragment and the target
concept, never any lens readout. Showing it the J-lens output would entangle
item selection with the behaviour of one of the systems under comparison, and a
set filtered that way could not support a fair claim about which lens recovers
more.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from j2_lens.jspace import call_mistral, record_spend

SCORER_MODEL = "glm-5-2"

SCORER_SYSTEM = """\
You grade items for an interpretability experiment.

Each item is a short TEXT, a PROBE WORD, and a TARGET CONCEPT. The probe word is
always the last word of the text, and the text is everything the model is shown
— nothing follows it.

What matters is that a reading instrument is applied at exactly one place: the
probe word, the final token. It reports which concepts the model is holding at
that point, having read the text up to and including the probe and nothing more.

So grade this: standing at the probe word, having just read this text, how
strongly does the target concept come to mind?

Grade "strong" when arriving at the probe word makes the concept nearly
unavoidable, while the text never names it. The probe word should be pulling its
weight: it is the word the instrument reads, so the concept should be live
precisely there, not merely somewhere earlier in the text.

Grade "medium" when the concept is a reasonable thing to have in mind at the
probe, but several other concepts fit about as well, or the link needs a step of
reasoning.

Grade "weak" when any of these hold:
- the concept does not really come to mind at the probe;
- the concept is restated by a word already in the text, or is a synonym of one
  — especially if the probe word itself is that synonym — so nothing is inferred;
- the concept is so generic that almost any text would evoke it;
- the text is more naturally about something else.

Examples, written as text | probe | concept -> grade:

"The scout climbed the ridge and raised a colored flag" | flag | signal ->
strong. The instrument reads at "flag", and a flag raised on a ridge is a signal
at exactly that word; the concept is never named.

"He reached for the dial on the wall and moved it to the opposite position" |
position | switch -> weak. At "position" the scene is a dial, and a dial is not a
switch; it reads as a clock or thermostat.

"The hungry restaurant patron eagerly devoured the hot, savory, and beautifully
plated dish" | dish | food -> weak. "Devoured", "restaurant" and "dish" state
food outright, and the probe word "dish" is itself the giveaway, so nothing is
inferred.

"The architect examined the load-bearing columns, cross-bracing, and steel beams"
| beams | structure -> medium. Apt at "beams", but generic: many building scenes
evoke it.

Be strict. Most items are not strong. Answer with JSON only."""


def score_batch(
    api_key: str,
    items: list[dict[str, Any]],
    ledger: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Grade one batch, returning a verdict per item in the input order.

    The raw exchange is returned as well so a threaded caller can record spend
    under its own lock instead of racing on the ledger file.
    """
    payload = {
        "items": [
            {
                "id": index,
                "text": item["prefix"],
                "probe_word": item["probe_term"],
                "target_concept": item["concept"],
            }
            for index, item in enumerate(items)
        ],
        "answer_format": {
            "verdicts": [
                {
                    "id": "int, echoing the item id",
                    "strength": "one of: strong, medium, weak",
                    "why": "one short sentence",
                }
            ]
        },
    }
    exchange = call_mistral(
        api_key,
        [
            {"role": "system", "content": SCORER_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        model=SCORER_MODEL,
        max_tokens=120 * len(items) + 400,
        temperature=0.0,
    )
    if ledger is not None:
        record_spend(ledger, exchange, note=f"quality score x{len(items)}")

    try:
        raw = json.loads(exchange["content"]).get("verdicts", [])
    except json.JSONDecodeError:
        raw = []
    by_id = {int(v["id"]): v for v in raw if "id" in v}
    verdicts = []
    for index in range(len(items)):
        verdict = by_id.get(index) or {}
        strength = str(verdict.get("strength", "")).strip().lower()
        # An unparseable or invented grade becomes None rather than a default,
        # so a failed call is visible instead of quietly grading everything.
        verdicts.append(
            {
                "strength": strength if strength in ("strong", "medium", "weak")
                else None,
                "why": str(verdict.get("why", "")).strip(),
            }
        )
    return verdicts, exchange
