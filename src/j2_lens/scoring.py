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
You grade items for an interpretability dataset.

Each item is a sentence FRAGMENT and a TARGET CONCEPT. The fragment stops \
mid-sentence. The question is only this: reading the fragment and nothing else, \
how strongly does a competent reader think of the target concept?

Grade "strong" when the fragment points at the concept specifically and almost \
unavoidably, while never naming it. A reader asked what this is about would \
very likely say the target word.

Grade "medium" when the concept is a reasonable thing to think of, but the \
fragment also fits several other concepts about as well, or the connection \
needs a step of reasoning.

Grade "weak" when any of these hold:
- the fragment does not really evoke the concept at all;
- the concept is essentially restated by a word already in the fragment, or is \
a synonym of one, so nothing has to be inferred;
- the concept is so abstract or generic that almost any scene would evoke it;
- the fragment is more naturally about something else.

Examples:
- fragment "The scout climbed the ridge and raised a colored flag", concept \
"signal" -> strong. A raised flag on a ridge is a signal, and the word is absent.
- fragment "He reached for the dial on the wall and moved it to the opposite \
position", concept "switch" -> weak. A dial is not a switch, and the scene reads \
as a clock or thermostat.
- fragment "The hungry restaurant patron eagerly devoured the hot savory dish", \
concept "food" -> weak. "Devoured", "restaurant" and "dish" make food explicit; \
nothing is inferred.
- fragment "The architect examined the load-bearing columns, cross-bracing and \
steel beams", concept "structure" -> medium. Apt, but generic: many building \
scenes evoke it.

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
            {"id": index, "fragment": item["prefix"], "target_concept": item["concept"]}
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
