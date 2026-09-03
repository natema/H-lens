"""Collapse a lens readout's token list into distinct concepts.

A lens spends its top-k slots on variants of the same word — `" sushi"`,
`" Sushi"`, `"寿司"`, or the fragment `" volcan"` beside `" volcano"` — while
the model's self-report spends its slots on distinct concepts. Measured on the
dataset, the J-lens's top-10 holds 7.8 distinct concepts on average, and the
lenses differ (R-lens 8.2), so the budget is both smaller than it looks and
unequal across the methods being compared.

This takes a deeper list, merges tokens that are the same concept — casing,
spacing, inflection, fragment, other script, typo — and returns the first k
distinct concepts in order of first appearance, so every lens and the
self-report are judged on the same number of concepts.

The collapser must report which input tokens it merged into each concept, which
makes the step auditable: a concept whose cited tokens are not all in the input
is a fabrication and is dropped, exactly as the judge's quoted matches are
verified.
"""

from __future__ import annotations

import json
from typing import Any

from j2_lens.jspace import call_mistral

COLLAPSER_MODEL = "glm-5-2"

COLLAPSER_SYSTEM = """\
You are given ordered lists of tokens read out of a language model. Many tokens
in a list are the SAME concept written differently: capitalisation, a leading
space, plural or inflected forms, a fragment of a longer word, the same word in
another language or script, or a typo. Your job is to merge those into distinct
concepts.

For each list, walk it in order. Each time you meet a token whose concept has
not appeared yet, start a new concept named by its ordinary lowercase English
word. Each later token that is the same concept is merged into it. Stop after
the first {k} distinct concepts.

Rules:
- Merge only genuine variants of one word. "sushi", " Sushi", "寿司" are one
  concept. "volcan", " volcano", " volcanic" are one concept. "hammer" and
  "anvil" are two: related, but different things.
- A fragment that could complete to several unrelated words, or a token that is
  not a word at all (punctuation, code, a suffix like "ing"), is dropped rather
  than counted as a concept.
- Name each concept by its plain English word, lowercase, singular where the
  tokens are plural forms of a noun.
- For every concept, list verbatim every input token you merged into it.

Answer with JSON only, in the form
{"lists": [{"id": <int>, "concepts": [{"name": <str>, "tokens": [<str>, ...]}]}]}
"""


def collapse_batch(
    api_key: str,
    lists: list[list[str]],
    *,
    k: int = 10,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Collapse several token lists in one call.

    Returns, per input list, the verified concepts in order — each a dict with
    ``name`` and the input ``tokens`` it absorbed — and the raw exchange so the
    caller can record spend.
    """
    payload = {
        "lists": [
            {"id": index, "tokens": tokens} for index, tokens in enumerate(lists)
        ],
        "k": k,
    }
    exchange = call_mistral(
        api_key,
        [
            {"role": "system", "content": COLLAPSER_SYSTEM.replace("{k}", str(k))},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        model=COLLAPSER_MODEL,
        max_tokens=220 * len(lists) + 300,
        temperature=0.0,
    )
    try:
        raw = json.loads(exchange["content"]).get("lists", [])
    except json.JSONDecodeError:
        raw = []
    by_id = {int(entry["id"]): entry for entry in raw if "id" in entry}

    results: list[list[dict[str, Any]]] = []
    for index, tokens in enumerate(lists):
        entry = by_id.get(index) or {}
        available = {t.strip().lower() for t in tokens} | set(tokens)
        verified: list[dict[str, Any]] = []
        for concept in entry.get("concepts", []) or []:
            name = str(concept.get("name", "")).strip().lower()
            cited = [str(t) for t in concept.get("tokens", []) or []]
            # Every cited token must really be in the input; a concept that
            # cites none, or cites one that is not there, is not accepted.
            if not name or not cited:
                continue
            if not all(t in available or t.strip().lower() in available for t in cited):
                continue
            verified.append({"name": name, "tokens": cited})
            if len(verified) >= k:
                break
        results.append(verified)
    return results, exchange
