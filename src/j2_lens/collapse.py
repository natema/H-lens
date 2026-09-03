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
You are given SEVERAL ordered lists of tokens read out of a language model.
Treat every list completely independently: process each one on its own, and
return a result for EVERY list id you were given, even when two lists look
similar or identical. Never merge across lists and never skip a list.

Within one list, many tokens are the SAME concept written differently:
capitalisation, a leading space, plural or inflected forms, a fragment of a
longer word, the same word in another language or script, or a typo. Merge those
into distinct concepts.

For each list, walk it in order. Each time you meet a token whose concept has
not yet appeared IN THAT LIST, start a new concept named by its ordinary
lowercase English word. Later tokens of the same concept are merged into it.
Stop that list after its first {k} distinct concepts, then move to the next list.

Rules:
- Merge ONLY spelling-level variants of ONE word: casing, a leading space, a
  plural or inflection, a fragment, a translation into another language or
  script, a typo. "sushi", " Sushi", "寿司" are one concept.
- Never merge different words, however related. Hyponyms, brands, materials,
  parts and associates are all separate concepts. " shoes", " sneakers",
  " sandals", " footwear", " Adidas" and " worn" are SIX concepts, not one
  concept "shoe". "hammer" and "anvil" are two. If in doubt, do not merge.
- A fragment that could complete to several unrelated words, or a token that is
  not a word at all (punctuation, code, a suffix like "ing"), is dropped rather
  than counted.
- Name each concept by its plain English word, lowercase, singular for plurals.
- For every concept, list verbatim every input token you merged into it.

Answer with JSON only, exactly this form, one entry per input list id:
{"lists": [{"id": <int>, "concepts": [{"name": <str>, "tokens": [<str>, ...]}]}]}
"""


def _normal(token: str) -> str:
    return token.strip().lower().lstrip("_-/(")


def _is_variant(surface: str, name: str) -> bool:
    """Is ``surface`` plausibly a spelling-level variant of ``name``?

    Non-Latin text is accepted (a translation cannot be checked lexically),
    as are very short fragments. A Latin-script word must share most of its
    letters with the name: this passes shoes/shoe, volcanic/volcano,
    matrimoniale/matrimony and aluminum/aluminium, and rejects
    sneakers/shoe, Adidas/shoe, worn/shoe.
    """
    s, n = _normal(surface), name.strip().lower()
    if not s or s == n:
        return True
    if not any(c.isascii() and c.isalpha() for c in s):
        return True
    if len(s) < 4:
        return True
    from difflib import SequenceMatcher

    if s.startswith(n[:4]) or n.startswith(s[:4]):
        return True
    return SequenceMatcher(None, s, n).ratio() >= 0.75


def unmerge_nonvariants(
    raw_tokens: list[str], concepts: list[dict[str, Any]], k: int
) -> list[dict[str, Any]]:
    """Split back out tokens the collapser merged that are not variants.

    Rebuilds the concept sequence in order of first appearance in
    ``raw_tokens``, so the "first k distinct concepts" semantics survive the
    correction. Purely mechanical, so it costs no API call and is auditable.
    """
    owner: dict[str, str] = {}
    for concept in concepts:
        for tok in concept["tokens"]:
            owner.setdefault(tok, concept["name"])
    rebuilt: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    for tok in raw_tokens:
        name = owner.get(tok)
        if name is None:
            continue
        if not _is_variant(tok, name):
            name = _normal(tok)
        if name not in index:
            if len(rebuilt) >= k:
                continue
            index[name] = len(rebuilt)
            rebuilt.append({"name": name, "tokens": []})
        rebuilt[index[name]]["tokens"].append(tok)
    return rebuilt


def collapse_batch(
    api_key: str,
    lists: list[list[str]],
    *,
    k: int = 10,
    retries: int = 2,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Collapse several token lists in one call.

    Returns, per input list, the verified concepts in order — each a dict with
    ``name`` and the input ``tokens`` it absorbed — and the raw exchange.

    A list that comes back empty for a non-empty input is treated as *missing*,
    not as an answer: an empty list would later be judged as "the lens named
    nothing", which is a silent corruption rather than a failure. Missing ids
    are retried and then the batch is bisected, as in ``judge_batch``.
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
        max_tokens=320 * len(lists) + 300,
        temperature=0.0,
    )
    try:
        raw = json.loads(exchange["content"]).get("lists", [])
    except (json.JSONDecodeError, AttributeError):
        raw = []
    if not isinstance(raw, list):
        raw = []
    by_id: dict[int, dict[str, Any]] = {}
    for entry in raw:
        if isinstance(entry, dict) and "id" in entry:
            try:
                by_id[int(entry["id"])] = entry
            except (TypeError, ValueError):
                continue

    results: list[list[dict[str, Any]]] = []
    missing: list[int] = []
    for index, tokens in enumerate(lists):
        entry = by_id.get(index) or {}
        available = {t.strip().lower() for t in tokens} | set(tokens)
        verified: list[dict[str, Any]] = []
        concepts = entry.get("concepts", []) if isinstance(entry, dict) else []
        for concept in concepts or []:
            if not isinstance(concept, dict):
                continue
            name = str(concept.get("name", "")).strip().lower()
            cited = [str(x) for x in (concept.get("tokens") or [])]
            if not name or not cited:
                continue
            if not all(x in available or x.strip().lower() in available for x in cited):
                continue
            verified.append({"name": name, "tokens": cited})
            if len(verified) >= k:
                break
        verified = unmerge_nonvariants(tokens, verified, k)
        if tokens and not verified:
            missing.append(index)
        results.append(verified)

    if missing:
        if retries > 0:
            return collapse_batch(api_key, lists, k=k, retries=retries - 1)
        if len(lists) > 1:
            middle = len(lists) // 2
            left, _ = collapse_batch(api_key, lists[:middle], k=k)
            right, exchange = collapse_batch(api_key, lists[middle:], k=k)
            return left + right, exchange
        raise RuntimeError("collapser returned no concepts for a non-empty list")
    return results, exchange
