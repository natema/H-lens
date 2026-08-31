"""Select target concepts from the model's own vocabulary.

A concept has to be a single token for the lens to be able to emit it, so the
natural place to draw concepts from is the vocabulary itself. Frequency is
estimated from the same pile-10k corpus the lens operators are fitted on, so
common words are preferred over vocabulary curiosities.

Selection is two stages. A mechanical filter produces candidates that are
single lowercase alphabetic tokens with a leading space, which is the form a
readout actually emits. A model then picks out the concrete nouns, because
whether " running" or " gentle" names a thing is a judgement no regular
expression settles.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from j2_lens.jspace import call_mistral, record_spend

# Leading space, lowercase, alphabetic, at least four characters. The leading
# space is what makes it a word-initial token rather than a continuation piece.
CANDIDATE_PATTERN = re.compile(r"^ [a-z]{4,}$")

FILTER_SYSTEM = """\
You select target concepts for an interpretability dataset.

A good concept is a CONCRETE COMMON NOUN naming a thing, place, activity, \
event, animal, occupation, or category that a short piece of text could evoke \
without naming it. Examples: basketball, volcano, wedding, hospital, chess, \
insomnia, harvest, telescope.

Reject:
- verbs, adjectives, adverbs, prepositions, pronouns, conjunctions
- abstract grammatical or discourse words: however, therefore, various, general
- words that are mainly a form of another word: running, better, houses
- proper nouns and brand names
- highly abstract nouns that nothing concrete evokes: aspect, factor, context, \
issue, matter, thing
- anything whose meaning is unclear out of context

Keep the bar high. A concept must be something a scene could point at without \
saying it.

Answer with JSON only."""


def frequency_ranked_candidates(
    tokenizer: Any, documents: list[str], limit: int = 4000
) -> list[tuple[str, int]]:
    """Vocabulary tokens matching the candidate shape, by corpus frequency."""
    counts: Counter[int] = Counter()
    for document in documents:
        encoded = tokenizer(document, truncation=True, max_length=2048)
        counts.update(encoded["input_ids"])
    ranked: list[tuple[str, int]] = []
    for token_id, count in counts.most_common():
        surface = tokenizer.convert_tokens_to_string(
            [tokenizer.convert_ids_to_tokens(token_id)]
        )
        if CANDIDATE_PATTERN.match(surface):
            ranked.append((surface.strip(), count))
        if len(ranked) >= limit:
            break
    return ranked


def filter_to_concepts(
    api_key: str,
    words: list[str],
    ledger: Path | None = None,
    batch_size: int = 60,
) -> tuple[list[str], list[str]]:
    """Split candidate words into accepted concepts and rejects."""
    accepted: list[str] = []
    rejected: list[str] = []
    for start in range(0, len(words), batch_size):
        chunk = words[start : start + batch_size]
        exchange = call_mistral(
            api_key,
            [
                {"role": "system", "content": FILTER_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "candidates": chunk,
                            "answer_format": {
                                "concepts": ["the accepted words, verbatim"]
                            },
                        }
                    ),
                },
            ],
            max_tokens=40 * len(chunk) + 300,
            temperature=0.0,
        )
        if ledger is not None:
            record_spend(ledger, exchange, note=f"concept filter x{len(chunk)}")
        try:
            keep = json.loads(exchange["content"]).get("concepts", [])
        except json.JSONDecodeError:
            keep = []
        # Only words actually offered may be accepted, so an invented one does
        # not enter the concept list.
        offered = set(chunk)
        kept = [w for w in keep if w in offered]
        accepted.extend(kept)
        rejected.extend(w for w in chunk if w not in set(kept))
        print(
            f"  filtered {start + len(chunk)}/{len(words)}: kept {len(kept)}",
            flush=True,
        )
    return accepted, rejected
