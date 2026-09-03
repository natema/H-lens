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

The model stage is not reproducible. Two runs at ``temperature=0`` over the
same 800 words in the same order and the same batch size agreed with Jaccard
0.74 and accepted 22.0% and 19.0% respectively; runs at different batch sizes
(10 to 80) agreed with Jaccard 0.68 to 0.77, so batch size is not the cause.
The acceptance *rate* is stable at 19 to 22% while *membership* churns by
roughly a quarter per run. A free-form list is more sensitive to small logit
differences than the judge's structured per-item verdict, which was stable.

Consequently ``configs/concepts.json`` is the artifact of record: rerunning
``j2-concepts`` produces a different list of similar size and quality, not the
same one. This does not bias the lens comparison, since no lens sees the
concept list and every item is validated downstream regardless of which
concepts were drawn, but it does mean the concept list cannot be regenerated
from the code alone.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from j2_lens.jspace import call_mistral
from j2_lens.spend import record_spend

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


EXPLICIT_SCREEN_SYSTEM = """\
You screen a word list for an interpretability dataset.

Flag any word that is sexually explicit, pornographic, or names a sex act, a \
sex worker, a sexual body part, or an adult-entertainment category. Also flag \
slurs.

Do NOT flag ordinary words merely because they can appear in an adult context: \
body, bedroom, dating, romance, kiss, marriage are all fine.

Answer with JSON only."""

# The screen over-flags clinical and historical vocabulary, which is not what it
# is for. These are kept even when flagged, and the decision is recorded here
# rather than applied by hand so a rerun reproduces the same concept list.
SCREEN_KEEP = frozenset({"uterus", "womb", "circumcision", "slave"})


def screen_explicit(
    api_key: str,
    words: list[str],
    ledger: Path | None = None,
    batch_size: int = 200,
) -> list[str]:
    """Return the words to remove as sexually explicit.

    Only words actually offered can be flagged, so the screen cannot invent an
    entry, and anything in SCREEN_KEEP is retained regardless.
    """
    flagged: set[str] = set()
    for start in range(0, len(words), batch_size):
        chunk = words[start : start + batch_size]
        exchange = call_mistral(
            api_key,
            [
                {"role": "system", "content": EXPLICIT_SCREEN_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "words": chunk,
                            "answer_format": {
                                "flagged": ["the words to remove, verbatim"]
                            },
                        }
                    ),
                },
            ],
            max_tokens=1500,
            temperature=0.0,
        )
        if ledger is not None:
            record_spend(ledger, exchange, note="explicit screen")
        try:
            got = json.loads(exchange["content"]).get("flagged", [])
        except json.JSONDecodeError:
            got = []
        offered = set(chunk)
        flagged.update(w for w in got if w in offered and w not in SCREEN_KEEP)
    return sorted(flagged)


def frequency_table(
    tokenizer: Any, documents: list[str], batch: int = 200
) -> dict[str, int]:
    """Corpus frequency of every word-initial lowercase alphabetic token.

    The vocabulary is multilingual, so shape alone does not identify English:
    " abogados" and " abertas" match the pattern as readily as " volcano".
    Counting over an English corpus supplies the missing signal, and the whole
    of pile-10k scans in seconds.
    """
    counts: Counter[int] = Counter()
    for start in range(0, len(documents), batch):
        encoded = tokenizer(
            documents[start : start + batch], truncation=True, max_length=2048
        )["input_ids"]
        for ids in encoded:
            counts.update(ids)
    frequency: dict[str, int] = {}
    for token_id, count in counts.items():
        surface = tokenizer.convert_tokens_to_string(
            [tokenizer.convert_ids_to_tokens(token_id)]
        )
        if CANDIDATE_PATTERN.match(surface):
            word = surface.strip()
            frequency[word] = frequency.get(word, 0) + count
    return frequency


def is_inflection(word: str, vocabulary: set[str]) -> bool:
    """True when the word is a plain inflection of another candidate.

    Keeping both "abandon" and "abandoned" spends filter calls on near
    duplicates and would later produce two items about the same concept.
    """
    return (
        (word.endswith("s") and word[:-1] in vocabulary)
        or (word.endswith("ing") and word[:-3] in vocabulary)
        or (word.endswith("ed") and word[:-2] in vocabulary)
    )


def select_candidates(
    frequency: dict[str, int], min_count: int = 3
) -> list[tuple[str, int]]:
    words = set(frequency)
    kept = [
        (word, count)
        for word, count in frequency.items()
        if count >= min_count and not is_inflection(word, words)
    ]
    return sorted(kept, key=lambda pair: -pair[1])


def filter_to_concepts(
    api_key: str,
    words: list[str],
    ledger: Path | None = None,
    batch_size: int = 80,
) -> tuple[list[str], list[str]]:
    """Split candidate words into accepted concepts and rejects."""
    accepted: list[str] = []
    rejected: list[str] = []
    seen: set[str] = set()
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
        # Dedupe: a word can be offered twice across batches only by mistake,
        # but the model can also repeat one inside a single reply.
        kept = [w for w in dict.fromkeys(keep) if w in offered and w not in seen]
        seen.update(kept)
        accepted.extend(kept)
        rejected.extend(w for w in chunk if w not in set(kept))
        print(
            f"  filtered {start + len(chunk)}/{len(words)}: kept {len(kept)}",
            flush=True,
        )
    return accepted, rejected


def main(argv: list[str] | None = None) -> None:
    import argparse

    from transformers import AutoTokenizer

    from j2_lens.baselines import MODEL_ID, MODEL_REVISION
    from j2_lens.development import load_pile_documents
    from j2_lens.jspace import load_api_key

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pile-docs", type=int, default=10000)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--limit", type=int, help="cap the candidates filtered")
    parser.add_argument("--output", type=Path, default=root / "configs/concepts.json")
    parser.add_argument("--ledger", type=Path, default=root / "pilot/spend.json")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-explicit-screen", action="store_true")
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    documents = load_pile_documents(args.pile_docs, offline=args.offline)
    print(f"scanning {len(documents)} documents", flush=True)
    frequency = frequency_table(tokenizer, documents)
    candidates = select_candidates(frequency, args.min_count)
    if args.limit:
        candidates = candidates[: args.limit]
    print(
        f"{len(frequency)} word-shaped tokens; {len(candidates)} candidates at "
        f"frequency >= {args.min_count} after dropping inflections",
        flush=True,
    )

    key = load_api_key(root / ".env")
    words = [word for word, _ in candidates]
    accepted, rejected = filter_to_concepts(
        key, words, ledger=args.ledger, batch_size=args.batch_size
    )
    removed: list[str] = []
    if not args.skip_explicit_screen:
        removed = screen_explicit(key, accepted, ledger=args.ledger)
        accepted = [w for w in accepted if w not in set(removed)]
        print(f"explicit screen removed {len(removed)}: {removed}", flush=True)
    counts = dict(candidates)
    payload = {
        "schema_version": 1,
        "source": (
            f"Single-token lowercase alphabetic words with a leading space, the "
            f"form a readout emits, present at least {args.min_count} times in "
            f"{len(documents)} NeelNanda/pile-10k documents, excluding plain "
            f"inflections of another candidate, then filtered by the generator "
            f"model to concrete common nouns."
        ),
        "n_candidates": len(words),
        "n_accepted": len(accepted),
        "removed_adult_content": removed,
        "screen_keep": sorted(SCREEN_KEEP),
        "concepts": [
            {"word": word, "pile_frequency": counts[word]} for word in accepted
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"accepted {len(accepted)}/{len(words)}; wrote {args.output}")


if __name__ == "__main__":
    main()
