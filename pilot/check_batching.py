"""Measure whether batching the adjudication degrades it.

Batching cuts cost and latency, but packing many items into one call risks the
judge losing track of which evidence belongs to which item. This adjudicates
the same recorded observations at several batch sizes and compares each against
the one-at-a-time verdicts, which are treated as the reference.
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from j2_lens.baselines import MODEL_ID, MODEL_REVISION
from j2_lens.jspace import (
    JUDGE_MODEL,
    annotate_fragments,
    judge_batch,
    load_api_key,
    vocabulary_words,
)

HERE = Path(__file__).resolve().parent
BATCH_SIZES = [1, 4, 8, 16]


def main() -> None:
    key = load_api_key(HERE.parents[1] / ".env")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    vocabulary = vocabulary_words(tokenizer)

    lines = (HERE / "observations.jsonl").read_text().splitlines()
    observations = [json.loads(line) for line in lines]
    cases = [
        {
            "concept": o["concept"],
            "lens_tokens": o["lens_top10"],
            "self_report": o["self_report"],
            "fragment_evidence": annotate_fragments(o["lens_top10"], vocabulary),
        }
        for o in observations
    ]
    print(f"judge = {JUDGE_MODEL}, {len(cases)} recorded items\n")

    results: dict[int, list[tuple[bool, bool]]] = {}
    for size in BATCH_SIZES:
        verdicts: list[dict] = []
        errors = 0
        for start in range(0, len(cases), size):
            chunk = cases[start : start + size]
            try:
                got, _ = judge_batch(key, chunk, ledger=HERE / "spend.json")
            except Exception as error:  # noqa: BLE001 - report, do not abort
                errors += 1
                print(f"  batch {size}: chunk at {start} failed: {error}")
                got = [{"lens": {}, "self_report": {}} for _ in chunk]
            verdicts.extend(got)
        results[size] = [
            (
                bool(v.get("lens", {}).get("present")),
                bool(v.get("self_report", {}).get("present")),
            )
            for v in verdicts
        ]
        fabrications = sum(len(v.get("fabricated_matches") or []) for v in verdicts)
        calls = -(-len(cases) // size)
        print(
            f"  size {size:>2}: {calls:>2} calls, "
            f"{fabrications} fabricated matches, {errors} structural errors"
        )

    reference = results[1]
    print(f"\n{'batch':>6}  {'flips vs singleton':>20}  {'rate':>7}")
    for size in BATCH_SIZES:
        flips = sum(a != b for a, b in zip(reference, results[size], strict=True))
        print(f"{size:>6}  {flips:>20}  {flips / len(cases):>6.1%}")


if __name__ == "__main__":
    main()
