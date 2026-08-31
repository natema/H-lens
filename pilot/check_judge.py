"""Score the judge against real recorded readouts with hand-assigned answers.

A pilot run only exercises whatever the generator happens to produce, so the
behaviour that most justifies having a judge — accepting a different surface
form while still refusing mere association — can go untested for a whole run.

Every list used here is real output captured in ``observations.jsonl``:
``lens_top10`` is the J-lens readout at the probe position and ``self_report``
is what Qwen3.5-4B answered when asked directly. Only the expected verdict is
supplied by hand, as the ground truth the judge is scored against.
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from j2_lens.baselines import MODEL_ID, MODEL_REVISION
from j2_lens.jspace import (
    JUDGE_MODEL,
    annotate_fragments,
    judge_agreement,
    load_api_key,
    vocabulary_words,
)

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "judge_fixtures.json"
LEDGER = HERE / "spend.json"


def main() -> None:
    key = load_api_key(HERE.parents[1] / ".env")
    fixtures = json.loads(FIXTURES.read_text())["fixtures"]
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    vocabulary = vocabulary_words(tokenizer)
    print(f"judge = {JUDGE_MODEL}, {len(fixtures)} recorded fixtures\n")

    failures = 0
    for fixture in fixtures:
        verdict, _ = judge_agreement(
            key,
            fixture["concept"],
            fixture["lens_top10"],
            fixture["self_report"],
            ledger=LEDGER,
            fragment_evidence=annotate_fragments(fixture["lens_top10"], vocabulary),
        )
        got = bool(verdict.get("agree"))
        expected = fixture["expected_agree"]
        ok = got == expected
        failures += not ok
        print(
            f"[{'ok  ' if ok else 'FAIL'}] {fixture['concept']:<11}"
            f" expect={str(expected):<5} got={str(got):<5}"
            f" lens={verdict['lens'].get('matched')!r}"
            f" self={verdict['self_report'].get('matched')!r}"
        )
        print(f"         prefix: {fixture['prefix']}")
        print(f"         lens  : {fixture['lens_top10']}")
        print(f"         self  : {fixture['self_report']}")
        print(f"         tests : {fixture['exercises']}")
        if verdict.get("fabricated_matches"):
            print(f"         FABRICATED: {verdict['fabricated_matches']}")
        if not ok:
            print(f"         judge reason: {verdict.get('reason')}")
        print()

    print(f"{len(fixtures) - failures}/{len(fixtures)} fixtures behaved as intended")


if __name__ == "__main__":
    main()
