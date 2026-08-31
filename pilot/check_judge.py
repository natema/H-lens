"""Probe the judge on cases with known correct answers.

The pilot only exercises whatever the generator happens to produce, so the
behaviour that matters most — accepting a different surface form while
rejecting mere association — can go untested for a whole run. These fixtures
pin it down directly.
"""

from __future__ import annotations

from pathlib import Path

from j2_lens.jspace import JUDGE_MODEL, judge_agreement, load_api_key

LEDGER = Path(__file__).resolve().parent / "spend.json"

# (concept, lens list, self-report list, expected agreement, what it tests)
FIXTURES = [
    (
        "japan",
        [" sushi", " restaurants", "寿司", "Japanese", " Sushi", " Japanese"],
        ["restaurant", "meal", "dinner", "food", "japan", "tokyo"],
        True,
        "demonymic form on one side, plain name on the other",
    ),
    (
        "japan",
        [" sushi", " rice", " fish", " chef", " temple"],
        ["food", "meal", "tokyo", "culture"],
        False,
        "association only; Tokyo is not Japan",
    ),
    (
        "wedding",
        [" Ceremony", " ceremony", " rites", " matrim"],
        ["wedding", "marriage", "bride"],
        False,
        "superordinate category and a fragment are not the concept",
    ),
    (
        "insomnia",
        [" sleep", " night", " waking", " noct"],
        ["insomnia", "sleeplessness"],
        False,
        "subword fragment is not the concept",
    ),
    (
        "volcano",
        [" volcan", " volcanic", "火山", " magma"],
        ["volcano", "lava"],
        True,
        "other-script and adjectival forms count",
    ),
    (
        "chess",
        [" pawn", " board", " king", " rook"],
        ["chess", "strategy"],
        False,
        "chess pieces are association, not the concept",
    ),
]


def main() -> None:
    key = load_api_key(Path(__file__).resolve().parents[2] / ".env")
    print(f"judge = {JUDGE_MODEL}\n")
    failures = 0
    for concept, lens, self_report, expected, description in FIXTURES:
        verdict, _ = judge_agreement(key, concept, lens, self_report, ledger=LEDGER)
        got = bool(verdict.get("agree"))
        ok = got == expected
        failures += not ok
        print(
            f"[{'ok ' if ok else 'FAIL'}] {concept:<10} expect={expected!s:<5} "
            f"got={got!s:<5} lens={verdict['lens'].get('matched')!r}"
        )
        print(f"        {description}")
        if not ok:
            print(f"        reason given: {verdict.get('reason')}")
    print(f"\n{len(FIXTURES) - failures}/{len(FIXTURES)} fixtures behaved as intended")


if __name__ == "__main__":
    main()
