"""Split the dataset into per-cell files for inspection.

`dataset.jsonl` stays the record. This derives two browsable views from it:

  data/cells/<cell>.jsonl   the same rows, one file per cell, quality grade
                            joined in, sorted strong -> medium -> weak
  data/browse/<cell>.md     a markdown table per cell with the columns a
                            person actually wants to eyeball

Regenerate after any change to dataset.jsonl or quality.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "data"
CELLS = ("self_report_only", "both", "lens_only", "neither")
GRADES = ("strong", "medium", "weak")
ORDER = {"strong": 0, "medium": 1, "weak": 2, None: 3}

HEADER = (
    "| quality | concept | probe | prefix | J | R "
    "| self-report | J-lens concepts (top-50 → 10) | why (grade) |"
)
RULE = "|---|---|---|---|---:|---:|---|---|---|"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def esc(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def row_markdown(r: dict) -> str:
    self_report = ", ".join(r.get("self_report") or [])
    lens = ", ".join(t.strip() for t in r["lens_top_k"])
    return (
        f"| {r['quality'] or ''} | **{esc(r['concept'])}** | `{esc(r['probe_term'])}` "
        f"| {esc(r['prefix'])} | {r['j_lens_rank']} | {r['r_lens_rank']} "
        f"| {esc(self_report)} | {esc(lens)} | {esc(r['quality_why'])} |"
    )


def main() -> None:
    quality = {r["concept"]: r for r in load(HERE / "quality.jsonl")}
    rows = load(HERE / "dataset.jsonl")
    for r in rows:
        grade = quality.get(r["concept"], {})
        r["quality"] = grade.get("quality")
        r["quality_why"] = grade.get("why", "")
    (HERE / "cells").mkdir(exist_ok=True)
    (HERE / "browse").mkdir(exist_ok=True)

    for cell in CELLS:
        sub = sorted(
            (r for r in rows if r["cell"] == cell),
            key=lambda r: (ORDER[r["quality"]], r["concept"]),
        )
        with (HERE / "cells" / f"{cell}.jsonl").open("w") as handle:
            for r in sub:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")

        by_grade = {g: sum(1 for r in sub if r["quality"] == g) for g in GRADES}
        counts = " · ".join(f"{g} {by_grade[g]}" for g in GRADES)
        lines = [
            f"# `{cell}` — {len(sub)} items",
            "",
            f"{counts}. Ranks are the concept token's rank at layer 12, lower is "
            "better. The J-lens column is its top-50 collapsed to the first 10 "
            "distinct concepts, which is what the judge saw; the raw top-10 "
            "tokens are in `lens_top_k_raw` in the JSONL. Both lists are shown "
            "in full.",
            "",
            HEADER,
            RULE,
            *(row_markdown(r) for r in sub),
        ]
        (HERE / "browse" / f"{cell}.md").write_text("\n".join(lines) + "\n")
        print(f"  {cell:<18} {len(sub):>5}  {counts}")

    index = [
        "# Browse the dataset",
        "",
        "One file per cell. `self_report_only` is the target set: the model names",
        "the concept and the J-lens does not, its top-50 collapsed to 10 distinct",
        "concepts. Within each file, `strong` items come first.",
        "",
        "| cell | items | file |",
        "|---|---:|---|",
    ]
    for cell in CELLS:
        n = sum(1 for r in rows if r["cell"] == cell)
        index.append(f"| `{cell}` | {n} | [{cell}.md]({cell}.md) |")
    (HERE / "browse" / "README.md").write_text("\n".join(index) + "\n")


if __name__ == "__main__":
    main()
