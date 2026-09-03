"""Rebuild the dataset's cells from the collapsed J-lens readout.

The dataset's cells are defined against the J-lens. Its raw top-10 holds ~7.8
distinct concepts against the self-report's 10, so the cells were computed on
an unequal budget. This takes the J-lens top-50 collapsed to its first 10
distinct concepts, judged with the same procedure, and rewrites each item's
cell from it.

Reads the pre-collapse record data/dataset_raw_top10.jsonl and writes:

  1. a before/after cell table, overall and by quality grade
  2. an over-merging audit of the collapse
  3. data/dataset.jsonl — the record: collapsed cells, the collapsed J-lens
     concepts and their merges, with the raw top-10 tokens, raw cell and raw
     judge verdict kept alongside on every row for audit

Promoted 2026-09-03 after inspection; dataset_raw_top10.jsonl is kept so the
promotion is reproducible and the raw cells stay one `jq` away.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "data"
CELLS = ("self_report_only", "both", "lens_only", "neither")


def jl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    dataset = {r["concept"]: r for r in jl(HERE / "dataset_raw_top10.jsonl")}
    quality = {r["concept"]: r["quality"] for r in jl(HERE / "quality.jsonl")}
    collapsed = {r["concept"]: r for r in jl(HERE / "collapsed_jlens_l12.jsonl")}
    verdicts = {
        r["concept"]: r
        for r in jl(HERE / "method_cells_collapsed_jlens_l12.jsonl")
        if r["method"] == "j_lens"
    }
    keep = [c for c in dataset if c in collapsed and c in verdicts]
    print(f"{len(keep)}/{len(dataset)} items have a collapsed J-lens verdict\n")

    # 1. before / after
    def table(items: list[str], label: str) -> None:
        old = Counter(dataset[c]["cell"] for c in items)
        new = Counter(verdicts[c]["cell"] for c in items)
        print(f"{label} (n={len(items)})")
        print(f"  {'cell':<20}{'raw top-10':>12}{'collapsed':>12}{'delta':>8}")
        for cell in CELLS:
            delta = new[cell] - old[cell]
            print(f"  {cell:<20}{old[cell]:>12}{new[cell]:>12}{delta:>+8}")
        moved = sum(1 for c in items if dataset[c]["cell"] != verdicts[c]["cell"])
        print(f"  items whose cell changed: {moved} = {moved / len(items):.1%}\n")

    table(keep, "ALL ITEMS")
    for grade in ("strong", "medium", "weak"):
        table([c for c in keep if quality.get(c) == grade], f"QUALITY {grade}")

    # 2. over-merging audit of the new collapse
    absorbed, single, big, short = [], 0, 0, 0
    for c in keep:
        merges = collapsed[c]["merges"]["j_lens"]
        sizes = [len(m["tokens"]) for m in merges]
        absorbed += sizes
        single += len(merges) == 1
        big += any(s >= 12 for s in sizes)
        short += len(merges) < 10
    n = len(keep)
    print("OVER-MERGING AUDIT of the tightened-prompt collapse (J-lens, layer 12)")
    print(f"  tokens absorbed per concept: median {statistics.median(absorbed):.0f}, "
          f"p90 {sorted(absorbed)[int(len(absorbed) * .9)]}, max {max(absorbed)}")
    print(f"  lists with a concept absorbing >=12 tokens: {big}/{n} = {big / n:.1%}"
          f"   (previous run, strong items: 22.6%)")
    print(f"  lists collapsed to a single concept:        {single}/{n} = "
          f"{single / n:.1%}   (previous run: 49/3304 = 1.5%)")
    print(f"  lists with fewer than 10 concepts:          {short}/{n} = "
          f"{short / n:.1%}\n")

    # 3. the dataset with collapsed cells, alongside the raw for audit
    out = HERE / "dataset.jsonl"
    with out.open("w") as handle:
        for c in keep:
            row = dict(dataset[c])
            v = verdicts[c]
            row["lens_top_k_raw"] = row.pop("lens_top_k")
            row["lens_top_k"] = collapsed[c]["top_k"]["j_lens"]
            row["lens_merges"] = collapsed[c]["merges"]["j_lens"]
            row["cell_raw_top10"] = row["cell"]
            row["judge_raw_top10"] = row["judge"]
            row["in_lens"] = v["in_lens"]
            row["judge"] = v["judge"]
            row["cell"] = v["cell"]
            row["lens_budget"] = "top-50 collapsed to first 10 distinct concepts"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(keep)} rows); raw record in dataset_raw_top10.jsonl")


if __name__ == "__main__":
    main()
