"""Compare cell counts judged on raw top-10 lists versus collapsed top-10.

The raw top-10 of a lens holds ~7.8 distinct concepts (R-lens 8.2) against the
self-report's 10, so the budget is unequal. Collapsing each lens's top-50 to its
first 10 distinct concepts equalises it. This prints, for each layer and method,
the four cells under both judgings on the same items, so the size of the
budget effect and its direction per lens can be read directly.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "data"
CELLS = ("self_report_only", "both", "lens_only", "neither")
METHODS = ("j_lens", "j2_raw", "j2_shuffled", "r_lens")


def load_cells(path: Path) -> dict[str, dict[str, str]]:
    by: dict[str, dict[str, str]] = {}
    if not path.exists():
        return by
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by.setdefault(r["method"], {})[r["concept"]] = r["cell"]
    return by


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", default="fixed",
        help="which collapsed judging to compare: 'fixed' (mechanically un-merged, "
        "the default) or '' for the original over-merged run",
    )
    args = parser.parse_args()
    tag = f"_{args.variant}" if args.variant else ""
    strong = {
        json.loads(line)["concept"]
        for line in (HERE / "quality.jsonl").read_text().splitlines()
        if line.strip() and json.loads(line)["quality"] == "strong"
    }
    dataset = {
        json.loads(line)["concept"]: json.loads(line)["cell"]
        # dataset.jsonl now carries the collapsed J-lens cells; the raw
        # top-10 cells this compares against are the pre-promotion record.
        for line in (HERE / "dataset_raw_top10.jsonl").read_text().splitlines()
        if line.strip()
    }
    for layer in (12, 6):
        raw = load_cells(HERE / f"method_cells{'' if layer == 12 else '_l6'}.jsonl")
        if layer == 12:
            raw["j_lens"] = dataset
        col = load_cells(HERE / f"method_cells_collapsed{tag}_strong_l{layer}.jsonl")
        if not col:
            print(f"layer {layer}: no collapsed cells yet")
            continue
        keep = [
            c for c in strong
            if all(c in raw.get(m, {}) and c in col.get(m, {}) for m in METHODS)
        ]
        print(f"\nLAYER {layer}, strong items (n={len(keep)})")
        print(f"  {'method':<12}{'':<10}" + "".join(f"{c:>18}" for c in CELLS))
        for m in METHODS:
            for label, by in (("raw top-10", raw), ("collapsed", col)):
                cnt = Counter(by[m][c] for c in keep)
                print(f"  {m:<12}{label:<10}" + "".join(f"{cnt[c]:>18}" for c in CELLS))
            r = Counter(raw[m][c] for c in keep)
            k = Counter(col[m][c] for c in keep)
            delta = "".join(f"{k[c] - r[c]:>+18}" for c in CELLS)
            print(f"  {'':<12}{'delta':<10}" + delta)
        jf = Counter(col["j_lens"][c] for c in keep)["self_report_only"]
        print(f"  -> collapsed, failures vs J-lens ({jf}): " + "  ".join(
            f"{m} {Counter(col[m][c] for c in keep)['self_report_only'] - jf:+d}"
            for m in METHODS[1:]
        ))
        moved = sum(1 for m in METHODS for c in keep if raw[m][c] != col[m][c])
        print(f"  cells that changed under collapse: {moved}/{len(keep) * len(METHODS)}"
              f" = {moved / (len(keep) * len(METHODS)):.1%}")


if __name__ == "__main__":
    main()
