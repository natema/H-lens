"""Collapse every method's top-50 readout into its first 10 distinct concepts.

Reads the deep lists written by ``eval_hlens.py --list-k 50`` and writes a file
in the same schema that ``judge_method_cells.py`` already accepts, with each
method's ``top_k`` replaced by the collapsed concept names. The merge record —
which tokens went into each concept — is kept alongside for audit.

Resumable per (concept, method).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from j2_lens.collapse import collapse_batch
from j2_lens.dataset import LEDGER_LOCK, append_jsonl, read_jsonl
from j2_lens.jspace import load_api_key
from j2_lens.spend import record_spend

HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lists", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=HERE / "pilot/spend.json")
    parser.add_argument("--methods", default="j_lens,j2_raw,j2_shuffled,r_lens")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    key = load_api_key(HERE / ".env")
    rows = read_jsonl(args.lists)
    done: dict[str, dict] = (
        {r["concept"]: r for r in read_jsonl(args.out)} if args.out.exists() else {}
    )

    jobs = []
    for row in rows:
        for method in methods:
            if method not in row["top_k"]:
                continue
            if done.get(row["concept"], {}).get("top_k", {}).get(method):
                continue
            jobs.append((row["concept"], method, row["top_k"][method]))
    print(f"{len(jobs)} lists to collapse across {len(methods)} methods", flush=True)
    if not jobs:
        return
    chunks = [jobs[i : i + args.batch] for i in range(0, len(jobs), args.batch)]

    def run(chunk):
        try:
            results, exchange = collapse_batch(
                key, [tokens for _, _, tokens in chunk], k=args.k
            )
        except Exception as error:  # noqa: BLE001 - one bad chunk must not stop the run
            print(f"  chunk failed: {error}", flush=True)
            return []
        with LEDGER_LOCK:
            record_spend(args.ledger, exchange, note=f"collapse x{len(chunk)}")
        return [
            (concept, method, tokens, verified)
            for (concept, method, tokens), verified in zip(chunk, results, strict=True)
        ]

    merged: dict[str, dict] = {c: r for c, r in done.items()}
    written = 0
    consecutive_failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for out in pool.map(run, chunks):
            # A systematic failure must stop the run, not be paid for chunk by
            # chunk: an earlier version spent through eight failing chunks.
            consecutive_failures = 0 if out else consecutive_failures + 1
            if consecutive_failures >= 3:
                raise SystemExit("three consecutive chunks failed; aborting")
            for concept, method, tokens, verified in out:
                entry = merged.setdefault(
                    concept,
                    {"concept": concept, "top_k": {}, "merges": {}, "raw_top_k": {}},
                )
                entry["top_k"][method] = [c["name"] for c in verified]
                entry["merges"][method] = verified
                entry["raw_top_k"][method] = tokens
                written += 1
            if written and written % 500 < args.batch * 4:
                print(f"  {written}/{len(jobs)}", flush=True)

    # carry the cell label through so downstream joins keep working
    cells = {r["concept"]: r.get("cell") for r in rows}
    for concept, entry in merged.items():
        entry["cell"] = cells.get(concept)
    args.out.write_text("")
    append_jsonl(args.out, list(merged.values()))

    complete = sum(
        1 for e in merged.values() if all(e["top_k"].get(m) for m in methods)
    )
    short = sum(
        1
        for e in merged.values()
        for m in methods
        if 0 < len(e["top_k"].get(m, [])) < args.k
    )
    print(f"wrote {args.out}: {complete}/{len(rows)} concepts fully collapsed; "
          f"{short} method-lists came back with fewer than {args.k} concepts")


if __name__ == "__main__":
    main()
