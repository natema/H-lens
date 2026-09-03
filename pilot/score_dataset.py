"""Grade every item's evocation strength and write a joinable quality file.

The grade depends only on the fragment and the target concept, so it applies to
any readout of the same items. It is written to its own file keyed by concept
rather than merged into a dataset, so it joins cleanly onto both the bfloat16
draft and the float32 rebuild.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from j2_lens.dataset import LEDGER_LOCK, append_jsonl, load_done, read_jsonl
from j2_lens.jspace import load_api_key
from j2_lens.scoring import score_batch
from j2_lens.spend import record_spend

HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readouts", type=Path, default=HERE / "data/readouts.jsonl")
    parser.add_argument("--out", type=Path, default=HERE / "data/quality.jsonl")
    parser.add_argument("--ledger", type=Path, default=HERE / "pilot/spend.json")
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    key = load_api_key(HERE / ".env")
    source = {
        r["concept"]: {"prefix": r["prefix"], "probe_term": r["probe_term"]}
        for r in read_jsonl(args.readouts)
        if r.get("screened")
    }
    done = load_done(args.out)
    todo = [
        {"concept": c, **fields} for c, fields in source.items() if c not in done
    ]
    print(f"{len(done)} graded, {len(todo)} to do", flush=True)
    if not todo:
        return
    chunks = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]

    def run(chunk: list[dict]) -> list[dict]:
        try:
            verdicts, exchange = score_batch(key, chunk)
        except Exception as error:  # noqa: BLE001 - a bad chunk must not stop the run
            print(f"  chunk failed: {error}", flush=True)
            return []
        with LEDGER_LOCK:
            record_spend(args.ledger, exchange, note=f"quality score x{len(chunk)}")
        return [
            {
                "concept": item["concept"],
                "probe_term": item["probe_term"],
                "quality": v["strength"],
                "why": v["why"],
            }
            for item, v in zip(chunk, verdicts, strict=True)
            if v["strength"] is not None
        ]

    written = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rows in pool.map(run, chunks):
            if rows:
                with LEDGER_LOCK:
                    append_jsonl(args.out, rows)
                written += len(rows)
                if written % 240 < args.batch:
                    print(f"  {written}/{len(todo)}", flush=True)

    counts = Counter(r["quality"] for r in read_jsonl(args.out))
    print(f"\n{dict(counts)}  -> {args.out}")


if __name__ == "__main__":
    main()
