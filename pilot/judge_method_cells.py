"""Assign the four cells to each lens, using the same judge as the dataset.

The cells are defined by a judge deciding whether a top-k readout *names* the
concept, which counts variants a raw token rank misses (" trail" for path). So
comparing methods by token rank answers a different question from the one the
dataset asks. This applies the identical procedure to every method: same judge,
same prompt, same vocabulary evidence, same self-report, only the readout list
differs.

The self-report side is unchanged by construction — it does not depend on any
lens — so a change in the cells is entirely a change in what the lens surfaced.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from j2_lens.baselines import MODEL_ID, MODEL_REVISION
from j2_lens.dataset import LEDGER_LOCK, append_jsonl, read_jsonl
from j2_lens.jspace import (
    annotate_fragments,
    judge_batch,
    load_api_key,
    record_spend,
    vocabulary_words,
)

HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    d = HERE / "data_fp32"
    parser.add_argument("--lists", type=Path, default=d / "hlens_lists.jsonl")
    parser.add_argument("--dataset", type=Path, default=d / "dataset.jsonl")
    parser.add_argument("--out", type=Path, default=d / "method_cells.jsonl")
    parser.add_argument("--ledger", type=Path, default=HERE / "pilot/spend.json")
    parser.add_argument(
        "--methods", default="j2_raw,j2_shuffled,r_lens",
        help="comma-separated; j_lens cells already exist in the dataset",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    key = load_api_key(HERE.parent / ".env")
    dataset = {r["concept"]: r for r in read_jsonl(args.dataset)}
    lists = read_jsonl(args.lists)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    vocabulary = vocabulary_words(tokenizer)

    done = {json.loads(line)["key"] for line in args.out.read_text().splitlines()
            if line.strip()} if args.out.exists() else set()

    jobs = []
    for row in lists:
        concept = row["concept"]
        if concept not in dataset:
            continue
        for method in methods:
            key_id = f"{method}|{concept}"
            if key_id in done or method not in row["top_k"]:
                continue
            jobs.append({
                "key": key_id,
                "method": method,
                "concept": concept,
                "lens_tokens": row["top_k"][method],
                "self_report": dataset[concept]["self_report"],
                "in_self_report": dataset[concept]["in_self_report"],
                "fragment_evidence": annotate_fragments(
                    row["top_k"][method], vocabulary
                ),
            })
    print(f"{len(done)} done, {len(jobs)} judgements to make", flush=True)
    if not jobs:
        return
    chunks = [jobs[i : i + args.batch] for i in range(0, len(jobs), args.batch)]

    def run(chunk: list[dict]) -> list[dict]:
        try:
            verdicts, exchange = judge_batch(key, chunk)
        except Exception as error:  # noqa: BLE001
            print(f"  chunk failed: {error}", flush=True)
            return []
        with LEDGER_LOCK:
            record_spend(args.ledger, exchange, note=f"method cells x{len(chunk)}")
        out = []
        for job, verdict in zip(chunk, verdicts, strict=True):
            in_a = bool(verdict.get("lens", {}).get("present"))
            in_b = bool(job["in_self_report"])
            out.append({
                "key": job["key"],
                "method": job["method"],
                "concept": job["concept"],
                "in_lens": in_a,
                "in_self_report": in_b,
                "cell": ("both" if in_a and in_b else "self_report_only" if in_b
                         else "lens_only" if in_a else "neither"),
                "judge": verdict,
            })
        return out

    written = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rows in pool.map(run, chunks):
            if rows:
                with LEDGER_LOCK:
                    append_jsonl(args.out, rows)
                written += len(rows)
                if written % 400 < args.batch:
                    print(f"  {written}/{len(jobs)}", flush=True)

    got = read_jsonl(args.out)
    for method in methods:
        c = Counter(r["cell"] for r in got if r["method"] == method)
        print(f"  {method:<13} {dict(c)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
