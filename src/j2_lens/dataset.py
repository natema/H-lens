"""Build the J-space dataset over the vocabulary-derived concept list.

Three phases, each resumable from its own JSONL, so a crash or an interrupt
costs only the work in flight:

1. **generate** — one item per concept from the generator model (API).
2. **read** — J/R/logit-lens readout at the primary layer, and the model's own
   self-report on the same prefix (GPU).
3. **judge** — adjudication of both lists (API).

All four outcome cells are recorded and nothing is filtered on agreement.
Keeping only agreement would retain exactly the items the J-lens already
handles, leaving no headroom and biasing any later comparison in its favour.

Self-report generation is batched for throughput. An item's output then depends
on which items share its batch, through padding width and reduction order, so
the batch size and the item ordering are both recorded. Reproducing the file
requires both to match; two passes at the same batch size agreed item for item
when measured, but that was checked at batch 1 only, so treat identical settings
as necessary rather than proven sufficient.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LEDGER_LOCK = threading.Lock()


def load_done(path: Path, key: str = "concept") -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)[key])
    return done


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def phase_generate(
    api_key: str,
    concepts: list[str],
    out: Path,
    ledger: Path,
    *,
    per_call: int,
    workers: int,
) -> None:
    from transformers import AutoTokenizer

    from j2_lens.baselines import MODEL_ID, MODEL_REVISION
    from j2_lens.jspace import generate_items, structural_problems
    from j2_lens.spend import record_spend

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )

    done = load_done(out)
    todo = [c for c in concepts if c not in done]
    print(f"[generate] {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    chunks = [todo[i : i + per_call] for i in range(0, len(todo), per_call)]

    def run(chunk: list[str]) -> list[dict[str, Any]]:
        try:
            items, exchange = generate_items(api_key, chunk)
        except Exception as error:  # noqa: BLE001 - one bad chunk must not stop the run
            print(f"[generate] chunk failed: {error}", flush=True)
            return []
        with LEDGER_LOCK:
            record_spend(ledger, exchange, note=f"generate x{len(chunk)}")
        # Only conforming items are written. A concept whose item fails the
        # screen stays absent from the file, so a later generate pass retries
        # it instead of banking a fragment whose evocative words get discarded.
        wanted = set(chunk)
        rows = []
        for item in items:
            if item.concept not in wanted:
                continue
            if structural_problems(item, tokenizer):
                continue
            rows.append({
                "concept": item.concept,
                "probe_term": item.probe_term,
                "sentence": item.sentence,
                "rationale": item.rationale,
            })
        return rows

    written = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(run, chunks):
            if rows:
                append_jsonl(out, rows)
                written += len(rows)
                print(f"[generate] {written}/{len(todo)}", flush=True)


def phase_read(
    items: list[dict[str, Any]],
    out: Path,
    *,
    layer: int,
    top_k: int,
    batch: int,
    max_new_tokens: int,
    device: str,
    dtype: str = "bfloat16",
    reuse_self_report: Path | None = None,
) -> None:
    import torch
    from jlens import from_hf
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    from j2_lens.baselines import MODEL_ID, MODEL_REVISION, load_lens
    from j2_lens.jspace import (
        SELF_REPORT_TEMPLATE,
        JSpaceItem,
        lens_readout,
        parse_concept_list,
        structural_problems,
    )

    done = load_done(out)
    todo = [row for row in items if row["concept"] not in done]
    # Reusing previously generated self-reports keeps the model dtype as the
    # only variable that changed, so a difference in the result is attributable
    # to precision rather than to regenerated text.
    cached: dict[str, Any] = {}
    if reuse_self_report is not None:
        for row in read_jsonl(reuse_self_report):
            if row.get("screened"):
                cached[row["concept"]] = {
                    "self_report": row.get("self_report"),
                    "self_report_raw": row.get("self_report_raw", ""),
                }
        print(f"[read] reusing {len(cached)} cached self-reports", flush=True)
    print(f"[read] {len(done)} done, {len(todo)} to do, dtype={dtype}", flush=True)
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch_dtype, local_files_only=True
    ).to(device)
    model = from_hf(hf_model, tokenizer, compile=False, force_bos=True)
    lenses = {name: load_lens(name, True)[0] for name in ("j_lens", "r_lens")}

    for start in range(0, len(todo), batch):
        chunk = todo[start : start + batch]
        prepared: list[tuple[dict[str, Any], JSpaceItem, dict[str, Any]]] = []
        rejected: list[dict[str, Any]] = []
        for row in chunk:
            item = JSpaceItem(
                row["concept"], row["probe_term"], row["sentence"],
                row.get("rationale", ""),
            )
            problems = structural_problems(item, tokenizer)
            if problems:
                rejected.append({**row, "screen_problems": problems, "screened": False})
                continue
            try:
                readout = lens_readout(
                    item, model, tokenizer, lenses, top_k=top_k, primary_layer=layer
                )
            except Exception as error:  # noqa: BLE001
                rejected.append(
                    {**row, "screen_problems": [str(error)], "screened": False}
                )
                continue
            prepared.append((row, item, readout))

        missing = [x for x in prepared if x[0]["concept"] not in cached]
        if prepared and not missing:
            answers = [
                cached[row["concept"]]["self_report_raw"] for row, _, _ in prepared
            ]
        elif prepared:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": SELF_REPORT_TEMPLATE.format(
                        term=item.probe_term, prefix=item.prefix, limit=top_k)}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                for _, item, _ in prepared
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                generated = hf_model.generate(
                    **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            width = encoded["input_ids"].shape[1]
            answers = [
                tokenizer.decode(generated[i, width:], skip_special_tokens=True)
                for i in range(len(prompts))
            ]
        else:
            answers = []

        rows = list(rejected)
        for (row, item, readout), answer in zip(prepared, answers, strict=True):
            entry = cached.get(row["concept"])
            reported = (
                entry["self_report"] if entry else parse_concept_list(answer, top_k)
            )
            rows.append({
                **row,
                "screened": True,
                "prefix": item.prefix,
                "probe_token": readout["probe_token"]["decoded"],
                "n_prefix_tokens": readout["n_prefix_tokens"],
                "primary_layer": readout["primary_layer"],
                "lens_top_k": readout["primary"]["j_lens"]["top_tokens"],
                "j_lens_rank": readout["primary"]["j_lens"]["target_rank"],
                "j_lens_margin_to_kth": readout["primary"]["j_lens"]["margin_to_kth"],
                "j_lens_variant": readout["primary"]["j_lens"]["variant"],
                "r_lens_rank": readout["primary"]["r_lens"]["target_rank"],
                "logit_lens_rank": readout["primary"]["logit_lens"]["target_rank"],
                "j_lens_best_rank_any_layer": readout["methods"]["j_lens"]["best_rank"],
                "j_lens_best_layer": readout["methods"]["j_lens"]["best_layer"],
                "self_report": reported,
                "self_report_raw": answer[:400],
            })
        append_jsonl(out, rows)
        print(f"[read] {min(start + batch, len(todo))}/{len(todo)}", flush=True)


def phase_judge(
    api_key: str,
    rows: list[dict[str, Any]],
    out: Path,
    ledger: Path,
    *,
    batch: int,
    workers: int,
) -> None:
    from transformers import AutoTokenizer

    from j2_lens.baselines import MODEL_ID, MODEL_REVISION
    from j2_lens.jspace import annotate_fragments, judge_batch, vocabulary_words
    from j2_lens.spend import record_spend

    done = load_done(out)
    todo = [r for r in rows if r.get("screened") and r["concept"] not in done]
    print(f"[judge] {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    vocabulary = vocabulary_words(tokenizer)
    chunks = [todo[i : i + batch] for i in range(0, len(todo), batch)]

    def run(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cases = [
            {
                "concept": r["concept"],
                "lens_tokens": r["lens_top_k"],
                "self_report": r["self_report"],
                "fragment_evidence": annotate_fragments(r["lens_top_k"], vocabulary),
            }
            for r in chunk
        ]
        try:
            verdicts, exchange = judge_batch(api_key, cases)
        except Exception as error:  # noqa: BLE001
            print(f"[judge] chunk failed: {error}", flush=True)
            return []
        with LEDGER_LOCK:
            record_spend(ledger, exchange, note=f"judge x{len(chunk)}")
        out_rows = []
        for row, verdict in zip(chunk, verdicts, strict=True):
            in_a = bool(verdict.get("lens", {}).get("present"))
            in_b = bool(verdict.get("self_report", {}).get("present"))
            out_rows.append({
                **row,
                "judge": verdict,
                "in_lens": in_a,
                "in_self_report": in_b,
                "cell": ("both" if in_a and in_b else "self_report_only" if in_b
                         else "lens_only" if in_a else "neither"),
            })
        return out_rows

    written = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for produced in pool.map(run, chunks):
            if produced:
                append_jsonl(out, produced)
                written += len(produced)
                print(f"[judge] {written}/{len(todo)}", flush=True)


def main(argv: list[str] | None = None) -> None:
    from j2_lens.jspace import PRIMARY_LAYER, load_api_key

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts", type=Path, default=root / "configs/concepts.json")
    parser.add_argument("--out-dir", type=Path, default=root / "data")
    parser.add_argument("--ledger", type=Path, default=root / "pilot/spend.json")
    parser.add_argument("--layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, help="only the first N concepts")
    parser.add_argument("--generate-per-call", type=int, default=10)
    parser.add_argument("--generate-passes", type=int, default=8)
    parser.add_argument("--read-batch", type=int, default=32)
    parser.add_argument("--judge-batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--reuse-self-report", type=Path,
                        help="take self-reports from an earlier readouts.jsonl")
    parser.add_argument(
        "--phase", choices=("all", "generate", "read", "judge"), default="all"
    )
    args = parser.parse_args(argv)

    key = load_api_key(root / ".env")
    concepts = [c["word"] for c in json.loads(args.concepts.read_text())["concepts"]]
    if args.limit:
        concepts = concepts[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    items_path = args.out_dir / "items.jsonl"
    reads_path = args.out_dir / "readouts.jsonl"
    final_path = args.out_dir / "dataset.jsonl"

    print(f"{len(concepts)} concepts, layer {args.layer}, "
          f"read batch {args.read_batch}, judge batch {args.judge_batch}", flush=True)

    if args.phase in ("all", "generate"):
        # Keep going while passes still recover concepts. A fixed pass count
        # stopped a converging loop early: 3344 -> 624 -> 139 -> 42 left, which
        # looked like 42 impossible concepts but was just the limit being hit.
        previous = None
        for attempt in range(args.generate_passes):
            remaining = len(concepts) - len(load_done(items_path))
            if not remaining:
                break
            if previous is not None and remaining >= previous:
                print(f"[generate] no progress, stopping with {remaining} left",
                      flush=True)
                break
            previous = remaining
            print(
                f"[generate] pass {attempt + 1}, {remaining} concepts left",
                flush=True,
            )
            phase_generate(key, concepts, items_path, args.ledger,
                           per_call=args.generate_per_call, workers=args.workers)
    if args.phase in ("all", "read"):
        wanted = set(concepts)
        items = [r for r in read_jsonl(items_path) if r["concept"] in wanted]
        phase_read(items, reads_path, layer=args.layer, top_k=args.top_k,
                   batch=args.read_batch, max_new_tokens=args.max_new_tokens,
                   device=args.device, dtype=args.dtype,
                   reuse_self_report=args.reuse_self_report)
    if args.phase in ("all", "judge"):
        phase_judge(key, read_jsonl(reads_path), final_path, args.ledger,
                    batch=args.judge_batch, workers=args.workers)

    rows = read_jsonl(final_path)
    cells: dict[str, int] = {}
    for row in rows:
        cells[row["cell"]] = cells.get(row["cell"], 0) + 1
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "created_at": datetime.now(UTC).isoformat(),
        "n_concepts": len(concepts),
        "n_items": len(rows),
        "primary_layer": args.layer,
        "top_k": args.top_k,
        "read_batch": args.read_batch,
        "dtype": args.dtype,
        "judge_batch": args.judge_batch,
        "reproducibility": (
            "Self-report generation is batched, so an item's output depends on "
            "which items share its batch. Rerunning with this read_batch and "
            "the concept ordering in items.jsonl reproduces the file exactly."
        ),
        "cells": cells,
    }, indent=2) + "\n")
    print(f"\ncells: {cells}\nwrote {final_path}")


if __name__ == "__main__":
    main()
