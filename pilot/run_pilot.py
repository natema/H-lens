"""Pilot run for the J-space dataset: generate items, then validate them twice.

Writes every intermediate artifact into this directory so the run can be
audited by hand:

    prompt_system.txt       system prompt sent to Mistral, verbatim
    prompt_user.txt         user prompt sent to Mistral, verbatim
    mistral_response.json   the request payload, raw JSON reply, and usage
    items.json              parsed items plus the derived probe prefixes
    results.json            both validation checks, per item, in full
    self_report_raw/        the model's untouched answer for each item

Usage:  uv run python pilot/run_pilot.py [--concepts a,b,c]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch
from jlens import from_hf
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from j2_lens.baselines import MODEL_ID, MODEL_REVISION, load_lens
from j2_lens.jspace import (
    GENERATOR_MODEL,
    GENERATOR_SYSTEM,
    JUDGE_BATCH_SIZE,
    PRIMARY_LAYER,
    annotate_fragments,
    generate_items,
    judge_batch,
    lens_readout,
    load_api_key,
    record_spend,
    self_report_concepts,
    structural_problems,
    verify_causal_equivalence,
    vocabulary_words,
)

DEFAULT_CONCEPTS = [
    "basketball", "japan", "chess", "wedding", "volcano", "insomnia",
]
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts", default=",".join(DEFAULT_CONCEPTS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument("--judge-batch", type=int, default=JUDGE_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]

    key = load_api_key(Path(__file__).resolve().parents[2] / ".env")
    print(f"Generating {len(concepts)} items with {GENERATOR_MODEL} ...", flush=True)
    items, exchange = generate_items(key, concepts)
    totals = record_spend(
        HERE / "spend.json",
        exchange,
        note=f"pilot generation: {len(concepts)} concepts",
    )
    print(
        f"  Mistral spend: this call "
        f"${exchange['usage']['prompt_tokens']}in/"
        f"{exchange['usage']['completion_tokens']}out"
        f" | cumulative ${totals['cost_usd']:.4f} of ~{totals['budget_eur']:.0f} EUR"
        f" over {totals['n_calls']} calls",
        flush=True,
    )

    (HERE / "prompt_system.txt").write_text(GENERATOR_SYSTEM + "\n")
    (HERE / "prompt_user.txt").write_text(
        exchange["request"]["messages"][1]["content"] + "\n"
    )
    (HERE / "mistral_response.json").write_text(
        json.dumps(
            {
                "model": exchange["model"],
                "usage": exchange["usage"],
                "request": exchange["request"],
                "raw_content": exchange["content"],
            },
            indent=2,
        )
        + "\n"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    screened = []
    for item in items:
        problems = structural_problems(item, tokenizer)
        screened.append(
            {
                "concept": item.concept,
                "probe_term": item.probe_term,
                "sentence": item.sentence,
                "prefix": item.prefix,
                "suffix_discarded": item.sentence[len(item.prefix) :],
                "rationale": item.rationale,
                "structural_problems": problems,
                "accepted": not problems,
            }
        )
        if problems:
            print(f"  REJECT {item.concept!r}: {'; '.join(problems)}", flush=True)
    (HERE / "items.json").write_text(json.dumps(screened, indent=2) + "\n")

    kept = [i for i, s in zip(items, screened, strict=True) if s["accepted"]]
    print(f"{len(kept)}/{len(items)} pass the structural screen\n", flush=True)
    if not kept:
        return

    hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, local_files_only=True
    ).to(args.device)
    model = from_hf(hf_model, tokenizer, compile=False, force_bos=True)
    lenses = {name: load_lens(name, True)[0] for name in ("j_lens", "r_lens")}
    vocabulary = vocabulary_words(tokenizer)

    raw_dir = HERE / "self_report_raw"
    raw_dir.mkdir(exist_ok=True)
    records = []
    pending = []
    for item in kept:
        readout = lens_readout(
            item, model, tokenizer, lenses, top_k=args.top_k,
            primary_layer=args.layer,
        )
        drift = verify_causal_equivalence(item, model, tokenizer, lenses["j_lens"])
        reported, raw = self_report_concepts(
            hf_model,
            tokenizer,
            item,
            limit=args.top_k,
            max_new_tokens=args.max_new_tokens,
        )
        (raw_dir / f"{item.concept}.txt").write_text(raw)

        j = readout["methods"]["j_lens"]
        primary = readout["primary"]
        rank = (
            reported.index(item.concept.lower()) + 1
            if reported and item.concept.lower() in reported
            else None
        )
        record = {
            "concept": item.concept,
            "probe_term": item.probe_term,
            "prefix": item.prefix,
            "rationale": item.rationale,
            "probe_token": readout["probe_token"]["decoded"],
            "concept_variants": readout["concept_variants"],
            "n_prefix_tokens": readout["n_prefix_tokens"],
            "causal_equivalence_relative_dev": drift,
            "primary_layer": readout["primary_layer"],
            "primary": primary,
            "j_lens_best_rank_any_layer": j["best_rank"],
            "j_lens_best_layer": j["best_layer"],
            "self_report": {
                "parsed": reported is not None,
                "concepts": reported,
                "exact_rank": rank,
            },
        }
        records.append(record)
        pending.append(
            {
                "concept": item.concept,
                "lens_tokens": primary["j_lens"]["top_tokens"],
                "self_report": reported,
                "fragment_evidence": annotate_fragments(
                    primary["j_lens"]["top_tokens"], vocabulary
                ),
            }
        )

    # One adjudication call for the whole batch.
    verdicts = []
    for start in range(0, len(pending), args.judge_batch):
        chunk = pending[start : start + args.judge_batch]
        got, _ = judge_batch(key, chunk, ledger=HERE / "spend.json")
        verdicts.extend(got)
    for record, verdict in zip(records, verdicts, strict=True):
        record["judge"] = verdict
        in_a = bool(verdict.get("lens", {}).get("present"))
        in_b = bool(verdict.get("self_report", {}).get("present"))
        record["in_lens"] = in_a
        record["in_self_report"] = in_b
        # All four cells are recorded and nothing is dropped. Keeping only
        # agreement would retain exactly the items the J-lens already handles,
        # leaving no headroom and biasing the comparison in its favour.
        record["cell"] = (
            "both" if in_a and in_b
            else "self_report_only" if in_b
            else "lens_only" if in_a
            else "neither"
        )
        print(
            f"  {record['concept']:<12} L{record['primary_layer']}"
            f" rank {record['primary']['j_lens']['target_rank']:>6}"
            f" | lens={in_a!s:<5} self={in_b!s:<5} -> {record['cell']}",
            flush=True,
        )

    (HERE / "results.json").write_text(json.dumps(records, indent=2) + "\n")

    # Append to a durable pool. Each run regenerates different items, so
    # without this the only surviving evidence is the latest run, and judge
    # fixtures would have to be invented rather than observed.
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    with (HERE / "observations.jsonl").open("a") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "generator": GENERATOR_MODEL,
                        "concept": record["concept"],
                        "probe_term": record["probe_term"],
                        "prefix": record["prefix"],
                        "primary_layer": record["primary_layer"],
                        "lens_top10": record["primary"]["j_lens"]["top_tokens"],
                        "j_lens_rank_primary": record["primary"]["j_lens"][
                            "target_rank"
                        ],
                        "r_lens_rank_primary": record["primary"]["r_lens"][
                            "target_rank"
                        ],
                        "self_report": record["self_report"]["concepts"],
                        "judge": record["judge"],
                        "cell": record["cell"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    cells = Counter(r["cell"] for r in records)
    totals = json.loads((HERE / "spend.json").read_text())["totals"]
    print(
        "\ncells: "
        + ", ".join(
            f"{name}={cells[name]}"
            for name in ("both", "self_report_only", "lens_only", "neither")
        )
        + f"   (self_report_only = J-lens failures: {cells['self_report_only']})"
    )
    print(
        f"Mistral spend: ${totals['cost_usd']:.4f} over {totals['n_calls']} calls"
    )
    print(f"wrote {HERE}")


if __name__ == "__main__":
    main()
