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
from pathlib import Path

import torch
from jlens import from_hf
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from j2_lens.baselines import MODEL_ID, MODEL_REVISION, load_lens
from j2_lens.jspace import (
    GENERATOR_MODEL,
    GENERATOR_SYSTEM,
    generate_items,
    lens_readout,
    load_api_key,
    record_spend,
    self_report_concepts,
    structural_problems,
    verify_causal_equivalence,
)

DEFAULT_CONCEPTS = [
    "basketball", "japan", "chess", "wedding", "volcano", "insomnia",
]
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts", default=",".join(DEFAULT_CONCEPTS))
    parser.add_argument("--top-k", type=int, default=10)
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

    raw_dir = HERE / "self_report_raw"
    raw_dir.mkdir(exist_ok=True)
    records = []
    for item in kept:
        readout = lens_readout(item, model, tokenizer, lenses, top_k=args.top_k)
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
            "j_lens_best_variant": readout["methods"]["j_lens"]["best_variant"],
            "j_lens_per_variant_rank":
                readout["methods"]["j_lens"]["per_variant_best_rank"],
            "n_prefix_tokens": readout["n_prefix_tokens"],
            "causal_equivalence_relative_dev": drift,
            "j_lens": {
                "best_rank": j["best_rank"],
                "best_layer": j["best_layer"],
                "top10_layers": j["top10_layers"],
                "in_top5": j["best_rank"] <= 5,
                "in_top10": j["best_rank"] <= 10,
            },
            "r_lens_best_rank": readout["methods"]["r_lens"]["best_rank"],
            "logit_lens_best_rank": readout["methods"]["logit_lens"]["best_rank"],
            "self_report": {
                "parsed": reported is not None,
                "concepts": reported,
                "rank": rank,
                "in_top5": rank is not None and rank <= 5,
                "in_top10": rank is not None,
            },
            "best_layer_top10_tokens": [
                t["decoded"]
                for t in j["layers"][str(j["best_layer"])]["top_tokens"]
            ],
        }
        record["agree_top10"] = record["j_lens"]["in_top10"] and record[
            "self_report"
        ]["in_top10"]
        record["agree_top5"] = record["j_lens"]["in_top5"] and record[
            "self_report"
        ]["in_top5"]
        records.append(record)
        status = (
            f"self-report #{rank}"
            if rank
            else ("miss" if reported is not None else "UNPARSED")
        )
        print(
            f"  {item.concept:<12} J-lens rank {j['best_rank']:>6}"
            f" @L{j['best_layer']:<3} | {status:<15}"
            f" | agree@10={record['agree_top10']}",
            flush=True,
        )

    (HERE / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    agree = sum(r["agree_top10"] for r in records)
    print(f"\n{agree}/{len(records)} items agree at top-10; wrote {HERE}")


if __name__ == "__main__":
    main()
