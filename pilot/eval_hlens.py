"""Evaluate every lens on the whole J-space dataset, all cells included.

Restricting the comparison to the `self_report_only` cell would be selection on
the outcome: that cell is *defined* by the J-lens failing, so J-lens is
conditioned to be at its worst there and regression to the mean would flatter
any alternative, including a worse one. The cells where the J-lens does well are
exactly where a correction can be caught doing damage, so every item is scored
and the results are broken down by cell and by item quality afterwards.

Applies, at the probe position and the fitted layer:
  logit lens, J-lens, J2 (real diagonal), J2 (coordinate-shuffled control), R-lens
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from jlens import from_hf
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from j2_lens.baselines import (
    MODEL_ID,
    MODEL_REVISION,
    describe_tokens,
    load_lens,
    rank_and_topk,
)
from j2_lens.dataset import append_jsonl, load_done, read_jsonl
from j2_lens.evaluation import capture_activations, residual_methods
from j2_lens.jspace import single_token_variants

HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=HERE / "data/dataset.jsonl"
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=HERE / "results/hessian_pile_l12_merged_qwen3.5-4b.pt",
    )
    parser.add_argument("--out", type=Path, default=HERE / "data/hlens.jsonl")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--list-k", type=int, default=None,
        help="how many readout tokens to store per method (default: top-k). A "
        "deeper list lets a later pass collapse variants to distinct concepts.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    list_k = args.list_k or args.top_k

    rows = read_jsonl(args.dataset)
    done = load_done(args.out)
    todo = [r for r in rows if r["concept"] not in done]
    print(f"{len(done)} done, {len(todo)} to do at layer {args.layer}", flush=True)
    if not todo:
        return

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    entry = artifact["layers"][args.layer]
    stats = {
        "source_mean": entry["source_mean"],
        "source_variance": entry["source_variance"],
    }
    operator = {
        "coordinates": entry["coordinates"],
        "diagonal_rows": entry["diagonal_rows"],
    }
    print(
        f"operator fitted on {artifact['metadata'].get('n_hessian_pairs')} Hessian "
        f"pairs / {artifact['metadata'].get('n_moment_pairs')} moment pairs "
        f"of {artifact['metadata']['development_corpus']['repo_id']}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32, local_files_only=True
    ).to(args.device)
    model = from_hf(hf_model, tokenizer, compile=False, force_bos=True)
    lenses = {name: load_lens(name, True)[0] for name in ("j_lens", "r_lens")}

    buffer: list[dict] = []
    for index, row in enumerate(todo, 1):
        input_ids = model.encode(row["prefix"], max_length=512)
        position = input_ids.shape[1] - 1
        activations = capture_activations(model, input_ids, [args.layer])
        source = activations[args.layer][0, position]
        methods = residual_methods(
            source,
            stats,
            operator,
            lenses["j_lens"].jacobians[args.layer],
            lenses["r_lens"].jacobians[args.layer],
        )
        variants = single_token_variants(row["concept"], tokenizer)
        scored: dict[str, int] = {}
        # The cells are defined by a judge reading the top-10 list, not by the
        # rank of one token, so the list has to be stored for every method or
        # the same cell assignment cannot be applied to them.
        listed: dict[str, list[str]] = {}
        for name, residual in methods.items():
            logits = model.unembed(residual[None]).float()[0]
            best = min(
                (
                    rank_and_topk(logits, token_id, list_k)
                    for _, token_id in variants
                ),
                key=lambda result: result[0],
            )
            scored[name] = best[0]
            listed[name] = [
                token["decoded"] for token in describe_tokens(tokenizer, best[2])
            ]
        buffer.append({
            "concept": row["concept"],
            "cell": row["cell"],
            "ranks": scored,
            "top_k": listed,
        })
        if len(buffer) >= 64 or index == len(todo):
            append_jsonl(args.out, buffer)
            buffer = []
            print(f"  {index}/{len(todo)}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
