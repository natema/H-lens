"""Run matched logit-, J-, and R-lens baselines on Qwen3.5-4B."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from huggingface_hub import hf_hub_download
from jlens import JacobianLens, from_hf
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LENS_REPO_ID = "camilablank/workspace-lenses"
LENS_REVISION = "781b23318040be06dba7546d254742c6fa623098"
JLENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
LENS_FILES = {
    "j_lens": "qwen3.5-4b/j-lens/lens.pt",
    "r_lens": "qwen3.5-4b/r-lens/lens.pt",
}
LENS_SHA256 = {
    "j_lens": "8ac16d8e5e988f19d9e19ac9b459c9e73336a866c8534d521200e4fcf2c21377",
    "r_lens": "848b9b8e76eb0448a926edc8bb374cb4369916dd6bdbd62f1d3141fa12b9f231",
}


@dataclass(frozen=True)
class BaselineCase:
    id: str
    category: str
    prompt: str
    probe_text: str
    probe_occurrence: int
    probe_token_within_span: int
    target_text: str
    reference_model: str
    reference_note: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def nth_span(text: str, needle: str, occurrence: int) -> tuple[int, int]:
    if occurrence < 0:
        raise ValueError("probe_occurrence must be non-negative")
    start = -1
    search_from = 0
    for _ in range(occurrence + 1):
        start = text.find(needle, search_from)
        if start < 0:
            raise ValueError(f"occurrence {occurrence} of {needle!r} not found")
        search_from = start + len(needle)
    return start, start + len(needle)


def token_indices_overlapping(
    offsets: list[tuple[int, int]], char_span: tuple[int, int]
) -> list[int]:
    char_start, char_end = char_span
    indices = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start
        and token_end > char_start
        and token_start < char_end
    ]
    if not indices:
        raise ValueError(f"no token overlaps character span {char_span}")
    return indices


def resolve_probe(
    prompt: str,
    probe_text: str,
    occurrence: int,
    token_within_span: int,
    offsets: list[tuple[int, int]],
) -> tuple[tuple[int, int], list[int], int]:
    char_span = nth_span(prompt, probe_text, occurrence)
    span_tokens = token_indices_overlapping(offsets, char_span)
    try:
        position = span_tokens[token_within_span]
    except IndexError as error:
        raise ValueError(
            f"token selector {token_within_span} invalid for indices {span_tokens}"
        ) from error
    return char_span, span_tokens, position


def rank_and_topk(
    logits: torch.Tensor, target_id: int, top_k: int
) -> tuple[int, float, list[int], list[float]]:
    if logits.ndim != 1:
        raise ValueError(f"expected rank-1 logits, got {tuple(logits.shape)}")
    target_logit = logits[target_id]
    rank = int(torch.count_nonzero(logits > target_logit).item()) + 1
    values, indices = torch.topk(logits, min(top_k, logits.numel()))
    return rank, float(target_logit), indices.tolist(), values.tolist()


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def load_cases(path: Path, selected: set[str]) -> list[BaselineCase]:
    cases = [
        BaselineCase(**item)
        for item in json.loads(path.read_text())["cases"]
    ]
    unknown = selected - {case.id for case in cases}
    if unknown:
        raise ValueError(f"unknown case IDs: {sorted(unknown)}")
    return [case for case in cases if not selected or case.id in selected]


def load_lens(method: str, offline: bool) -> tuple[JacobianLens, dict[str, Any]]:
    filename = LENS_FILES[method]
    path = Path(
        hf_hub_download(
            LENS_REPO_ID,
            filename,
            revision=LENS_REVISION,
            local_files_only=offline,
        )
    )
    digest = sha256_file(path)
    if digest != LENS_SHA256[method]:
        raise RuntimeError(f"{method} SHA-256 mismatch: {digest}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    raw_j = checkpoint["J"]
    if isinstance(raw_j, torch.Tensor):
        source_layers = [int(layer) for layer in checkpoint["source_layers"]]
        jacobians = dict(zip(source_layers, raw_j, strict=True))
    else:
        jacobians = {int(layer): matrix for layer, matrix in raw_j.items()}
    lens = JacobianLens(
        jacobians,
        n_prompts=int(checkpoint["n_prompts"]),
        d_model=int(checkpoint["d_model"]),
    )
    metadata = {
        "filename": filename,
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "n_prompts": lens.n_prompts,
        "d_model": lens.d_model,
        "source_layers": lens.source_layers,
        "provenance": jsonable(checkpoint.get("provenance", {})),
    }
    return lens, metadata


def check_lens_pair(
    lenses: dict[str, JacobianLens], metadata: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    j_lens, r_lens = lenses["j_lens"], lenses["r_lens"]
    if (
        j_lens.source_layers != r_lens.source_layers
        or j_lens.source_layers != list(range(31))
    ):
        raise RuntimeError("unexpected or mismatched source layers")
    if {j_lens.d_model, r_lens.d_model} != {2560}:
        raise RuntimeError("expected d_model=2560")
    if {j_lens.n_prompts, r_lens.n_prompts} != {25}:
        raise RuntimeError("expected n_prompts=25")
    identity = torch.eye(2560)
    anchor_errors = {
        method: float(torch.max(torch.abs(lens.jacobians[30] - identity)))
        for method, lens in lenses.items()
    }
    if any(error != 0.0 for error in anchor_errors.values()):
        raise RuntimeError(f"non-identity target anchors: {anchor_errors}")
    fields = ("model_id", "target_layer", "skip_first", "dataset_id")
    matched = {
        field: metadata["j_lens"]["provenance"].get(field)
        == metadata["r_lens"]["provenance"].get(field)
        for field in fields
    }
    if not all(matched.values()):
        raise RuntimeError(f"lens provenance mismatch: {matched}")
    return {"anchor_max_abs_error": anchor_errors, "matched_fields": matched}


def describe_tokens(tokenizer: Any, token_ids: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "id": int(token_id),
            "token": tokenizer.convert_ids_to_tokens(int(token_id)),
            "decoded": tokenizer.decode([int(token_id)]),
        }
        for token_id in token_ids
    ]


def summarize_vector(
    logits: torch.Tensor, target_id: int, top_k: int, tokenizer: Any
) -> dict[str, Any]:
    rank, target_logit, top_ids, top_logits = rank_and_topk(
        logits, target_id, top_k
    )
    top_tokens = describe_tokens(tokenizer, top_ids)
    for token, logit in zip(top_tokens, top_logits, strict=True):
        token["logit"] = logit
    return {
        "target_rank": rank,
        "target_logit": target_logit,
        "top_tokens": top_tokens,
    }


def summarize_layers(
    layer_logits: dict[int, torch.Tensor],
    target_id: int,
    top_k: int,
    tokenizer: Any,
) -> dict[str, Any]:
    layers = {
        str(layer): summarize_vector(logits[0], target_id, top_k, tokenizer)
        for layer, logits in sorted(layer_logits.items())
    }
    best_layer, best = min(
        layers.items(), key=lambda item: (item[1]["target_rank"], int(item[0]))
    )
    return {
        "best_rank": best["target_rank"],
        "best_layer": int(best_layer),
        "top10_layers": [
            int(layer)
            for layer, values in layers.items()
            if values["target_rank"] <= 10
        ],
        "layers": layers,
    }


def encode_case(tokenizer: Any, case: BaselineCase) -> dict[str, Any]:
    encoded = tokenizer(
        case.prompt,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    token_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = [tuple(map(int, offset)) for offset in encoded["offset_mapping"]]
    char_span, span_tokens, position = resolve_probe(
        case.prompt,
        case.probe_text,
        case.probe_occurrence,
        case.probe_token_within_span,
        offsets,
    )
    target_ids = tokenizer.encode(case.target_text, add_special_tokens=False)
    if len(target_ids) != 1:
        raise ValueError(
            f"target {case.target_text!r} is not one token: "
            f"{describe_tokens(tokenizer, target_ids)}"
        )
    tokens = describe_tokens(tokenizer, token_ids)
    for token, offset in zip(tokens, offsets, strict=True):
        token["offset"] = list(offset)
    return {
        "input_ids": token_ids,
        "tokens": tokens,
        "char_span": list(char_span),
        "span_token_indices": span_tokens,
        "probe_position": position,
        "probe_token": tokens[position],
        "target_id": int(target_ids[0]),
        "target_token": describe_tokens(tokenizer, target_ids)[0],
    }


def evaluate_case(
    case: BaselineCase,
    tokenizer: Any,
    model: Any,
    lenses: dict[str, JacobianLens],
    top_k: int,
) -> dict[str, Any]:
    tokenization = encode_case(tokenizer, case)
    position = tokenization["probe_position"]
    target_id = tokenization["target_id"]
    methods: dict[str, Any] = {}
    reference_logits: torch.Tensor | None = None

    for method in ("j_lens", "r_lens"):
        layer_logits, model_logits, input_ids = lenses[method].apply(
            model, case.prompt, positions=[position]
        )
        if input_ids[0].tolist() != tokenization["input_ids"]:
            raise RuntimeError(f"token IDs differ in {method} for {case.id}")
        methods[method] = summarize_layers(
            layer_logits, target_id, top_k, tokenizer
        )
        if reference_logits is None:
            reference_logits = model_logits
        elif not torch.equal(reference_logits, model_logits):
            raise RuntimeError(f"model logits changed in {method} for {case.id}")

    logit_logits, model_logits, input_ids = lenses["j_lens"].apply(
        model, case.prompt, positions=[position], use_jacobian=False
    )
    if input_ids[0].tolist() != tokenization["input_ids"]:
        raise RuntimeError(f"token IDs differ in logit lens for {case.id}")
    if reference_logits is None or not torch.equal(reference_logits, model_logits):
        raise RuntimeError(f"model logits changed in logit lens for {case.id}")
    methods["logit_lens"] = summarize_layers(
        logit_logits, target_id, top_k, tokenizer
    )

    result = {
        **asdict(case),
        "evaluated_model": MODEL_ID,
        "source_model_mismatch": case.reference_model != MODEL_ID,
        "tokenization": tokenization,
        "methods": methods,
        "model_readout_at_probe": summarize_vector(
            reference_logits[0], target_id, top_k, tokenizer
        ),
    }
    summary = " ".join(
        f"{method}=rank{values['best_rank']}@L{values['best_layer']}"
        for method, values in methods.items()
    )
    print(
        f"{case.id}: probe={tokenization['probe_token']['decoded']!r} "
        f"target={tokenization['target_token']['decoded']!r} {summary}",
        flush=True,
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=root / "configs" / "baseline_cases.json"
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "baselines_qwen3.5-4b.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cases = load_cases(args.cases, set(args.case))
    print(f"Loading {MODEL_ID}@{MODEL_REVISION}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=args.offline
    )
    hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        local_files_only=args.offline,
    ).to(args.device)
    model = from_hf(hf_model, tokenizer, compile=False, force_bos=True)
    if (model.n_layers, model.d_model) != (32, 2560):
        raise RuntimeError(
            f"unexpected model shape {(model.n_layers, model.d_model)}"
        )

    print(f"Loading matched lenses @{LENS_REVISION}", flush=True)
    lenses: dict[str, JacobianLens] = {}
    lens_metadata: dict[str, dict[str, Any]] = {}
    for method in ("j_lens", "r_lens"):
        lenses[method], lens_metadata[method] = load_lens(method, args.offline)
    pair_checks = check_lens_pair(lenses, lens_metadata)
    results = [
        evaluate_case(case, tokenizer, model, lenses, args.top_k) for case in cases
    ]

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(torch.device(args.device))
        if args.device.startswith("cuda")
        else None,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_dtype": str(next(hf_model.parameters()).dtype),
        "jlens_revision": JLENS_REVISION,
        "lens_repo_id": LENS_REPO_ID,
        "lens_revision": LENS_REVISION,
        "lens_artifacts": lens_metadata,
        "pair_checks": pair_checks,
        "top_k": args.top_k,
    }
    payload = {"metadata": metadata, "cases": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
