"""Fit a diagonal-Hessian correction to the J-lens and evaluate held-out probes.

The default estimator, ``forward``, computes the complete vector-valued
``H[e_j, e_j]`` for every residual coordinate by nested forward-mode JVPs, so the
operator is a dense ``d_model x d_model`` diagonal rather than a low-rank
approximation. Two earlier estimators remain selectable: ``gaussian``, which is
low rank and randomised, and ``coordinate``, which uses centred finite
differences. Neither is used for the reported results.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from jlens import from_hf
from jlens.hooks import ActivationRecorder
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from j2_lens.baselines import (
    JLENS_REVISION,
    LENS_REPO_ID,
    LENS_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    BaselineCase,
    check_lens_pair,
    describe_tokens,
    encode_case,
    load_cases,
    load_lens,
    rank_and_topk,
    sha256_file,
)
from j2_lens.development import (
    DEFAULT_N_DOCS,
    DEFAULT_T_MAX,
    PILE_FILE,
    PILE_REPO_ID,
    PILE_REVISION,
    build_development_cases,
    load_pile_documents,
)


def valid_positions(seq_len: int, skip_first: int) -> list[int]:
    positions = list(range(skip_first, seq_len - 1))
    if not positions:
        raise ValueError(
            f"prompt of length {seq_len} has no positions after skip_first="
            f"{skip_first} and final-position exclusion"
        )
    return positions


def apply_low_rank_diagonal(
    output_vectors: torch.Tensor,
    input_features: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    """Apply ``mean_k output_k input_k^T`` without forming a dense matrix."""
    if output_vectors.ndim != 2 or input_features.ndim != 2:
        raise ValueError("operator factors must be rank-2")
    if output_vectors.shape[0] != input_features.shape[0]:
        raise ValueError("operator factors must have the same rank")
    if features.shape[-1] != input_features.shape[1]:
        raise ValueError("feature dimension does not match the operator")
    coefficients = features @ input_features.T
    return coefficients @ output_vectors / output_vectors.shape[0]


def apply_diagonal_operator(
    operator: dict[str, Any], features: torch.Tensor, *, shuffled: bool = False
) -> torch.Tensor:
    """Apply either the Gaussian low-rank or direct-coordinate operator."""
    device = features.device
    if "diagonal_rows" in operator:
        rows = operator["diagonal_rows"].to(device)
        if shuffled:
            rows = torch.roll(rows, shifts=1, dims=0)
        coordinates = operator["coordinates"].to(device)
        return features[..., coordinates] @ rows
    output_vectors = operator["output_vectors"].to(device)
    if shuffled:
        output_vectors = torch.roll(output_vectors, shifts=1, dims=0)
    return apply_low_rank_diagonal(
        output_vectors, operator["input_features"].to(device), features
    )


def normalized_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(target.float())
    if float(denominator.item()) == 0.0:
        return math.nan
    return float(
        (torch.linalg.vector_norm((prediction - target).float()) / denominator).item()
    )


def cosine_similarity(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            prediction.float(), target.float(), dim=0
        ).item()
    )


def load_split(path: Path, cases: list[BaselineCase]) -> dict[str, Any]:
    config = json.loads(path.read_text())
    known = {case.id for case in cases}
    development = list(config.get("development_case_ids", []))
    heldout = list(config["heldout_case_ids"])
    source = config.get("development_source", "cases")
    if source not in ("cases", "pile"):
        raise ValueError(f"unknown development_source {source!r}")
    if source == "cases" and not development:
        raise ValueError("development split must be nonempty")
    if source == "pile" and development:
        raise ValueError(
            "a pile development corpus must not also list development_case_ids"
        )
    if not heldout:
        raise ValueError("held-out split must be nonempty")
    if set(development) & set(heldout):
        raise ValueError("development and held-out cases must be disjoint")
    unknown = (set(development) | set(heldout)) - known
    if unknown:
        raise ValueError(f"unknown case IDs in split: {sorted(unknown)}")
    layers = [int(layer) for layer in config["layers"]]
    target_layer = int(config["target_layer"])
    if not layers or min(layers) < 0 or max(layers) >= target_layer:
        raise ValueError("source layers must be nonempty and below the target")
    if int(config["directions_per_layer"]) < 1:
        raise ValueError("directions_per_layer must be positive")
    if float(config["finite_difference_epsilon"]) <= 0:
        raise ValueError("finite_difference_epsilon must be positive")
    return config


def replace_hidden(output: Any, position: int, perturbation: torch.Tensor) -> Any:
    hidden = output if torch.is_tensor(output) else output[0]
    changed = hidden.clone()
    changed[0, position] = changed[0, position] + perturbation
    if torch.is_tensor(output):
        return changed
    if isinstance(output, tuple):
        return (changed, *output[1:])
    if isinstance(output, list):
        return [changed, *output[1:]]
    raise TypeError(f"unsupported block output type: {type(output)!r}")


@contextmanager
def residual_intervention(
    model: Any, source_layer: int, position: int, perturbation: torch.Tensor
):
    def hook(module: Any, inputs: Any, output: Any) -> Any:
        return replace_hidden(output, position, perturbation)

    handle = model.layers[source_layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def capture_activations(
    model: Any, input_ids: torch.Tensor, layers: list[int]
) -> dict[int, torch.Tensor]:
    with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
    return {layer: recorder.activations[layer].detach() for layer in layers}


def pad_pair_batch(
    prepared: dict[str, dict[str, Any]],
    pairs: list[tuple[str, int]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad the development prompts into one rectangular batch.

    Returns the padded ``input_ids``, the probed ``positions``, and the
    per-row exclusive reduction end ``length - 1``.

    Right padding is exact rather than approximate here: attention is causal,
    so a real position never attends to a token that follows it, and the
    reduction stops at each row's own penultimate token. The residuals at the
    positions actually read are therefore bit-identical to an unpadded forward
    pass, and ``pad_id`` cannot affect the result.
    """
    lengths = [int(prepared[case_id]["input_ids"].shape[1]) for case_id, _ in pairs]
    width = max(lengths)
    device = prepared[pairs[0][0]]["input_ids"].device
    padded = torch.full(
        (len(pairs), width), pad_id, dtype=torch.long, device=device
    )
    for row, ((case_id, _), length) in enumerate(zip(pairs, lengths, strict=True)):
        padded[row, :length] = prepared[case_id]["input_ids"][0, :length]
    positions = torch.tensor(
        [position for _, position in pairs], dtype=torch.long, device=device
    )
    ends = torch.tensor(
        [length - 1 for length in lengths], dtype=torch.long, device=device
    )
    if not bool((positions < ends).all()):
        raise ValueError("every probed position must precede its reduction end")
    return padded, positions, ends


def reduce_target_sums(
    target: torch.Tensor, positions: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    """Sum target-layer residuals over ``[position, end)`` for each row.

    ``ends`` is per row, so prompts of different lengths may share a batch:
    each row stops at its own penultimate token instead of at the padded
    batch width. Built as a mask-and-matmul so it stays differentiable under
    ``torch.func.jvp``.
    """
    index = torch.arange(target.shape[1], device=target.device)[None, :]
    mask = (index >= positions[:, None]) & (index < ends[:, None])
    return (target * mask[:, :, None].to(target.dtype)).sum(dim=1)


def intervened_target_sum(
    model: Any,
    input_ids: torch.Tensor,
    *,
    source_layer: int,
    target_layer: int,
    position: int,
    perturbation: torch.Tensor,
) -> torch.Tensor:
    with (
        torch.no_grad(),
        residual_intervention(model, source_layer, position, perturbation),
        ActivationRecorder(model.layers, at=[target_layer]) as recorder,
    ):
        model.forward(input_ids)
    target = recorder.activations[target_layer].detach().float()
    return target[0, position : input_ids.shape[1] - 1].sum(dim=0)


def batched_intervened_target_sums(
    model: Any,
    input_ids: torch.Tensor,
    *,
    source_layer: int,
    target_layer: int,
    positions: torch.Tensor,
    coordinates: torch.Tensor,
    step: float,
    ends: torch.Tensor,
) -> torch.Tensor:
    if input_ids.shape[0] != positions.numel() or positions.shape != coordinates.shape:
        raise ValueError("batched interventions must have one position and coordinate")
    if ends.shape != positions.shape:
        raise ValueError("batched interventions must have one end per row")

    def hook(module: Any, inputs: Any, output: Any) -> Any:
        hidden = output if torch.is_tensor(output) else output[0]
        changed = hidden.clone()
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        changed[batch, positions, coordinates] += step
        if torch.is_tensor(output):
            return changed
        if isinstance(output, tuple):
            return (changed, *output[1:])
        if isinstance(output, list):
            return [changed, *output[1:]]
        raise TypeError(f"unsupported block output type: {type(output)!r}")

    handle = model.layers[source_layer].register_forward_hook(hook)
    try:
        with torch.no_grad(), ActivationRecorder(
            model.layers, at=[target_layer]
        ) as recorder:
            model.forward(input_ids)
    finally:
        handle.remove()
    target = recorder.activations[target_layer].detach().float()
    return reduce_target_sums(target, positions, ends)


def batched_forward_diagonal_curvature(
    model: Any,
    input_ids: torch.Tensor,
    *,
    source_layer: int,
    target_layer: int,
    positions: torch.Tensor,
    directions: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Return vector-valued ``H[e_j,e_j]`` using forward-over-forward AD."""
    if input_ids.shape[0] != positions.numel() or directions.shape != (
        input_ids.shape[0],
        model.d_model,
    ):
        raise ValueError("forward directions must match the intervention batch")
    if ends.shape != positions.shape:
        raise ValueError("forward directions must have one end per row")

    def downstream(perturbations: torch.Tensor) -> torch.Tensor:
        def hook(module: Any, inputs: Any, output: Any) -> Any:
            hidden = output if torch.is_tensor(output) else output[0]
            position_mask = torch.nn.functional.one_hot(
                positions, num_classes=hidden.shape[1]
            ).to(hidden.dtype)
            changed = hidden + position_mask[:, :, None] * perturbations[:, None, :]
            if torch.is_tensor(output):
                return changed
            if isinstance(output, tuple):
                return (changed, *output[1:])
            if isinstance(output, list):
                return [changed, *output[1:]]
            raise TypeError(f"unsupported block output type: {type(output)!r}")

        handle = model.layers[source_layer].register_forward_hook(hook)
        try:
            with ActivationRecorder(model.layers, at=[target_layer]) as recorder:
                model.forward(input_ids)
        finally:
            handle.remove()
        target = recorder.activations[target_layer].float()
        return reduce_target_sums(target, positions, ends)

    zero = torch.zeros_like(directions)

    def first_directional_derivative(perturbations: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(
            downstream,
            (perturbations,),
            (directions,),
        )[1]

    return torch.func.jvp(
        first_directional_derivative,
        (zero,),
        (directions,),
    )[1]


def batched_coordinate_curvature(
    model: Any,
    input_ids: torch.Tensor,
    repeated_clean: torch.Tensor,
    *,
    source_layer: int,
    target_layer: int,
    positions: torch.Tensor,
    coordinates: torch.Tensor,
    step: float,
    n_coordinates: int,
    n_pairs: int,
    ends: torch.Tensor,
) -> torch.Tensor:
    plus = batched_intervened_target_sums(
        model,
        input_ids,
        source_layer=source_layer,
        target_layer=target_layer,
        positions=positions,
        coordinates=coordinates,
        step=step,
        ends=ends,
    )
    minus = batched_intervened_target_sums(
        model,
        input_ids,
        source_layer=source_layer,
        target_layer=target_layer,
        positions=positions,
        coordinates=coordinates,
        step=-step,
        ends=ends,
    )
    per_pair = (plus - 2 * repeated_clean + minus) / (step * step)
    return per_pair.reshape(n_coordinates, n_pairs, model.d_model).mean(dim=1)


def centered_second_difference(
    model: Any,
    input_ids: torch.Tensor,
    clean_sum: torch.Tensor,
    direction: torch.Tensor,
    *,
    source_layer: int,
    target_layer: int,
    position: int,
    step: float,
) -> torch.Tensor:
    plus = intervened_target_sum(
        model,
        input_ids,
        source_layer=source_layer,
        target_layer=target_layer,
        position=position,
        perturbation=step * direction,
    )
    minus = intervened_target_sum(
        model,
        input_ids,
        source_layer=source_layer,
        target_layer=target_layer,
        position=position,
        perturbation=-step * direction,
    )
    return (plus - 2 * clean_sum + minus) / (step * step)


def prepare_cases(
    model: Any,
    tokenizer: Any,
    cases: list[BaselineCase],
    layers: list[int],
    target_layer: int,
) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    record_layers = sorted({*layers, target_layer, model.n_layers - 1})
    for case in cases:
        tokenization = encode_case(tokenizer, case)
        input_ids = model.encode(case.prompt, max_length=512)
        if input_ids[0].tolist() != tokenization["input_ids"]:
            raise RuntimeError(f"tokenization mismatch for {case.id}")
        prepared[case.id] = {
            "case": case,
            "tokenization": tokenization,
            "input_ids": input_ids,
            "activations": capture_activations(model, input_ids, record_layers),
        }
    return prepared


def prepare_development_cases(
    model: Any,
    tokenizer: Any,
    corpus: list[dict[str, Any]],
    layers: list[int],
    target_layer: int,
) -> dict[str, dict[str, Any]]:
    """Prepare corpus documents used only to estimate moments and curvature.

    Unlike evaluation cases these have no probe span and no target token, so
    they skip ``encode_case`` entirely. Only ``input_ids`` and the recorded
    activations are needed downstream.
    """
    prepared: dict[str, dict[str, Any]] = {}
    record_layers = sorted({*layers, target_layer, model.n_layers - 1})
    for document in corpus:
        input_ids = model.encode(document["prompt"], max_length=512)
        if input_ids.shape[1] < 8:
            continue
        prepared[document["id"]] = {
            "case": None,
            "tokenization": None,
            "input_ids": input_ids,
            "activations": capture_activations(model, input_ids, record_layers),
        }
    if not prepared:
        raise ValueError("no usable development documents after encoding")
    return prepared


def subsample_pairs(
    pairs: list[tuple[str, int]], limit: int | None, seed: int
) -> list[tuple[str, int]]:
    """Take a deterministic, spread-out subsample of development pairs.

    The averaged Hessian costs one forward-over-forward pass per (coordinate,
    pair), so its sample count is the binding compute constraint while the
    activation moments are nearly free. Subsampling here lets the moments use
    every pair while the curvature average uses as many as the budget allows.
    """
    if limit is None or limit >= len(pairs):
        return list(pairs)
    if limit < 1:
        raise ValueError("hessian pair limit must be positive")
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(len(pairs), generator=generator)[:limit]
    return [pairs[index] for index in sorted(order.tolist())]


def development_statistics(
    prepared: dict[str, dict[str, Any]],
    development_ids: list[str],
    layers: list[int],
    target_layer: int,
    skip_first: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], list[tuple[str, int]]]:
    pairs: list[tuple[str, int]] = []
    for case_id in development_ids:
        seq_len = prepared[case_id]["input_ids"].shape[1]
        pairs.extend(
            (case_id, position)
            for position in valid_positions(seq_len, skip_first)
        )
    statistics_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    target_rows = torch.stack(
        [
            prepared[case_id]["activations"][target_layer][0, position].float()
            for case_id, position in pairs
        ]
    )
    target_mean = target_rows.mean(dim=0)
    for layer in layers:
        source_rows = torch.stack(
            [
                prepared[case_id]["activations"][layer][0, position].float()
                for case_id, position in pairs
            ]
        )
        source_mean = source_rows.mean(dim=0)
        source_variance = (source_rows - source_mean).square().mean(dim=0)
        statistics_by_layer[layer] = {
            "source_rows": source_rows,
            "source_mean": source_mean,
            "source_variance": source_variance,
            "target_rows": target_rows,
            "target_mean": target_mean,
        }
    return statistics_by_layer, pairs


def fit_diagonal_operator(
    model: Any,
    prepared: dict[str, dict[str, Any]],
    pairs: list[tuple[str, int]],
    *,
    source_layer: int,
    target_layer: int,
    directions: int,
    epsilon: float,
    step_checks: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    output_vectors: list[torch.Tensor] = []
    input_features: list[torch.Tensor] = []
    sample_records: list[dict[str, Any]] = []
    pair_order: list[int] = []
    started = time.perf_counter()
    while len(pair_order) < directions:
        pair_order.extend(
            torch.randperm(
                len(pairs), generator=generator, device=generator.device
            ).tolist()
        )
    for sample_index, pair_index in enumerate(pair_order[:directions]):
        case_id, position = pairs[pair_index]
        item = prepared[case_id]
        clean_target = item["activations"][target_layer].float()
        clean_sum = clean_target[
            0, position : item["input_ids"].shape[1] - 1
        ].sum(dim=0)
        direction = torch.randn(
            model.d_model,
            generator=generator,
            device=item["input_ids"].device,
            dtype=clean_sum.dtype,
        )

        curvature = centered_second_difference(
            model,
            item["input_ids"],
            clean_sum,
            direction,
            source_layer=source_layer,
            target_layer=target_layer,
            position=position,
            step=epsilon,
        )
        record: dict[str, Any] = {
            "sample": sample_index,
            "case_id": case_id,
            "position": position,
            "direction_rms": float(torch.sqrt(direction.square().mean()).item()),
            "curvature_rms": float(torch.sqrt(curvature.square().mean()).item()),
        }
        if sample_index < step_checks:
            half_curvature = centered_second_difference(
                model,
                item["input_ids"],
                clean_sum,
                direction,
                source_layer=source_layer,
                target_layer=target_layer,
                position=position,
                step=epsilon / 2,
            )
            difference = torch.linalg.vector_norm(curvature - half_curvature)
            denominator = torch.linalg.vector_norm(half_curvature)
            record["half_step_relative_error"] = float(
                (difference / torch.clamp(denominator, min=1e-30)).item()
            )
            record["half_step_cosine"] = cosine_similarity(
                curvature, half_curvature
            )
            record["half_step_curvature_rms"] = float(
                torch.sqrt(half_curvature.square().mean()).item()
            )
        if not torch.isfinite(curvature).all():
            raise RuntimeError(f"non-finite curvature sample at layer {source_layer}")
        output_vectors.append(curvature.cpu())
        input_features.append(((direction.square() - 1) / 2).cpu())
        sample_records.append(record)
        print(
            f"L{source_layer} diagonal sample {sample_index + 1}/{directions} "
            f"{case_id}@{position} rms={record['curvature_rms']:.3e}",
            flush=True,
        )
    return {
        "output_vectors": torch.stack(output_vectors),
        "input_features": torch.stack(input_features),
        "samples": sample_records,
        "elapsed_seconds": time.perf_counter() - started,
    }


def fit_coordinate_operator(
    model: Any,
    prepared: dict[str, dict[str, Any]],
    pairs: list[tuple[str, int]],
    *,
    source_layer: int,
    target_layer: int,
    epsilon: float,
    coordinate_batch_size: int,
    max_coordinates: int,
    check_half_step: bool,
) -> dict[str, Any]:
    """Compute development-averaged ``H[e_j,e_j]`` in coordinate batches."""
    if coordinate_batch_size < 1:
        raise ValueError("coordinate_batch_size must be positive")
    if not 1 <= max_coordinates <= model.d_model:
        raise ValueError("max_coordinates must lie in [1, d_model]")
    pair_input_ids, pair_positions, pair_ends = pad_pair_batch(prepared, pairs)
    clean_sums = torch.stack(
        [
            prepared[case_id]["activations"][target_layer][
                0, position : prepared[case_id]["input_ids"].shape[1] - 1
            ]
            .sum(dim=0)
            .float()
            for case_id, position in pairs
        ]
    )
    rows: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    n_pairs = len(pairs)
    for batch_start in range(0, max_coordinates, coordinate_batch_size):
        batch_coordinates = torch.arange(
            batch_start,
            min(batch_start + coordinate_batch_size, max_coordinates),
            device=pair_input_ids.device,
        )
        n_coordinates = batch_coordinates.numel()
        input_ids = pair_input_ids.repeat(n_coordinates, 1)
        positions = pair_positions.repeat(n_coordinates)
        ends = pair_ends.repeat(n_coordinates)
        coordinates = batch_coordinates.repeat_interleave(n_pairs)
        repeated_clean = clean_sums.repeat(n_coordinates, 1)

        curvature = batched_coordinate_curvature(
            model,
            input_ids,
            repeated_clean,
            source_layer=source_layer,
            target_layer=target_layer,
            positions=positions,
            coordinates=coordinates,
            step=epsilon,
            n_coordinates=n_coordinates,
            n_pairs=n_pairs,
            ends=ends,
        )
        record: dict[str, Any] = {
            "coordinate_start": batch_start,
            "coordinate_end": batch_start + n_coordinates,
            "mean_curvature_rms": float(
                torch.sqrt(curvature.square().mean()).item()
            ),
        }
        if check_half_step and batch_start == 0:
            half_curvature = batched_coordinate_curvature(
                model,
                input_ids,
                repeated_clean,
                source_layer=source_layer,
                target_layer=target_layer,
                positions=positions,
                coordinates=coordinates,
                step=epsilon / 2,
                n_coordinates=n_coordinates,
                n_pairs=n_pairs,
                ends=ends,
            )
            per_coordinate_norm = torch.linalg.vector_norm(half_curvature, dim=1)
            per_coordinate_error = torch.linalg.vector_norm(
                curvature - half_curvature, dim=1
            ) / torch.clamp(per_coordinate_norm, min=1e-30)
            per_coordinate_cosine = torch.nn.functional.cosine_similarity(
                curvature, half_curvature, dim=1
            )
            record.update(
                {
                    "half_step_mean_relative_error": float(
                        per_coordinate_error.mean().item()
                    ),
                    "half_step_max_relative_error": float(
                        per_coordinate_error.max().item()
                    ),
                    "half_step_mean_cosine": float(
                        per_coordinate_cosine.mean().item()
                    ),
                    "half_step_min_cosine": float(
                        per_coordinate_cosine.min().item()
                    ),
                }
            )
        if not torch.isfinite(curvature).all():
            raise RuntimeError(
                f"non-finite coordinate curvature at layer {source_layer}"
            )
        rows.append(curvature.cpu())
        records.append(record)
        print(
            f"L{source_layer} coordinates {batch_start}:"
            f"{batch_start + n_coordinates}/{max_coordinates} "
            f"rms={record['mean_curvature_rms']:.3e}",
            flush=True,
        )
    return {
        "coordinates": torch.arange(max_coordinates),
        "diagonal_rows": torch.cat(rows, dim=0),
        "samples": records,
        "elapsed_seconds": time.perf_counter() - started,
    }


def fit_forward_operator(
    model: Any,
    prepared: dict[str, dict[str, Any]],
    pairs: list[tuple[str, int]],
    *,
    source_layer: int,
    target_layer: int,
    coordinate_batch_size: int,
    max_coordinates: int,
    coordinate_offset: int = 0,
) -> dict[str, Any]:
    """Compute every development-averaged diagonal with forward-mode AD.

    ``coordinate_offset`` selects the half-open coordinate range
    ``[offset, offset + max_coordinates)``. Coordinates are independent, so a
    long fit can be split into disjoint shards run concurrently on separate
    GPUs and merged afterwards with ``j2-merge``.
    """
    if coordinate_batch_size < 1:
        raise ValueError("coordinate_batch_size must be positive")
    if not 1 <= max_coordinates <= model.d_model:
        raise ValueError("max_coordinates must lie in [1, d_model]")
    if not 0 <= coordinate_offset < model.d_model:
        raise ValueError("coordinate_offset must lie in [0, d_model)")
    stop = coordinate_offset + max_coordinates
    if stop > model.d_model:
        raise ValueError("coordinate shard runs past d_model")
    pair_input_ids, pair_positions, pair_ends = pad_pair_batch(prepared, pairs)
    rows: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    n_pairs = len(pairs)
    for batch_start in range(coordinate_offset, stop, coordinate_batch_size):
        batch_coordinates = torch.arange(
            batch_start,
            min(batch_start + coordinate_batch_size, stop),
            device=pair_input_ids.device,
        )
        n_coordinates = batch_coordinates.numel()
        input_ids = pair_input_ids.repeat(n_coordinates, 1)
        positions = pair_positions.repeat(n_coordinates)
        ends = pair_ends.repeat(n_coordinates)
        coordinates = batch_coordinates.repeat_interleave(n_pairs)
        directions = torch.zeros(
            input_ids.shape[0],
            model.d_model,
            device=input_ids.device,
            dtype=next(model.layers[source_layer].parameters()).dtype,
        )
        directions[
            torch.arange(input_ids.shape[0], device=input_ids.device), coordinates
        ] = 1
        per_pair = batched_forward_diagonal_curvature(
            model,
            input_ids,
            source_layer=source_layer,
            target_layer=target_layer,
            positions=positions,
            directions=directions,
            ends=ends,
        )
        curvature = per_pair.reshape(
            n_coordinates, n_pairs, model.d_model
        ).mean(dim=1)
        if not torch.isfinite(curvature).all():
            raise RuntimeError(
                f"non-finite forward curvature at layer {source_layer}"
            )
        record = {
            "coordinate_start": batch_start,
            "coordinate_end": batch_start + n_coordinates,
            "mean_curvature_rms": float(
                torch.sqrt(curvature.square().mean()).item()
            ),
        }
        rows.append(curvature.detach().cpu())
        records.append(record)
        print(
            f"L{source_layer} forward coordinates {batch_start}:"
            f"{batch_start + n_coordinates}/{stop} "
            f"rms={record['mean_curvature_rms']:.3e}",
            flush=True,
        )
    return {
        "coordinates": torch.arange(coordinate_offset, stop),
        "diagonal_rows": torch.cat(rows, dim=0),
        "samples": records,
        "elapsed_seconds": time.perf_counter() - started,
    }


def load_reusable_operators(
    path: Path,
    layers: list[int],
    config: dict[str, Any],
    estimator: str,
    stats_by_layer: dict[int, dict[str, torch.Tensor]],
    tolerance: float,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Reload a previously fitted operator instead of refitting it.

    The operator depends only on the development cases, so it may be reused
    verbatim when the held-out set grows. Reuse is refused unless the stored
    development split, estimator, target layer, and coordinate budget match the
    current configuration, and unless the freshly recomputed development
    moments agree with the stored ones.
    """
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    stored = artifact["metadata"]
    stored_split = stored["split"]
    mismatches = []
    if list(stored_split.get("development_case_ids", [])) != list(
        config.get("development_case_ids", [])
    ):
        mismatches.append("development_case_ids")
    if stored_split.get("development_source", "cases") != config.get(
        "development_source", "cases"
    ):
        mismatches.append("development_source")
    stored_corpus = stored.get("development_corpus")
    if stored_corpus is not None:
        if stored_corpus.get("revision") != PILE_REVISION:
            mismatches.append("development_corpus_revision")
    if int(stored_split["target_layer"]) != int(config["target_layer"]):
        mismatches.append("target_layer")
    if int(stored_split["skip_first"]) != int(config["skip_first"]):
        mismatches.append("skip_first")
    if stored["effective_estimator"] != estimator:
        mismatches.append("estimator")
    if stored["model_revision"] != MODEL_REVISION:
        mismatches.append("model_revision")
    if mismatches:
        raise ValueError(
            f"cannot reuse {path}: mismatched {', '.join(mismatches)}"
        )

    operators: dict[int, dict[str, Any]] = {}
    for layer in layers:
        if layer not in artifact["layers"]:
            raise ValueError(f"{path} has no fitted layer {layer}")
        entry = artifact["layers"][layer]
        for key in ("source_mean", "source_variance", "target_mean"):
            fresh = stats_by_layer[layer][key].cpu()
            drift = float((fresh - entry[key].cpu()).abs().max().item())
            if drift > tolerance:
                raise ValueError(
                    f"{path} layer {layer} {key} drifted by {drift:.3e} "
                    f"(tolerance {tolerance:.3e})"
                )
        operators[layer] = {
            **{
                key: entry[key]
                for key in (
                    "output_vectors",
                    "input_features",
                    "coordinates",
                    "diagonal_rows",
                )
                if key in entry
            },
            "samples": entry["samples"],
            "elapsed_seconds": entry["elapsed_seconds"],
        }
        print(f"Reused fitted layer {layer} from {path}", flush=True)
    return operators, stored


def development_summary_for_layer(
    stats: dict[str, torch.Tensor],
    operator: dict[str, Any],
    j_matrix: torch.Tensor,
) -> dict[str, Any]:
    device = stats["source_rows"].device
    source_delta = stats["source_rows"] - stats["source_mean"]
    raw_j = stats["source_rows"] @ j_matrix.to(device).T
    quadratic_features = source_delta.square() - stats["source_variance"]
    correction = 0.5 * apply_diagonal_operator(operator, quadratic_features)
    shuffled_correction = 0.5 * apply_diagonal_operator(
        operator, quadratic_features, shuffled=True
    )

    predictions = {
        "j_lens": raw_j,
        "j2_raw": raw_j + correction,
        "j2_shuffled": raw_j + shuffled_correction,
    }
    development_metrics = {
        method: {
            "normalized_error": normalized_error(prediction, stats["target_rows"]),
            "cosine": cosine_similarity(
                prediction.flatten(), stats["target_rows"].flatten()
            ),
        }
        for method, prediction in predictions.items()
    }
    return {"development_metrics": development_metrics}


def residual_methods(
    source: torch.Tensor,
    stats: dict[str, torch.Tensor],
    operator: dict[str, Any],
    j_matrix: torch.Tensor,
    r_matrix: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = source.device
    source_mean = stats["source_mean"].to(device)
    source_variance = stats["source_variance"].to(device)
    j_matrix = j_matrix.to(device)
    r_matrix = r_matrix.to(device)
    delta = source.float() - source_mean
    raw_j = j_matrix @ source.float()
    quadratic_features = delta.square() - source_variance
    correction = 0.5 * apply_diagonal_operator(operator, quadratic_features)
    shuffled_correction = 0.5 * apply_diagonal_operator(
        operator, quadratic_features, shuffled=True
    )
    return {
        "logit_lens": source.float(),
        "j_lens": raw_j,
        "j2_raw": raw_j + correction,
        "j2_shuffled": raw_j + shuffled_correction,
        "r_lens": r_matrix @ source.float(),
    }


def readout_metrics(
    model: Any,
    tokenizer: Any,
    residual: torch.Tensor,
    target_residual: torch.Tensor,
    model_logits: torch.Tensor,
    target_id: int,
    top_k: int,
) -> dict[str, Any]:
    logits = model.unembed(residual[None]).float()[0]
    rank, target_logit, top_ids, top_logits = rank_and_topk(logits, target_id, top_k)
    model_log_probs = torch.log_softmax(model_logits.float(), dim=-1)
    predicted_log_probs = torch.log_softmax(logits, dim=-1)
    kl = torch.sum(
        model_log_probs.exp() * (model_log_probs - predicted_log_probs)
    )
    top_tokens = describe_tokens(tokenizer, top_ids)
    for token, logit in zip(top_tokens, top_logits, strict=True):
        token["logit"] = logit
    return {
        "target_rank": rank,
        "target_logit": target_logit,
        "target_in_top_k": rank <= top_k,
        "kl_model_to_lens": float(kl.item()),
        "residual_cosine": cosine_similarity(residual, target_residual),
        "normalized_residual_error": normalized_error(residual, target_residual),
        "top_tokens": top_tokens,
    }


def evaluate_heldout(
    model: Any,
    tokenizer: Any,
    prepared: dict[str, dict[str, Any]],
    heldout_ids: list[str],
    layers: list[int],
    target_layer: int,
    stats_by_layer: dict[int, dict[str, torch.Tensor]],
    operators: dict[int, dict[str, Any]],
    lenses: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    results = []
    for case_id in heldout_ids:
        item = prepared[case_id]
        case = item["case"]
        tokenization = item["tokenization"]
        position = tokenization["probe_position"]
        target_id = tokenization["target_id"]
        target_residual = item["activations"][target_layer][0, position].float()
        final_residual = item["activations"][model.n_layers - 1][0, position].float()
        model_logits = model.unembed(final_residual[None]).float()[0]
        layer_results: dict[str, Any] = {}
        for layer in layers:
            methods = residual_methods(
                item["activations"][layer][0, position],
                stats_by_layer[layer],
                operators[layer],
                lenses["j_lens"].jacobians[layer],
                lenses["r_lens"].jacobians[layer],
            )
            layer_results[str(layer)] = {
                method: readout_metrics(
                    model,
                    tokenizer,
                    residual,
                    target_residual,
                    model_logits,
                    target_id,
                    top_k,
                )
                for method, residual in methods.items()
            }
        results.append(
            {
                "id": case.id,
                "category": case.category,
                "prompt": case.prompt,
                "probe_position": position,
                "probe_token": tokenization["probe_token"],
                "target_token": tokenization["target_token"],
                "layers": layer_results,
            }
        )
        print(f"Evaluated held-out case {case.id}", flush=True)
    return results


def aggregate_results(
    cases: list[dict[str, Any]], layers: list[int]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for layer in layers:
        methods = cases[0]["layers"][str(layer)]
        layer_summary: dict[str, Any] = {}
        for method in methods:
            rows = [case["layers"][str(layer)][method] for case in cases]
            ranks = [row["target_rank"] for row in rows]
            layer_summary[method] = {
                "mean_target_rank": sum(ranks) / len(ranks),
                "median_target_rank": statistics.median(ranks),
                "top_1_count": sum(rank <= 1 for rank in ranks),
                "top_10_count": sum(rank <= 10 for rank in ranks),
                "mean_kl_model_to_lens": sum(
                    row["kl_model_to_lens"] for row in rows
                )
                / len(rows),
                "mean_residual_cosine": sum(
                    row["residual_cosine"] for row in rows
                )
                / len(rows),
                "mean_normalized_residual_error": sum(
                    row["normalized_residual_error"] for row in rows
                )
                / len(rows),
            }
        base_ranks = [
            case["layers"][str(layer)]["j_lens"]["target_rank"] for case in cases
        ]
        for comparison in ("j2_raw", "j2_shuffled"):
            comparison_ranks = [
                case["layers"][str(layer)][comparison]["target_rank"]
                for case in cases
            ]
            layer_summary[comparison]["rank_wins_vs_j_lens"] = sum(
                new < old
                for new, old in zip(comparison_ranks, base_ranks, strict=True)
            )
            layer_summary[comparison]["rank_losses_vs_j_lens"] = sum(
                new > old
                for new, old in zip(comparison_ranks, base_ranks, strict=True)
            )
        summary[str(layer)] = layer_summary
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=root / "configs" / "baseline_cases.json"
    )
    parser.add_argument(
        "--split", type=Path, default=root / "configs" / "evaluation_split.json"
    )
    parser.add_argument("--directions", type=int)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--step-checks", type=int)
    parser.add_argument(
        "--estimator",
        choices=("forward", "gaussian", "coordinate"),
        default="forward",
    )
    parser.add_argument("--layer", action="append", type=int, default=[])
    parser.add_argument("--coordinate-batch-size", type=int, default=16)
    parser.add_argument("--max-coordinates", type=int)
    parser.add_argument(
        "--coordinate-offset",
        type=int,
        default=0,
        help="first coordinate of this shard; see j2-merge to combine shards",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="stop after fitting; skip held-out evaluation (for shard jobs)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=root / "results" / "hessian_lens_qwen3.5-4b.pt",
    )
    parser.add_argument(
        "--pile-docs",
        type=int,
        help="number of pile-10k documents for a corpus development set",
    )
    parser.add_argument(
        "--pile-t-max",
        type=int,
        help="token budget per development document (J-lens used 128)",
    )
    parser.add_argument(
        "--hessian-pairs",
        type=int,
        help=(
            "cap the (document, position) samples used for the averaged "
            "Hessian; the activation moments always use every pair"
        ),
    )
    parser.add_argument(
        "--reuse-artifact",
        type=Path,
        help=(
            "reload a previously fitted operator instead of refitting; the "
            "development split, estimator, and model revision must match"
        ),
    )
    parser.add_argument(
        "--reuse-tolerance",
        type=float,
        default=1e-4,
        help="maximum allowed drift in recomputed development moments",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "evaluation_qwen3.5-4b.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    all_cases = load_cases(args.cases, set())
    config = load_split(args.split, all_cases)
    directions = args.directions or int(config["directions_per_layer"])
    epsilon = args.epsilon or float(config["finite_difference_epsilon"])
    step_checks = (
        args.step_checks
        if args.step_checks is not None
        else int(config["step_check_directions"])
    )
    layers = args.layer or [int(layer) for layer in config["layers"]]
    target_layer = int(config["target_layer"])
    if directions < 1 or epsilon <= 0 or not 0 <= step_checks <= directions:
        raise ValueError("invalid directions, epsilon, or step-check count")
    if any(layer < 0 or layer >= target_layer for layer in layers):
        raise ValueError("source layers must be below the target layer")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    selected_ids = set(config.get("development_case_ids", [])) | set(
        config["heldout_case_ids"]
    )
    cases = [case for case in all_cases if case.id in selected_ids]
    print(f"Loading float32 {MODEL_ID}@{MODEL_REVISION}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=args.offline
    )
    hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=args.offline,
    ).to(args.device)
    model = from_hf(hf_model, tokenizer, compile=False, force_bos=True)
    if (model.n_layers, model.d_model) != (32, 2560):
        raise RuntimeError(
            f"unexpected model shape {(model.n_layers, model.d_model)}"
        )
    lenses: dict[str, Any] = {}
    lens_metadata: dict[str, dict[str, Any]] = {}
    for method in ("j_lens", "r_lens"):
        lenses[method], lens_metadata[method] = load_lens(method, args.offline)
    pair_checks = check_lens_pair(lenses, lens_metadata)

    prepared = prepare_cases(model, tokenizer, cases, layers, target_layer)
    corpus_metadata: dict[str, Any] | None = None
    if config.get("development_source") == "pile":
        n_docs = args.pile_docs or int(config.get("pile_docs", DEFAULT_N_DOCS))
        t_max = args.pile_t_max or int(config.get("pile_t_max", DEFAULT_T_MAX))
        documents = load_pile_documents(n_docs, offline=args.offline)
        corpus = build_development_cases(documents, tokenizer, t_max=t_max)
        prepared.update(
            prepare_development_cases(
                model, tokenizer, corpus, layers, target_layer
            )
        )
        development_case_ids = [
            document["id"] for document in corpus if document["id"] in prepared
        ]
        corpus_metadata = {
            "source": "pile",
            "repo_id": PILE_REPO_ID,
            "revision": PILE_REVISION,
            "file": PILE_FILE,
            "n_docs_requested": n_docs,
            "t_max": t_max,
            "document_ids": development_case_ids,
            "token_lengths": [
                int(prepared[case_id]["input_ids"].shape[1])
                for case_id in development_case_ids
            ],
        }
    else:
        development_case_ids = list(config["development_case_ids"])

    stats_by_layer, development_pairs = development_statistics(
        prepared,
        development_case_ids,
        layers,
        target_layer,
        int(config["skip_first"]),
    )
    hessian_limit = args.hessian_pairs or config.get("hessian_pairs")
    hessian_pairs = subsample_pairs(
        development_pairs,
        int(hessian_limit) if hessian_limit else None,
        int(config["seed"]),
    )
    print(
        f"Development: {len(development_case_ids)} documents, "
        f"{len(development_pairs)} pairs for moments, "
        f"{len(hessian_pairs)} pairs for the averaged Hessian",
        flush=True,
    )
    generator = torch.Generator(device=args.device)
    generator.manual_seed(int(config["seed"]))
    operators: dict[int, dict[str, Any]] = {}
    development_summaries: dict[int, dict[str, Any]] = {}
    reused_from: dict[str, Any] | None = None
    if args.reuse_artifact is not None:
        operators, reused_metadata = load_reusable_operators(
            args.reuse_artifact,
            layers,
            config,
            args.estimator,
            stats_by_layer,
            args.reuse_tolerance,
        )
        reused_from = {
            "path": str(args.reuse_artifact),
            "sha256": sha256_file(args.reuse_artifact),
            "fitted_at": reused_metadata["created_at"],
        }
    for layer in layers:
        if layer in operators:
            pass
        elif args.estimator == "forward":
            operators[layer] = fit_forward_operator(
                model,
                prepared,
                hessian_pairs,
                source_layer=layer,
                target_layer=target_layer,
                coordinate_batch_size=args.coordinate_batch_size,
                max_coordinates=args.max_coordinates or model.d_model,
                coordinate_offset=args.coordinate_offset,
            )
        elif args.estimator == "gaussian":
            operators[layer] = fit_diagonal_operator(
                model,
                prepared,
                hessian_pairs,
                source_layer=layer,
                target_layer=target_layer,
                directions=directions,
                epsilon=epsilon,
                step_checks=step_checks,
                generator=generator,
            )
        else:
            operators[layer] = fit_coordinate_operator(
                model,
                prepared,
                hessian_pairs,
                source_layer=layer,
                target_layer=target_layer,
                epsilon=epsilon,
                coordinate_batch_size=args.coordinate_batch_size,
                max_coordinates=args.max_coordinates or model.d_model,
                check_half_step=step_checks > 0,
            )
        development_summaries[layer] = development_summary_for_layer(
            stats_by_layer[layer],
            operators[layer],
            lenses["j_lens"].jacobians[layer],
        )
        development_metrics = development_summaries[layer]["development_metrics"]
        print(
            f"L{layer} development normalized error: "
            f"J={development_metrics['j_lens']['normalized_error']:.3e} "
            f"J2={development_metrics['j2_raw']['normalized_error']:.3e}",
            flush=True,
        )

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_dtype": str(next(hf_model.parameters()).dtype),
        "jlens_revision": JLENS_REVISION,
        "lens_repo_id": LENS_REPO_ID,
        "lens_revision": LENS_REVISION,
        "lens_artifacts": lens_metadata,
        "pair_checks": pair_checks,
        "split": config,
        "effective_directions_per_layer": directions,
        "effective_finite_difference_epsilon": epsilon,
        "effective_step_check_directions": step_checks,
        "effective_estimator": args.estimator,
        "effective_coordinate_batch_size": args.coordinate_batch_size,
        "effective_max_coordinates": args.max_coordinates or model.d_model,
        "effective_coordinate_offset": args.coordinate_offset,
        "hessian_scalar_calibration": None,
        "reused_operator": reused_from,
        "development_corpus": corpus_metadata,
        "development_case_ids": development_case_ids,
        "n_moment_pairs": len(development_pairs),
        "n_hessian_pairs": len(hessian_pairs),
        "quadratic_coefficient": 0.5,
        "diagonal_estimator": (
            "Forward: compute all vector-valued H[e_j,e_j] by nested JVPs. "
            "Gaussian: for z~N(0,I), average H[z,z] outer (z^2-1)/2. "
            "Coordinate: directly compute every development-averaged "
            "H[e_j,e_j]. Both use centered residual interventions."
        ),
        "operator_output": (
            "sum of target-layer residuals from the sampled source position "
            "through the penultimate token, matching the J-lens fitting reduction"
        ),
    }
    artifact = {
        "metadata": metadata,
        "layers": {
            layer: {
                **{
                    key: operators[layer][key]
                    for key in (
                        "output_vectors",
                        "input_features",
                        "coordinates",
                        "diagonal_rows",
                    )
                    if key in operators[layer]
                },
                "source_mean": stats_by_layer[layer]["source_mean"].cpu(),
                "source_variance": stats_by_layer[layer]["source_variance"].cpu(),
                "target_mean": stats_by_layer[layer]["target_mean"].cpu(),
                "development_summary": development_summaries[layer],
                "samples": operators[layer]["samples"],
                "elapsed_seconds": operators[layer]["elapsed_seconds"],
            }
            for layer in layers
        },
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact_temporary = args.artifact.with_suffix(args.artifact.suffix + ".tmp")
    torch.save(artifact, artifact_temporary)
    artifact_temporary.replace(args.artifact)
    artifact_record = {
        "filename": args.artifact.name,
        "size_bytes": args.artifact.stat().st_size,
        "sha256": sha256_file(args.artifact),
    }
    if args.fit_only:
        print(
            f"Wrote {args.artifact} (fit only; merge shards with j2-merge)",
            flush=True,
        )
        return

    heldout = evaluate_heldout(
        model,
        tokenizer,
        prepared,
        config["heldout_case_ids"],
        layers,
        target_layer,
        stats_by_layer,
        operators,
        lenses,
        int(config["top_k"]),
    )
    aggregate = aggregate_results(heldout, layers)

    payload = {
        "metadata": metadata,
        "artifact": artifact_record,
        "fit": {
            str(layer): {
                "samples": operators[layer]["samples"],
                "elapsed_seconds": operators[layer]["elapsed_seconds"],
                "development_summary": development_summaries[layer],
            }
            for layer in layers
        },
        "aggregate": aggregate,
        "heldout_cases": heldout,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    output_temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output_temporary.replace(args.output)
    print(f"Wrote {args.artifact} and {args.output}", flush=True)


if __name__ == "__main__":
    main()
