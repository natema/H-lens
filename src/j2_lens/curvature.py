"""Estimate coordinate-diagonal versus total downstream Hessian energy.

Historical diagnostic, retained for provenance. It informed the decision to
estimate the full diagonal at all — coordinate-diagonal terms carry under 1% of
the Hessian's Frobenius energy — but it constructs no operator and is not part
of the reported pipeline. Nothing under ``src/`` or ``pilot/`` imports it; only
its own test does.

The randomized projections here are not the ones later removed from the lens.
Estimating ``||H||_F`` without materialising a Hessian requires random probes,
so randomization is intrinsic to this question. The lens itself is built in
``evaluation.py`` by forward-mode autodiff, with no random component, no
importance sampling and no ``w``-projection.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from jlens import from_hf
from jlens.hooks import ActivationRecorder
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from j2_lens.baselines import (
    MODEL_ID,
    MODEL_REVISION,
    BaselineCase,
    encode_case,
    load_cases,
)

DEFAULT_CASES = ("typo_aganst", "multihop_sushi", "association_jordan")
DEFAULT_LAYERS = (6, 12, 20)
TARGET_LAYER = 30
NATURAL_IMPORTANCE_WEIGHT = 0.5


def rademacher(
    size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool = False,
) -> torch.Tensor:
    values = torch.randint(
        0, 2, (size,), generator=generator, device=device, dtype=torch.int64
    )
    result = values.mul(2).sub(1).to(dtype)
    return result / math.sqrt(size) if normalize else result


def hessian_vector_product(
    gradient: torch.Tensor,
    source: torch.Tensor,
    direction: torch.Tensor,
    *,
    retain_graph: bool,
) -> torch.Tensor:
    return torch.autograd.grad(
        gradient,
        source,
        grad_outputs=direction,
        retain_graph=retain_graph,
    )[0]


def energy_summary(
    total_squared: list[float], diagonal_squared: list[float], d_model: int
) -> dict[str, float]:
    if not total_squared or not diagonal_squared:
        raise ValueError("both total and diagonal samples are required")
    return weighted_energy_summary(
        total_squared,
        [d_model * squared for squared in diagonal_squared],
    )


def weighted_energy_summary(
    total_squared: list[float], diagonal_energy_samples: list[float]
) -> dict[str, float]:
    """Summarize samples that individually estimate full diagonal energy."""
    if not total_squared or not diagonal_energy_samples:
        raise ValueError("both total and diagonal samples are required")
    total = sum(total_squared) / len(total_squared)
    diagonal = sum(diagonal_energy_samples) / len(diagonal_energy_samples)
    return {
        "total_energy": total,
        "diagonal_energy": diagonal,
        "diagonal_fraction": diagonal / total if total > 0 else math.nan,
    }


def natural_approximation_summary(
    records: list[dict[str, float]],
) -> dict[str, float]:
    """Summarize projected full-versus-diagonal natural-direction curvature.

    Each record contains two independent coordinate-sampling estimates of the
    diagonal quadratic form.  Their product is an unbiased estimate of its
    squared magnitude; using the same noisy estimate twice would bias the
    diagonal energy upward.
    """
    if not records:
        raise ValueError("natural-direction records are required")
    full_energy = sum(record["full_value"] ** 2 for record in records) / len(
        records
    )
    diagonal_energy = sum(
        record["diagonal_split_a"] * record["diagonal_split_b"]
        for record in records
    ) / len(records)
    cross_energy = sum(
        record["full_value"] * record["diagonal_estimate"] for record in records
    ) / len(records)
    error_energy = full_energy - 2 * cross_energy + diagonal_energy
    return {
        "full_energy": full_energy,
        "diagonal_energy_unbiased": diagonal_energy,
        "full_diagonal_cross_energy": cross_energy,
        "approximation_error_energy_unbiased": error_energy,
        "relative_approximation_error": (
            error_energy / full_energy if full_energy > 0 else math.nan
        ),
        "explained_fraction": (
            1 - error_energy / full_energy if full_energy > 0 else math.nan
        ),
    }


def clean_activations(
    model: Any,
    input_ids: torch.Tensor,
    source_layers: list[int],
    target_layer: int,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    with torch.no_grad(), ActivationRecorder(
        model.layers, at=[*source_layers, target_layer]
    ) as recorder:
        model.forward(input_ids)
    target = recorder.activations[target_layer].detach()
    source_means = {
        layer: recorder.activations[layer].detach()[0].mean(dim=0)
        for layer in source_layers
    }
    return target, source_means


def projected_hessian_samples(
    model: Any,
    input_ids: torch.Tensor,
    *,
    source_layer: int,
    target_layer: int,
    position: int,
    clean_target: torch.Tensor,
    source_mean: torch.Tensor,
    output_probes: int,
    total_probes: int,
    diagonal_probes: int,
    symmetry_checks: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    d_model = model.d_model
    device = input_ids.device
    total_records: list[dict[str, Any]] = []
    diagonal_records: list[dict[str, Any]] = []
    natural_records: list[dict[str, float]] = []
    zero_intervention_errors: list[float] = []
    source_rms_values: list[float] = []
    started = time.perf_counter()

    for output_probe in range(output_probes):
        with ActivationRecorder(
            model.layers,
            at=[source_layer, target_layer],
            start_graph_at=source_layer,
        ) as recorder:
            model.forward(input_ids)
        source = recorder.activations[source_layer]
        target = recorder.activations[target_layer]
        if not source.is_leaf or not source.requires_grad:
            raise RuntimeError(
                "source activation is not the intended leaf autograd root"
            )
        zero_intervention_errors.append(
            float(torch.max(torch.abs(target.detach() - clean_target)).item())
        )
        source_rms_values.append(
            float(torch.sqrt(torch.mean(source.detach()[0, position].square())).item())
        )
        natural_delta = source.detach()[0, position] - source_mean
        natural_coordinate_mass = natural_delta.square()
        natural_coordinate_mass /= natural_coordinate_mass.sum()
        coordinate_probabilities = (
            (1 - NATURAL_IMPORTANCE_WEIGHT) / d_model
            + NATURAL_IMPORTANCE_WEIGHT * natural_coordinate_mass
        )

        output_direction = rademacher(
            d_model,
            generator=generator,
            device=device,
            dtype=target.dtype,
            normalize=True,
        )
        downstream_output = target[0, position:].mean(dim=0)
        scalar = torch.dot(downstream_output, output_direction)
        gradient = torch.autograd.grad(
            scalar, source, create_graph=True, retain_graph=True
        )[0]

        n_symmetry = min(symmetry_checks, total_probes)
        n_hvps = total_probes + n_symmetry + diagonal_probes + 1
        hvp_index = 0
        for probe_index in range(total_probes):
            left = rademacher(
                d_model,
                generator=generator,
                device=device,
                dtype=source.dtype,
            )
            right = rademacher(
                d_model,
                generator=generator,
                device=device,
                dtype=source.dtype,
            )
            direction = torch.zeros_like(source)
            direction[0, position] = left
            hvp_index += 1
            hvp = hessian_vector_product(
                gradient,
                source,
                direction,
                retain_graph=hvp_index < n_hvps,
            )
            value = float(torch.dot(hvp[0, position], right).item())
            record = {
                "output_probe": output_probe,
                "probe": probe_index,
                "value": value,
                "squared": value * value,
            }
            if probe_index < n_symmetry:
                swapped_direction = torch.zeros_like(source)
                swapped_direction[0, position] = right
                hvp_index += 1
                swapped_hvp = hessian_vector_product(
                    gradient,
                    source,
                    swapped_direction,
                    retain_graph=hvp_index < n_hvps,
                )
                swapped = float(torch.dot(swapped_hvp[0, position], left).item())
                absolute_error = abs(value - swapped)
                record.update(
                    {
                        "swapped_value": swapped,
                        "symmetry_absolute_error": absolute_error,
                        "symmetry_relative_error": absolute_error
                        / max(abs(value), abs(swapped), 1e-30),
                    }
                )
            total_records.append(record)

        for probe_index in range(diagonal_probes):
            coordinate = int(
                torch.multinomial(
                    coordinate_probabilities,
                    1,
                    replacement=True,
                    generator=generator,
                ).item()
            )
            probability = float(coordinate_probabilities[coordinate].item())
            direction = torch.zeros_like(source)
            direction[0, position, coordinate] = 1.0
            hvp_index += 1
            hvp = hessian_vector_product(
                gradient,
                source,
                direction,
                retain_graph=hvp_index < n_hvps,
            )
            value = float(hvp[0, position, coordinate].item())
            natural_weighted_value = (
                float(natural_delta[coordinate].square().item())
                * value
                / probability
            )
            diagonal_records.append(
                {
                    "output_probe": output_probe,
                    "probe": probe_index,
                    "coordinate": coordinate,
                    "probability": probability,
                    "value": value,
                    "squared": value * value,
                    "energy_weighted": value * value / probability,
                    "natural_weighted_value": natural_weighted_value,
                }
            )

        direction = torch.zeros_like(source)
        direction[0, position] = natural_delta
        hvp_index += 1
        hvp = hessian_vector_product(
            gradient,
            source,
            direction,
            retain_graph=hvp_index < n_hvps,
        )
        full_value = float(torch.dot(hvp[0, position], natural_delta).item())
        output_weighted = [
            record["natural_weighted_value"]
            for record in diagonal_records
            if record["output_probe"] == output_probe
        ]
        split_a = output_weighted[::2]
        split_b = output_weighted[1::2]
        natural_records.append(
            {
                "output_probe": output_probe,
                "full_value": full_value,
                "diagonal_estimate": sum(output_weighted) / len(output_weighted),
                "diagonal_split_a": sum(split_a) / len(split_a),
                "diagonal_split_b": sum(split_b) / len(split_b),
            }
        )

    summary = weighted_energy_summary(
        [record["squared"] for record in total_records],
        [record["energy_weighted"] for record in diagonal_records],
    )
    per_output = []
    for output_probe in range(output_probes):
        output_total = [
            record["squared"]
            for record in total_records
            if record["output_probe"] == output_probe
        ]
        output_diagonal = [
            record["energy_weighted"]
            for record in diagonal_records
            if record["output_probe"] == output_probe
        ]
        per_output.append(
            {
                "output_probe": output_probe,
                **weighted_energy_summary(output_total, output_diagonal),
            }
        )
    symmetry_errors = [
        record["symmetry_relative_error"]
        for record in total_records
        if "symmetry_relative_error" in record
    ]
    natural_summary = natural_approximation_summary(natural_records)
    return {
        "source_layer": source_layer,
        "target_layer": target_layer,
        "output_aggregation": "mean over positions probe_position..end",
        "output_direction": "normalized Rademacher (covariance I/d_model)",
        "input_total_directions": "independent unnormalized Rademacher",
        "input_diagonal_directions": (
            "coordinate basis vectors sampled from a 50/50 mixture of uniform "
            "and natural-displacement-squared importance probabilities"
        ),
        "output_probes": output_probes,
        "total_probes_per_output": total_probes,
        "diagonal_probes_per_output": diagonal_probes,
        "symmetry_checks_per_output": n_symmetry,
        "symmetry_max_relative_error": max(symmetry_errors, default=math.nan),
        "zero_intervention_max_abs_error": max(zero_intervention_errors),
        "source_activation_rms": sum(source_rms_values) / len(source_rms_values),
        "natural_displacement": {
            "definition": (
                "probe activation minus the mean activation over token positions "
                "in the same prompt"
            ),
            "rms": float(torch.sqrt(torch.mean(natural_delta.square())).item()),
            "coordinate_estimator": (
                "delta_j^2 * w^T H[e_j,e_j] / p(j), with mixture importance "
                "probability p"
            ),
            **natural_summary,
            "per_output_probe": natural_records,
        },
        "elapsed_seconds": time.perf_counter() - started,
        **summary,
        "per_output_probe": per_output,
        "total_samples": total_records,
        "diagonal_samples": diagonal_records,
    }


def evaluate_context(
    model: Any,
    tokenizer: Any,
    case: BaselineCase,
    *,
    layers: list[int],
    target_layer: int,
    output_probes: int,
    total_probes: int,
    diagonal_probes: int,
    symmetry_checks: int,
    seed: int,
) -> dict[str, Any]:
    tokenization = encode_case(tokenizer, case)
    input_ids = model.encode(case.prompt, max_length=512)
    if input_ids[0].tolist() != tokenization["input_ids"]:
        raise RuntimeError(f"tokenization mismatch for {case.id}")
    position = tokenization["probe_position"]
    clean_target, source_means = clean_activations(
        model, input_ids, layers, target_layer
    )
    layer_results = []
    for layer_offset, layer in enumerate(layers):
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(seed + 1009 * layer_offset)
        result = projected_hessian_samples(
            model,
            input_ids,
            source_layer=layer,
            target_layer=target_layer,
            position=position,
            clean_target=clean_target,
            source_mean=source_means[layer],
            output_probes=output_probes,
            total_probes=total_probes,
            diagonal_probes=diagonal_probes,
            symmetry_checks=symmetry_checks,
            generator=generator,
        )
        layer_results.append(result)
        print(
            f"{case.id} L{layer}->L{target_layer}: "
            f"diag/total={result['diagonal_fraction']:.3e} "
            f"zero_err={result['zero_intervention_max_abs_error']:.1e} "
            f"({result['elapsed_seconds']:.1f}s)",
            flush=True,
        )
    return {
        "id": case.id,
        "prompt": case.prompt,
        "probe_text": case.probe_text,
        "probe_position": position,
        "probe_token": tokenization["probe_token"],
        "layers": layer_results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=root / "configs" / "baseline_cases.json"
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--layer", action="append", type=int, default=[])
    parser.add_argument("--target-layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--output-probes", type=int, default=2)
    parser.add_argument("--total-probes", type=int, default=4)
    parser.add_argument("--diagonal-probes", type=int, default=8)
    parser.add_argument("--symmetry-checks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "curvature_energy_qwen3.5-4b.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    selected = set(args.case) if args.case else set(DEFAULT_CASES)
    cases = load_cases(args.cases, selected)
    layers = args.layer or list(DEFAULT_LAYERS)
    if not layers or any(layer < 0 or layer >= args.target_layer for layer in layers):
        raise ValueError("source layers must be nonempty and below the target layer")
    if min(args.output_probes, args.total_probes) < 1:
        raise ValueError("all probe counts must be positive")
    if args.diagonal_probes < 2:
        raise ValueError("--diagonal-probes must be at least 2")
    if args.symmetry_checks < 0:
        raise ValueError("--symmetry-checks must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

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

    results = [
        evaluate_context(
            model,
            tokenizer,
            case,
            layers=layers,
            target_layer=args.target_layer,
            output_probes=args.output_probes,
            total_probes=args.total_probes,
            diagonal_probes=args.diagonal_probes,
            symmetry_checks=args.symmetry_checks,
            seed=args.seed + 100_003 * case_index,
        )
        for case_index, case in enumerate(cases)
    ]
    payload = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_dtype": str(next(hf_model.parameters()).dtype),
            "target_layer": args.target_layer,
            "layers": layers,
            "seed": args.seed,
            "estimator": (
                "E[w,u,v](w^T H[u,v])^2 for total; "
                "d E[w,j](w^T H[e_j,e_j])^2 for diagonal"
            ),
        },
        "contexts": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
