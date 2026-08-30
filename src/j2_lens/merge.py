"""Merge coordinate shards of a forward-mode diagonal into one operator.

Coordinates are independent, so a long fit is split into disjoint ranges run
concurrently on separate GPUs (``--coordinate-offset`` / ``--max-coordinates``
with ``--fit-only``) and recombined here.

The merge refuses to combine shards that were not fitted under identical
conditions, and requires the union of their coordinate ranges to be an exact,
gap-free, duplicate-free cover of the coordinates it claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

# Metadata that must agree across shards for the merge to be meaningful.
SHARED_KEYS = (
    "model_id",
    "model_revision",
    "lens_revision",
    "effective_estimator",
    "quadratic_coefficient",
    "n_hessian_pairs",
    "n_moment_pairs",
    "development_case_ids",
)


def merge_shards(
    paths: list[Path], expect_d_model: int | None = None
) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("merging needs at least two shards")
    shards = [
        torch.load(path, map_location="cpu", weights_only=False) for path in paths
    ]
    reference = shards[0]["metadata"]
    for path, shard in zip(paths[1:], shards[1:], strict=True):
        mismatched = [
            key
            for key in SHARED_KEYS
            if shard["metadata"].get(key) != reference.get(key)
        ]
        if mismatched:
            raise ValueError(f"{path} disagrees on {', '.join(mismatched)}")

    layer_ids = {layer for shard in shards for layer in shard["layers"]}
    if len(layer_ids) != 1:
        raise ValueError(f"shards cover different layers: {sorted(layer_ids)}")
    layer = layer_ids.pop()

    entries = [shard["layers"][layer] for shard in shards]
    for path, entry in zip(paths[1:], entries[1:], strict=True):
        for key in ("source_mean", "source_variance", "target_mean"):
            if not torch.allclose(entry[key], entries[0][key], atol=1e-6):
                raise ValueError(f"{path} has different development {key}")

    coordinates = torch.cat([entry["coordinates"] for entry in entries])
    rows = torch.cat([entry["diagonal_rows"] for entry in entries], dim=0)
    order = torch.argsort(coordinates)
    coordinates, rows = coordinates[order], rows[order]

    unique = torch.unique(coordinates)
    if unique.numel() != coordinates.numel():
        raise ValueError("shards overlap: duplicate coordinates")
    expected = torch.arange(int(coordinates[0]), int(coordinates[-1]) + 1)
    if not torch.equal(coordinates, expected):
        raise ValueError("shards leave gaps in the coordinate range")
    if expect_d_model is not None and coordinates.numel() != expect_d_model:
        raise ValueError(
            f"merged {coordinates.numel()} coordinates, expected {expect_d_model}"
        )
    if not torch.isfinite(rows).all():
        raise RuntimeError("merged diagonal contains non-finite values")

    metadata = dict(reference)
    metadata["effective_coordinate_offset"] = int(coordinates[0])
    metadata["effective_max_coordinates"] = int(coordinates.numel())
    metadata["merged_from"] = [str(path) for path in paths]
    return {
        "metadata": metadata,
        "layers": {
            layer: {
                "coordinates": coordinates,
                "diagonal_rows": rows,
                "source_mean": entries[0]["source_mean"],
                "source_variance": entries[0]["source_variance"],
                "target_mean": entries[0]["target_mean"],
                "development_summary": entries[0]["development_summary"],
                "samples": [s for entry in entries for s in entry["samples"]],
                "elapsed_seconds": sum(
                    float(entry["elapsed_seconds"]) for entry in entries
                ),
            }
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-d-model", type=int, default=2560)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    merged = merge_shards(args.shards, args.expect_d_model)
    layer = next(iter(merged["layers"]))
    entry = merged["layers"][layer]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(merged, temporary)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "layer": layer,
                "coordinates": int(entry["coordinates"].numel()),
                "diagonal_shape": list(entry["diagonal_rows"].shape),
                "total_fit_seconds": round(entry["elapsed_seconds"]),
                "shards": len(args.shards),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
