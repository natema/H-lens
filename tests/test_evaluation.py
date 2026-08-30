from __future__ import annotations

import pytest
import torch
from torch import nn

from j2_lens.evaluation import (
    apply_diagonal_operator,
    apply_low_rank_diagonal,
    batched_forward_diagonal_curvature,
    valid_positions,
)


def test_valid_positions_matches_j_lens_reduction() -> None:
    assert valid_positions(seq_len=9, skip_first=4) == [4, 5, 6, 7]


def test_valid_positions_rejects_short_prompt() -> None:
    with pytest.raises(ValueError, match="no positions"):
        valid_positions(seq_len=5, skip_first=4)


def test_apply_low_rank_diagonal() -> None:
    outputs = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    features = torch.tensor([3.0, 5.0])
    result = apply_low_rank_diagonal(outputs, inputs, features)
    assert torch.equal(result, torch.tensor([3.0, 10.0]))


def test_apply_direct_coordinate_diagonal_and_shuffle() -> None:
    operator = {
        "coordinates": torch.tensor([0, 2]),
        "diagonal_rows": torch.tensor([[2.0, 0.0], [0.0, 4.0]]),
    }
    features = torch.tensor([3.0, 99.0, 5.0])
    assert torch.equal(
        apply_diagonal_operator(operator, features), torch.tensor([6.0, 20.0])
    )
    assert torch.equal(
        apply_diagonal_operator(operator, features, shuffled=True),
        torch.tensor([10.0, 12.0]),
    )


class _IdentityLayer(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _CubeLayer(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden**3


class _ToyForwardModel:
    d_model = 2

    def __init__(self) -> None:
        self.layers = nn.ModuleList([_IdentityLayer(), _CubeLayer()])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = values
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def test_forward_over_forward_returns_vector_hessian_diagonal() -> None:
    model = _ToyForwardModel()
    values = torch.tensor(
        [
            [[2.0, 3.0], [5.0, 7.0], [0.0, 0.0]],
            [[11.0, 13.0], [17.0, 19.0], [0.0, 0.0]],
        ]
    )
    positions = torch.tensor([0, 1])
    directions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = batched_forward_diagonal_curvature(
        model,
        values,
        source_layer=0,
        target_layer=1,
        positions=positions,
        directions=directions,
    )
    assert torch.equal(result, torch.tensor([[12.0, 0.0], [0.0, 114.0]]))
