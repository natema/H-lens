from __future__ import annotations

import pytest
import torch
from torch import nn

from j2_lens.evaluation import (
    apply_diagonal_operator,
    apply_low_rank_diagonal,
    batched_forward_diagonal_curvature,
    reduce_target_sums,
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
        ends=torch.tensor([2, 2]),
    )
    assert torch.equal(result, torch.tensor([[12.0, 0.0], [0.0, 114.0]]))


def test_reduce_target_sums_respects_per_row_end() -> None:
    target = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [8.0, 8.0]],
            [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [8.0, 8.0]],
        ]
    )
    result = reduce_target_sums(
        target, positions=torch.tensor([0, 1]), ends=torch.tensor([2, 4])
    )
    # Row 0 stops at its own penultimate token; row 1 runs further.
    assert torch.equal(result, torch.tensor([[3.0, 3.0], [14.0, 14.0]]))


class _CausalMixLayer(nn.Module):
    """Crude causal mixer: position i depends only on positions <= i."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(hidden, dim=1)


class _ToyCausalModel:
    d_model = 2

    def __init__(self) -> None:
        self.layers = nn.ModuleList([_IdentityLayer(), _CausalMixLayer(), _CubeLayer()])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = values
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def test_right_padding_does_not_change_mixed_length_curvature() -> None:
    """A padded mixed-length batch must equal per-prompt unpadded results.

    This is the property that lets the development set contain prompts of
    different lengths: under causal mixing, a real position never sees the
    padding, and each row's reduction stops at its own penultimate token.
    """
    model = _ToyCausalModel()
    short = torch.tensor([[[2.0, 3.0], [5.0, 7.0], [1.0, 4.0]]])
    long = torch.tensor(
        [[[11.0, 13.0], [17.0, 19.0], [23.0, 29.0], [31.0, 37.0], [41.0, 43.0]]]
    )
    kwargs = {"source_layer": 0, "target_layer": 2}

    separate = [
        batched_forward_diagonal_curvature(
            model,
            values,
            positions=torch.tensor([1]),
            directions=torch.tensor([[1.0, 0.0]]),
            ends=torch.tensor([values.shape[1] - 1]),
            **kwargs,
        )
        for values in (short, long)
    ]

    width = long.shape[1]
    padded = torch.zeros(2, width, 2)
    padded[0, : short.shape[1]] = short[0]
    padded[1] = long[0]
    together = batched_forward_diagonal_curvature(
        model,
        padded,
        positions=torch.tensor([1, 1]),
        directions=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        ends=torch.tensor([short.shape[1] - 1, long.shape[1] - 1]),
        **kwargs,
    )

    assert torch.allclose(together[0], separate[0][0])
    assert torch.allclose(together[1], separate[1][0])


def test_padding_value_cannot_affect_the_result() -> None:
    model = _ToyCausalModel()
    base = torch.tensor([[[2.0, 3.0], [5.0, 7.0], [1.0, 4.0], [0.0, 0.0]]])
    noisy = base.clone()
    noisy[0, 3] = torch.tensor([99.0, -50.0])
    kwargs = {
        "source_layer": 0,
        "target_layer": 2,
        "positions": torch.tensor([1]),
        "directions": torch.tensor([[1.0, 0.0]]),
        "ends": torch.tensor([2]),
    }
    quiet = batched_forward_diagonal_curvature(model, base, **kwargs)
    loud = batched_forward_diagonal_curvature(model, noisy, **kwargs)
    assert torch.equal(quiet, loud)
