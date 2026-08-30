from __future__ import annotations

import pytest

from j2_lens.curvature import (
    energy_summary,
    natural_approximation_summary,
    weighted_energy_summary,
)


def test_energy_summary_applies_coordinate_sampling_factor() -> None:
    summary = energy_summary(
        total_squared=[8.0, 12.0], diagonal_squared=[1.0, 3.0], d_model=4
    )
    assert summary["total_energy"] == 10.0
    assert summary["diagonal_energy"] == 8.0
    assert summary["diagonal_fraction"] == pytest.approx(0.8)


def test_energy_summary_rejects_missing_samples() -> None:
    with pytest.raises(ValueError, match="required"):
        energy_summary([], [1.0], d_model=2)


def test_weighted_energy_summary_averages_importance_samples() -> None:
    summary = weighted_energy_summary(
        total_squared=[8.0, 12.0], diagonal_energy_samples=[4.0, 12.0]
    )
    assert summary["total_energy"] == 10.0
    assert summary["diagonal_energy"] == 8.0
    assert summary["diagonal_fraction"] == pytest.approx(0.8)


def test_natural_approximation_summary_uses_independent_splits() -> None:
    summary = natural_approximation_summary(
        [
            {
                "full_value": 2.0,
                "diagonal_estimate": 2.0,
                "diagonal_split_a": 1.0,
                "diagonal_split_b": 3.0,
            },
            {
                "full_value": 4.0,
                "diagonal_estimate": 4.0,
                "diagonal_split_a": 2.0,
                "diagonal_split_b": 6.0,
            },
        ]
    )
    assert summary["full_energy"] == 10.0
    assert summary["diagonal_energy_unbiased"] == 7.5
    assert summary["full_diagonal_cross_energy"] == 10.0
    assert summary["approximation_error_energy_unbiased"] == -2.5
    assert summary["explained_fraction"] == 1.25


def test_natural_approximation_summary_rejects_missing_records() -> None:
    with pytest.raises(ValueError, match="required"):
        natural_approximation_summary([])
