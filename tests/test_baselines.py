from __future__ import annotations

import pytest
import torch

from j2_lens.baselines import nth_span, rank_and_topk, resolve_probe


def test_nth_span_uses_zero_indexed_occurrences() -> None:
    text = "the first and the second and the third"
    assert nth_span(text, "the", 2) == (29, 32)


def test_nth_span_rejects_missing_occurrence() -> None:
    with pytest.raises(ValueError, match="not found"):
        nth_span("one one", "one", 2)


def test_resolve_probe_can_select_final_subtoken() -> None:
    prompt = "The vote was aganst"
    offsets = [(0, 3), (4, 8), (9, 12), (13, 15), (15, 19)]
    char_span, span_tokens, position = resolve_probe(
        prompt, "aganst", 0, -1, offsets
    )
    assert char_span == (13, 19)
    assert span_tokens == [3, 4]
    assert position == 4


def test_rank_and_topk_uses_one_based_competition_rank() -> None:
    logits = torch.tensor([0.0, 3.0, 1.0, 2.0])
    rank, target_logit, top_ids, top_logits = rank_and_topk(
        logits, target_id=2, top_k=3
    )
    assert rank == 3
    assert target_logit == 1.0
    assert top_ids == [1, 3, 2]
    assert top_logits == [3.0, 2.0, 1.0]
