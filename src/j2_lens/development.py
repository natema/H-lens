"""Development corpus for estimating the activation moments and the Hessian.

The published J-lens was fitted on generic pretraining text, not on the
evaluation prompts: its provenance records ``dataset_id NeelNanda/pile-10k``,
25 documents, ``t_max 128``, ``skip_first 4``. Estimating the second-order
correction on the same distribution keeps the two operators comparable and
leaves every case in the evaluation battery held out.

The corpus is read straight from the dataset repository's single parquet file,
so it needs no extra dataset runtime. Documents are taken in file order, which
is deterministic and independent of any lens output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

PILE_REPO_ID = "NeelNanda/pile-10k"
PILE_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
PILE_FILE = "data/train-00000-of-00001-4746b8785c874cc7.parquet"

# Matches the published J-lens fitting provenance.
DEFAULT_N_DOCS = 25
DEFAULT_T_MAX = 128


def load_pile_documents(
    n_docs: int = DEFAULT_N_DOCS, *, offline: bool = False
) -> list[str]:
    """Return the first ``n_docs`` documents in dataset file order."""
    path = Path(
        hf_hub_download(
            PILE_REPO_ID,
            PILE_FILE,
            repo_type="dataset",
            revision=PILE_REVISION,
            local_files_only=offline,
        )
    )
    table = pq.read_table(path, columns=["text"])
    if table.num_rows < n_docs:
        raise ValueError(f"{PILE_REPO_ID} has only {table.num_rows} documents")
    return [str(value) for value in table["text"][:n_docs].to_pylist()]


def build_development_cases(
    documents: list[str],
    tokenizer: Any,
    *,
    t_max: int = DEFAULT_T_MAX,
    min_tokens: int = 8,
) -> list[dict[str, Any]]:
    """Truncate each document to ``t_max`` tokens and keep the decoded text.

    Truncation is applied in token space so the prompt length is exactly the
    published ``t_max`` budget; the decoded text is what the evaluator later
    re-encodes, and the caller checks that the round trip is stable.
    """
    cases = []
    for index, document in enumerate(documents):
        token_ids = tokenizer(
            document, truncation=True, max_length=t_max, add_special_tokens=False
        )["input_ids"]
        if len(token_ids) < min_tokens:
            continue
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        if not text.strip():
            continue
        cases.append(
            {
                "id": f"pile_{index:04d}",
                "category": "development",
                "prompt": text,
                "n_tokens_requested": len(token_ids),
            }
        )
    if not cases:
        raise ValueError("no usable development documents")
    return cases
