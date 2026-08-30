"""Screen candidate evaluation cases before they are frozen into the battery.

The screen is purely mechanical and uses no lens output, so it cannot select
cases on the basis of J-lens, R-lens, or H-lens performance:

1. the target text must be a single token of the pinned tokenizer;
2. the probe span must resolve to exactly one token position;
3. for the ``multihop`` and ``multilingual`` categories the model must answer
   the surface question correctly, matching the R-lens filtering rule.

Accepted cases are written to a frozen battery file in the schema that
``j2_lens.baselines.BaselineCase`` accepts, with the screening-only fields
(``expected_answer``, ``source``) dropped.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from j2_lens.baselines import (
    MODEL_ID,
    MODEL_REVISION,
    BaselineCase,
    encode_case,
)

ANSWERABLE_CATEGORIES = ("multihop", "multilingual")
CASE_FIELDS = tuple(field.name for field in fields(BaselineCase))


def greedy_continuation(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int, device: str
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def screen_candidate(
    candidate: dict[str, Any],
    tokenizer: Any,
    hf_model: Any,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    case_kwargs = {key: candidate[key] for key in CASE_FIELDS}
    case = BaselineCase(**case_kwargs)
    report: dict[str, Any] = {
        "id": case.id,
        "category": case.category,
        "source": candidate.get("source"),
        "accepted": False,
        "rejection_reasons": [],
    }

    try:
        tokenization = encode_case(tokenizer, case)
    except ValueError as error:
        report["rejection_reasons"].append(f"tokenization: {error}")
        return report

    report["probe_position"] = tokenization["probe_position"]
    report["probe_token"] = tokenization["probe_token"]["decoded"]
    report["span_token_indices"] = tokenization["span_token_indices"]
    report["target_token"] = tokenization["target_token"]["decoded"]
    report["n_prompt_tokens"] = len(tokenization["input_ids"])

    # The target must not already be readable off the prompt surface: a probe
    # that sits after the answer would test nothing about early-layer transport.
    if case.target_text.strip().lower() in case.prompt.lower():
        report["rejection_reasons"].append("target string occurs in the prompt")

    expected = candidate.get("expected_answer")
    if case.category in ANSWERABLE_CATEGORIES and hf_model is None:
        report["rejection_reasons"].append(
            "answerability not checked (model not loaded)"
        )
    elif case.category in ANSWERABLE_CATEGORIES:
        if not expected:
            report["rejection_reasons"].append(
                "answerable category without expected_answer"
            )
        else:
            continuation = greedy_continuation(
                hf_model, tokenizer, case.prompt, max_new_tokens, device
            )
            report["continuation"] = continuation
            report["expected_answer"] = expected
            report["answered_correctly"] = (
                expected.lower() in continuation.lower()
            )
            if not report["answered_correctly"]:
                report["rejection_reasons"].append(
                    f"model answered {continuation!r}, expected {expected!r}"
                )

    report["accepted"] = not report["rejection_reasons"]
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=root / "configs/candidate_cases.json"
    )
    parser.add_argument(
        "--battery", type=Path, default=root / "configs/battery_cases.json"
    )
    parser.add_argument(
        "--report", type=Path, default=root / "results/screening_qwen3.5-4b.json"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run tokenizer-only checks and skip loading the model",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = json.loads(args.candidates.read_text())
    candidates = payload["candidates"]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=args.offline
    )
    hf_model = None
    if not args.dry_run:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            local_files_only=args.offline,
        ).to(args.device)

    reports = []
    for candidate in candidates:
        report = screen_candidate(
            candidate,
            tokenizer,
            hf_model,
            args.max_new_tokens,
            args.device,
        )
        reports.append(report)
        status = "accept" if report["accepted"] else "REJECT"
        print(
            f"{status} {report['id']:<28} probe={report.get('probe_token')!r:<14} "
            f"target={report.get('target_token')!r} "
            + ("; ".join(report["rejection_reasons"])),
            flush=True,
        )

    accepted_ids = {report["id"] for report in reports if report["accepted"]}
    battery = {
        "schema_version": 1,
        "selection_rule": payload["selection_rule"],
        "cases": [
            {key: candidate[key] for key in CASE_FIELDS}
            for candidate in candidates
            if candidate["id"] in accepted_ids
        ],
    }
    args.battery.parent.mkdir(parents=True, exist_ok=True)
    args.battery.write_text(json.dumps(battery, indent=2) + "\n")

    report_payload = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "device": None if args.dry_run else args.device,
            "dry_run": args.dry_run,
            "max_new_tokens": args.max_new_tokens,
            "n_candidates": len(candidates),
            "n_accepted": len(accepted_ids),
        },
        "candidates": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report_payload, indent=2) + "\n")
    print(
        f"Accepted {len(accepted_ids)}/{len(candidates)}; "
        f"wrote {args.battery} and {args.report}",
        flush=True,
    )

if __name__ == "__main__":
    main()
