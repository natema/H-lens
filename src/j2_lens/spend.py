"""API cost accounting.

Kept apart from the pipeline modules because it is about billing, not about
probes, concepts or lenses, and because the pricing table has to be revised
whenever a provider changes rates.

Token counts are exact — the API returns them per call. Cost is derived: there is
no billing endpoint (``/v1/usage``, ``/v1/billing``, ``/v1/credits`` all 404), so
it is tokens times the rates documented at ``docs.mistral.ai/models/<id>``.
``console.mistral.ai`` remains authoritative.

The budget is a fact about an account rather than about this code, so it is read
from ``J2_BUDGET_EUR`` and omitted from the ledger when unset.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def budget_eur() -> float | None:
    """Declared budget, or None when the environment does not name one."""
    raw = os.environ.get("J2_BUDGET_EUR")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None



PRICING_USD_PER_MTOK = {
    # docs.mistral.ai/models/mistral-large
    "mistral-large-latest": (0.5, 0.5, 1.5),
    "mistral-large-2512": (0.5, 0.5, 1.5),
    # docs.mistral.ai/models/zai-glm-5-2
    "glm-5-2": (1.4, 0.14, 4.4),
    "zai-glm-5-2": (1.4, 0.14, 4.4),
}
def call_cost_usd(model: str, usage: dict[str, Any]) -> float | None:
    """Cost of one call, or None when the model has no documented rate.

    Returning None rather than guessing keeps an unpriced model visible in the
    ledger instead of silently contributing zero to the running total.
    """
    rates = PRICING_USD_PER_MTOK.get(model)
    if rates is None or not usage:
        return None
    rate_in, rate_cached, rate_out = rates
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    fresh = max(0, usage.get("prompt_tokens", 0) - cached)
    return (
        fresh * rate_in
        + cached * rate_cached
        + usage.get("completion_tokens", 0) * rate_out
    ) / 1_000_000
def record_spend(
    ledger: Path, exchange: dict[str, Any], note: str = ""
) -> dict[str, Any]:
    """Append one API call to a running cost ledger and return the totals.

    Every paid call goes through here so the budget is tracked from measured
    token counts rather than estimated after the fact.
    """

    entries = []
    if ledger.exists():
        entries = json.loads(ledger.read_text())["calls"]
    usage = exchange.get("usage") or {}
    model = exchange.get("model") or exchange["request"]["model"]
    entries.append(
        {
            "at": datetime.now(UTC).isoformat(),
            "model": model,
            "note": note,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens", 0
            ),
            "completion_tokens": usage.get("completion_tokens"),
            "cost_usd": call_cost_usd(model, usage),
            "priced": model in PRICING_USD_PER_MTOK,
        }
    )
    total = sum(e["cost_usd"] or 0.0 for e in entries)
    unpriced = sorted({e["model"] for e in entries if not e.get("priced")})
    payload = {
        "calls": entries,
        "totals": {
            "n_calls": len(entries),
            "prompt_tokens": sum(e["prompt_tokens"] or 0 for e in entries),
            "completion_tokens": sum(e["completion_tokens"] or 0 for e in entries),
            "cached_tokens": sum(e.get("cached_tokens") or 0 for e in entries),
            "cost_usd": round(total, 6),
            "unpriced_models": unpriced,
            "budget_eur": budget_eur(),
            "note": (
                "Token counts are exact, returned by the API per call. The "
                "Mistral API exposes no billing endpoint, so cost is tokens "
                "times the rates documented at docs.mistral.ai/models/<id>: "
                "mistral-large 0.5 in / 1.5 out, zai-glm-5-2 1.4 in / 0.14 "
                "cached in / 4.4 out, USD per million tokens. Cached input is "
                "billed separately at its lower rate. Any model listed in "
                "unpriced_models contributes nothing to cost_usd and needs a "
                "rate added. console.mistral.ai remains authoritative. No "
                "USD-to-EUR conversion is applied."
            ),
        },
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps(payload, indent=2) + chr(10))
    return payload["totals"]
