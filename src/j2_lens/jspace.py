"""Build and validate a J-space dataset of probe/concept pairs.

An item is a sentence, a probe term inside it, and a target concept. The item
is useful only if the concept is genuinely evoked at the probe position, so two
independent checks are applied and must agree:

1. **J-lens check** — is the concept token in the top-k of the J-lens readout at
   the probe position, at any layer?
2. **Self-report check** — asked directly, does the model itself list the
   concept among the associations of that term in that context?

Causality is what makes this well posed. Attention is causal, so the residual at
the probe position depends only on the tokens up to and including the probe.
Everything after the probe is irrelevant to the lens and must therefore also be
hidden from the self-report question, or the two checks would not be asking
about the same thing. The prefix is the item; the rest of the sentence is
decoration. A probe at the very start of a sentence has no context at all and is
rejected.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
GENERATOR_MODEL = "glm-5-2"

# USD per million tokens, (input, output), from mistral.ai/pricing.
PRICING_USD_PER_MTOK = {
    # mistral.ai/pricing
    "mistral-large-latest": (0.5, 1.5),
    "mistral-large-2512": (0.5, 1.5),
    # GLM-5.2 served through the Mistral API. Mistral does not list a rate for
    # it on its own pricing page; this is a third-party aggregator figure and
    # should be treated as approximate.
    "glm-5-2": (1.54, 4.84),
}
BUDGET_EUR = 100.0


def call_cost_usd(model: str, usage: dict[str, Any]) -> float | None:
    rates = PRICING_USD_PER_MTOK.get(model)
    if rates is None or not usage:
        return None
    rate_in, rate_out = rates
    return (
        usage.get("prompt_tokens", 0) * rate_in
        + usage.get("completion_tokens", 0) * rate_out
    ) / 1_000_000


def record_spend(
    ledger: Path, exchange: dict[str, Any], note: str = ""
) -> dict[str, Any]:
    """Append one API call to a running cost ledger and return the totals.

    Every paid call goes through here so the budget is tracked from measured
    token counts rather than estimated after the fact.
    """
    from datetime import UTC, datetime

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
            "completion_tokens": usage.get("completion_tokens"),
            "cost_usd": call_cost_usd(model, usage),
        }
    )
    total = sum(e["cost_usd"] or 0.0 for e in entries)
    payload = {
        "calls": entries,
        "totals": {
            "n_calls": len(entries),
            "prompt_tokens": sum(e["prompt_tokens"] or 0 for e in entries),
            "completion_tokens": sum(e["completion_tokens"] or 0 for e in entries),
            "cost_usd": round(total, 6),
            "budget_eur": BUDGET_EUR,
            "note": (
                "Token counts are exact, returned by the API per call. Cost is "
                "derived: the Mistral API exposes no billing endpoint, so it is "
                "tokens times published rates. Mistral Large (0.5/1.5 per Mtok) "
                "comes from mistral.ai/pricing; glm-5-2 (1.54/4.84) is not on "
                "Mistral's own pricing page and comes from a third-party "
                "aggregator, so treat it as approximate. The console at "
                "console.mistral.ai is the authoritative spend figure. No "
                "USD-to-EUR conversion is applied."
            ),
        },
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps(payload, indent=2) + chr(10))
    return payload["totals"]

GENERATOR_SYSTEM = """\
You build evaluation items for a mechanistic-interpretability experiment on a \
language model.

Background. A "lens" reads out, at one token position inside a text, which \
concepts the model is currently representing. Attention is causal: the model's \
state at a token depends ONLY on that token and the text BEFORE it. Text after \
the probe token cannot influence the readout and is ignored entirely.

Your job. Given a target concept, invent a natural sentence containing a probe \
term, such that everything up to and including the probe term makes a competent \
reader think of the target concept.

Hard requirements:
- The probe term must NOT be the first word. It needs preceding context to do \
any work.
- The target concept must NOT appear anywhere in the text up to and including \
the probe term. The concept has to be evoked, never stated.
- The evocation must come from the text BEFORE the probe term plus the probe \
term itself. Never rely on what follows; it will be deleted.
- The target concept must be a single common English word, lowercase.
- The probe term must be a single word that appears verbatim exactly once.
- Prefer specific, concrete associations a model would plausibly encode.

Answer with JSON only."""


def concept_variants(concept: str) -> list[str]:
    """Surface forms of a concept that could carry it in the vocabulary.

    A concept is a token, and which token depends on casing: the model holds
    Japan as " Japan", not " japan". Scoring only the lowercase form would
    report a concept as absent while the readout is full of it, so all plausible
    single-token forms are considered and the best-ranked one is used.
    """
    bare = concept.strip()
    forms = [bare, bare.lower(), bare.capitalize(), bare.upper()]
    out: list[str] = []
    for form in forms:
        for surface in (f" {form}", form):
            if surface not in out:
                out.append(surface)
    return out


def single_token_variants(concept: str, tokenizer: Any) -> list[tuple[str, int]]:
    """Variants that are exactly one token, with their ids."""
    found: list[tuple[str, int]] = []
    seen: set[int] = set()
    for surface in concept_variants(concept):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1 and int(ids[0]) not in seen:
            seen.add(int(ids[0]))
            found.append((surface, int(ids[0])))
    return found


@dataclass(frozen=True)
class JSpaceItem:
    concept: str
    probe_term: str
    sentence: str
    rationale: str

    @property
    def prefix(self) -> str:
        """The sentence truncated immediately after the probe term.

        This is the whole item as far as the model is concerned: the lens
        readout at the probe and the self-report question must both see exactly
        this and nothing more.
        """
        match = re.search(rf"\b{re.escape(self.probe_term)}\b", self.sentence)
        if match is None:
            raise ValueError(f"probe {self.probe_term!r} not in sentence")
        return self.sentence[: match.end()]


def load_api_key(env_path: Path) -> str:
    key = os.environ.get("MISTRAL_API_KEY")
    if key:
        return key
    for line in env_path.read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "MISTRAL_API_KEY":
            return value.strip().strip("'\"")
    raise RuntimeError(f"MISTRAL_API_KEY not in environment or {env_path}")


def call_mistral(
    api_key: str, messages: list[dict[str, str]], *, max_tokens: int = 1200
) -> dict[str, Any]:
    payload = {
        "model": GENERATOR_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        MISTRAL_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Mistral {error.code}: {error.read()[:400]!r}") from error
    return {
        "request": payload,
        "content": body["choices"][0]["message"]["content"],
        "usage": body.get("usage"),
        "model": body.get("model"),
    }


def generate_items(
    api_key: str, concepts: list[str]
) -> tuple[list[JSpaceItem], dict[str, Any]]:
    user = (
        "Build one item for each of these target concepts: "
        + ", ".join(repr(concept) for concept in concepts)
        + '.\n\nReturn {"items": [{"concept": ..., "probe_term": ..., '
        '"sentence": ..., "rationale": ...}]}. The rationale states, in one '
        "sentence, why the text up to and including the probe term evokes the "
        "concept."
    )
    exchange = call_mistral(
        api_key,
        [
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    payload = json.loads(exchange["content"])
    items = [
        JSpaceItem(
            concept=item["concept"].strip(),
            probe_term=item["probe_term"].strip(),
            sentence=item["sentence"].strip(),
            rationale=item.get("rationale", "").strip(),
        )
        for item in payload["items"]
    ]
    return items, exchange


def structural_problems(item: JSpaceItem, tokenizer: Any) -> list[str]:
    """Mechanical checks that need no model forward pass."""
    problems: list[str] = []
    occurrences = len(
        re.findall(rf"\b{re.escape(item.probe_term)}\b", item.sentence)
    )
    if occurrences == 0:
        return [f"probe {item.probe_term!r} absent from the sentence"]
    if occurrences > 1:
        problems.append(f"probe {item.probe_term!r} occurs {occurrences} times")

    prefix = item.prefix
    before = prefix[: prefix.rfind(item.probe_term)].strip()
    if not before:
        problems.append("probe term is at the start; the prefix has no context")
    if re.search(rf"\b{re.escape(item.concept)}", prefix, flags=re.IGNORECASE):
        problems.append(f"concept {item.concept!r} is stated in the prefix")

    if not single_token_variants(item.concept, tokenizer):
        problems.append(
            f"no single-token form of {item.concept!r} "
            f"(tried {concept_variants(item.concept)})"
        )
    return problems


THINK_END = "</think>"


def strip_reasoning(text: str) -> str | None:
    """Return the answer after Qwen3.5's reasoning block, or None.

    Returning None when the block never closes matters: the reasoning text is
    full of ordinary words, so parsing it would silently manufacture a concept
    list out of the model's scratchpad instead of its answer.
    """
    if THINK_END not in text:
        return None
    return text.rsplit(THINK_END, 1)[-1].strip()


def parse_concept_list(
    text: str, limit: int, *, expect_reasoning: bool = False
) -> list[str] | None:
    """Parse the model's concept list, or None if it never produced an answer.

    With thinking disabled the closed ``<think></think>`` block sits in the
    prompt, so the generated text is the answer and contains no ``</think>``.
    With thinking enabled the block must close in the generated text, and if it
    does not there is no answer to parse.
    """
    answer = strip_reasoning(text) if expect_reasoning else text.strip()
    if answer is None:
        return None
    words: list[str] = []
    for line in answer.splitlines():
        cleaned = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
        cleaned = cleaned.strip("*_`\"'.,;:()[]")
        if not cleaned:
            continue
        first = cleaned.split()[0].strip("*_`\"'.,;:()[]").lower()
        if first and first not in words:
            words.append(first)
        if len(words) >= limit:
            break
    return words


SELF_REPORT_TEMPLATE = """\
Here is the beginning of a sentence. It stops immediately after the word \
"{term}".

{prefix}

What are the first {limit} concepts you think about when you encounter the \
word "{term}" in the sentence above? Include the things it refers to or is an \
instance of, not only its qualities.

Reply with {limit} single lowercase words, one per line, nothing else."""


def self_report_concepts(
    model: Any,
    tokenizer: Any,
    item: JSpaceItem,
    *,
    limit: int,
    max_new_tokens: int,
    enable_thinking: bool = False,
) -> tuple[list[str] | None, str]:
    """Ask the model itself which concepts the prefix evokes at the probe.

    The question shows only ``item.prefix`` — the same tokens the lens readout
    depends on — so the two checks are asking about the same model state.

    Thinking is off by default: the reasoning block routinely runs past any
    sane token budget, and a truncated block yields no answer at all.
    """
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": SELF_REPORT_TEMPLATE.format(
                    term=item.probe_term, prefix=item.prefix, limit=limit
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    import torch

    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(
        generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return parse_concept_list(raw, limit, expect_reasoning=enable_thinking), raw


def lens_readout(
    item: JSpaceItem,
    model: Any,
    tokenizer: Any,
    lenses: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Rank the concept in each lens's readout at the probe position.

    The lens runs on the prefix, whose last token is the probe. Because
    attention is causal this is identical to running on the full sentence and
    reading at the probe position; ``verify_causal_equivalence`` checks that.
    """
    from j2_lens.baselines import describe_tokens, summarize_layers

    encoded = tokenizer(item.prefix, return_offsets_mapping=True)
    input_ids = [int(i) for i in encoded["input_ids"]]
    position = len(input_ids) - 1
    variants = single_token_variants(item.concept, tokenizer)
    if not variants:
        raise ValueError(f"concept {item.concept!r} has no single-token form")

    readouts = {}
    for name in ("j_lens", "r_lens"):
        layer_logits, _, ids = lenses[name].apply(
            model, item.prefix, positions=[position]
        )
        if ids[0].tolist() != input_ids:
            raise RuntimeError(f"tokenization mismatch for {item.concept}")
        readouts[name] = layer_logits
    readouts["logit_lens"] = lenses["j_lens"].apply(
        model, item.prefix, positions=[position], use_jacobian=False
    )[0]

    methods: dict[str, Any] = {}
    for name, layer_logits in readouts.items():
        scored = {
            surface: summarize_layers(layer_logits, token_id, top_k, tokenizer)
            for surface, token_id in variants
        }
        best_surface = min(scored, key=lambda s: scored[s]["best_rank"])
        methods[name] = {
            **scored[best_surface],
            "best_variant": best_surface,
            "per_variant_best_rank": {
                s: v["best_rank"] for s, v in scored.items()
            },
        }
    return {
        "probe_position": position,
        "probe_token": describe_tokens(tokenizer, [input_ids[position]])[0],
        "concept_variants": [s for s, _ in variants],
        "n_prefix_tokens": len(input_ids),
        "methods": methods,
    }


def verify_causal_equivalence(
    item: JSpaceItem, model: Any, tokenizer: Any, lens: Any
) -> float:
    """Relative deviation between reading the prefix and the full sentence.

    Should be zero up to floating-point noise, since a causal model cannot let
    the suffix reach the probe position. Measured relative to the logit scale
    because the absolute number is dtype-dependent: on this model the same
    check gives ~7e-3 in bfloat16 and ~5e-7 in float32, and only the float32
    figure reflects the mathematics.
    """
    prefix_ids = tokenizer(item.prefix)["input_ids"]
    position = len(prefix_ids) - 1
    full_ids = tokenizer(item.sentence)["input_ids"]
    if full_ids[: len(prefix_ids)] != list(prefix_ids):
        return float("nan")  # the probe token re-tokenizes across the boundary
    short, _, _ = lens.apply(model, item.prefix, positions=[position])
    long, _, _ = lens.apply(model, item.sentence, positions=[position])
    deviation = max(
        float((short[layer] - long[layer]).abs().max().item()) for layer in short
    )
    scale = max(float(short[layer].abs().max().item()) for layer in short)
    return deviation / scale if scale else float("nan")
