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

# USD per million tokens, from the official per-model documentation pages at
# docs.mistral.ai/models/<id>: (input, cached_input, output).
# Cached input is billed at its own lower rate and the API reports the cached
# count per call, so it is priced separately rather than folded into input.
PRICING_USD_PER_MTOK = {
    # docs.mistral.ai/models/mistral-large
    "mistral-large-latest": (0.5, 0.5, 1.5),
    "mistral-large-2512": (0.5, 0.5, 1.5),
    # docs.mistral.ai/models/zai-glm-5-2
    "glm-5-2": (1.4, 0.14, 4.4),
    "zai-glm-5-2": (1.4, 0.14, 4.4),
}
BUDGET_EUR = 100.0


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
            "budget_eur": BUDGET_EUR,
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


def vocabulary_words(tokenizer: Any) -> list[str]:
    """Every distinct surface string in the vocabulary, lowercased."""
    words = {
        tokenizer.convert_tokens_to_string([surface]).strip().lower()
        for surface in tokenizer.get_vocab()
    }
    return sorted(w for w in words if w)


def fragment_completions(
    token: str, words: list[str], limit: int = 12
) -> list[str]:
    """Vocabulary words that extend ``token``, as evidence about a fragment.

    A readout token may be a split piece of a longer word. Whether it names
    anything is a fact about the vocabulary, not a matter of recall: "matrim"
    is extended only by the matrimony family, while "Vol" is extended by
    volatile, volcano, Voldemort and many more. Computing this and handing it
    to the judge replaces a guess with evidence.
    """
    stem = token.strip().lower()
    if not stem:
        return []
    return [w for w in words if w.startswith(stem) and w != stem][:limit]


def annotate_fragments(
    tokens: list[str], words: list[str], limit: int = 10
) -> dict[str, dict[str, Any]]:
    """Vocabulary evidence for the readout tokens that need it.

    Tokens with no completions are already whole words, so they are omitted
    entirely; the judge is told to read absence from this map as "whole word".
    That keeps the payload small enough to batch many items into one call.
    """
    # Deliberately no "is this a whole word" flag: every readout token is by
    # construction a vocabulary token, so such a flag is true for all of them
    # and would contradict the completion evidence for a genuine fragment.
    annotated: dict[str, dict[str, Any]] = {}
    for token in tokens:
        completions = fragment_completions(token, words, limit)
        if completions:
            annotated[token] = {
                "n_completions": len(completions),
                "completions": completions,
            }
    return annotated


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
    api_key: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 1200,
    model: str = GENERATOR_MODEL,
    temperature: float = 0.7,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": temperature,
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


JUDGE_MODEL = "glm-5-2"

JUDGE_SYSTEM = """\
You compare concept lists produced by two readouts of the same language model, \
and decide whether a target concept is PRESENT in each list.

PRESENT means an entry in the list names the target concept. Allowed as the \
same concept: different casing; inflections and plurals; the same word in \
another language or script; adjectival and demonymic forms; and exact synonyms \
that denote the very same thing.

PRESENT examples for target "japan": "Japan", "Japon", "Japanese", "\u65e5\u672c".
PRESENT examples for target "wedding": "Wedding", "weddings", "nuptials", \
"matrimony", "marriage".

ABSENT means no entry names it. Three specific traps, all ABSENT:

1. Association. Entries merely caused by, part of, or found near the concept. \
   For "japan": "sushi", "rice", "chef", "temple", "Tokyo". For "chess": \
   "pawn", "board", "king", "rook", "move". For "wedding": "altar", "bride", \
   "vows", "oath".
2. Broader or narrower categories. A superordinate is not the concept. For \
   "wedding": "ceremony", "ceremonies", "ceremonial", "rites", "proceedings" \
   are ABSENT, because funerals and graduations are ceremonies too. For \
   "volcano": "mountain", "disaster" are ABSENT.
3. Uninformative fragments. Tokenizers split words, so list A may contain a \
   truncated piece of one. Do not judge this from memory. The field \
   "vocabulary_evidence" lists, for those entries of list A that are extended \
   by longer vocabulary words, what those words are. An entry that does NOT \
   appear in that map has no completions and is already a whole word; judge it \
   normally. For an entry that does appear:

   - If "completions" all belong to one word family, the entry names that \
     word. Then apply the normal test to that word. Example: "matrim" is \
     extended only by matrimon/matrimoni/matrimoniale/matrimonio, so it names \
     matrimony, and matrimony denotes a wedding, so it is PRESENT for \
     "wedding".
   - If "completions" span unrelated words, the entry names nothing and is \
     ABSENT for every target. Example: "Vol" is extended by volatile, volcano, \
     Voldemort and others.
   - Identifying a word is not enough on its own. "noct" is extended only by \
     "noctur", so it names nocturnal, but nocturnal is not insomnia, so it is \
     ABSENT for "insomnia" on meaning.

   Cite the evidence you used in "reason" when an entry is a fragment.

Two hard rules:

- The value you put in "matched" MUST be copied verbatim from the list you \
  claim it appears in. Never invent it, never adjust it, never take it from \
  the other list.
- "reason" must describe what you actually found. If the concept is absent \
  from a list, say so; do not claim it is named.

Be strict. A liberal reading makes every list match every concept and destroys \
the measurement. When genuinely uncertain, answer false.

Answer with JSON only."""


def judge_agreement(
    api_key: str,
    concept: str,
    lens_tokens: list[str],
    self_report: list[str] | None,
    ledger: Path | None = None,
    fragment_evidence: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask a judge model whether the concept is named in each list.

    Exact string matching is too brittle for this: the lens may surface
    " Japon" or " Japanese" while the self-report says "japan", and both name
    the same concept. The judge decides semantic identity while refusing mere
    association, which is what keeps the criterion from loosening until
    everything matches.
    """
    user = json.dumps(
        {
            "target_concept": concept,
            "list_a_lens_readout": lens_tokens,
            "list_a_vocabulary_evidence": fragment_evidence or {},
            "list_b_model_self_report": self_report if self_report else [],
            "answer_format": {
                "lens": {"present": "bool", "matched": "string or null"},
                "self_report": {"present": "bool", "matched": "string or null"},
                "agree": "bool, true only if present in BOTH",
                "reason": "one short sentence",
            },
        },
        ensure_ascii=False,
    )
    exchange = call_mistral(
        api_key,
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=JUDGE_MODEL,
        max_tokens=400,
        temperature=0.0,  # the judge is a measurement, so make it repeatable
    )
    if ledger is not None:
        record_spend(ledger, exchange, note=f"judge: {concept}")
    verdict = json.loads(exchange["content"])
    return verify_quoted_match(verdict, lens_tokens, self_report), exchange


def verify_quoted_match(
    verdict: dict[str, Any],
    lens_tokens: list[str],
    self_report: list[str] | None,
) -> dict[str, Any]:
    """Reject any match the judge did not copy from the list it cites.

    A "matched" value absent from its own list is a fabrication, so the claim
    is discarded rather than trusted. This caught a judge reporting the
    fragment "noct" as a match while justifying it as "directly named".
    """
    sources = {
        "lens": [str(x) for x in lens_tokens],
        "self_report": [str(x) for x in (self_report or [])],
    }
    verdict["fabricated_matches"] = []
    for key, entries in sources.items():
        side = verdict.get(key)
        if not isinstance(side, dict) or not side.get("present"):
            continue
        claimed = str(side.get("matched") or "").strip().lower()
        if claimed not in {e.strip().lower() for e in entries}:
            verdict["fabricated_matches"].append(
                {"list": key, "claimed": side.get("matched")}
            )
            side["rejected_claim"] = side.get("matched")
            side["present"] = False
            side["matched"] = None
    verdict["agree"] = bool(
        verdict.get("lens", {}).get("present")
        and verdict.get("self_report", {}).get("present")
    )
    return verdict


def judge_batch(
    api_key: str,
    cases: list[dict[str, Any]],
    ledger: Path | None = None,
    retries: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adjudicate several items in one call.

    Each case supplies ``concept``, ``lens_tokens``, ``self_report`` and
    ``fragment_evidence``. Verdicts are matched back by index and every index
    must come back, so a short or renumbered reply is an error rather than a
    silent misalignment of verdicts to items.
    """
    payload = {
        "items": [
            {
                "id": index,
                "target_concept": case["concept"],
                "list_a_lens_readout": case["lens_tokens"],
                "vocabulary_evidence": case.get("fragment_evidence") or {},
                "list_b_model_self_report": case.get("self_report") or [],
            }
            for index, case in enumerate(cases)
        ],
        "answer_format": {
            "verdicts": [
                {
                    "id": "int, echoing the item id",
                    "lens": {"present": "bool", "matched": "string or null"},
                    "self_report": {"present": "bool", "matched": "string or null"},
                    "reason": "one short sentence",
                }
            ]
        },
    }
    exchange = call_mistral(
        api_key,
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        model=JUDGE_MODEL,
        max_tokens=350 * len(cases) + 300,
        temperature=0.0,
    )
    if ledger is not None:
        record_spend(ledger, exchange, note=f"judge batch of {len(cases)}")

    try:
        raw = json.loads(exchange["content"]).get("verdicts", [])
    except json.JSONDecodeError:
        raw = []
    by_id = {int(v["id"]): v for v in raw if "id" in v}
    missing = sorted(set(range(len(cases))) - set(by_id))
    if missing:
        # Malformed or short replies happen at every batch size, including one
        # item, so this is not a batching artefact. Retry, then split: a batch
        # that keeps failing is bisected down to singletons rather than
        # silently returning fewer verdicts than items.
        if retries > 0:
            return judge_batch(api_key, cases, ledger, retries - 1)
        if len(cases) > 1:
            middle = len(cases) // 2
            left, _ = judge_batch(api_key, cases[:middle], ledger)
            right, exchange = judge_batch(api_key, cases[middle:], ledger)
            return left + right, exchange
        raise RuntimeError(f"judge omitted verdicts for items {missing}")

    verdicts = []
    for index, case in enumerate(cases):
        verdicts.append(
            verify_quoted_match(
                by_id[index], case["lens_tokens"], case.get("self_report")
            )
        )
    return verdicts, exchange


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
Read this text fragment. It breaks off mid-sentence, immediately after the word \
"{term}".

{prefix}

Think about the whole fragment, not the word "{term}" on its own. What is the \
situation about? Name the activity, event, place, domain, object or entity it \
involves, including things that are strongly implied but never stated \
anywhere in the text.

Give the {limit} concepts that come to mind. Do not simply list synonyms, parts \
or properties of "{term}".

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


PRIMARY_LAYER = 12

# Measured on 24 recorded items: two singleton passes agree perfectly (0/24
# flips at temperature 0), and batches of 4, 8 and 16 each differ from
# singleton on the same single borderline item (4.2%), with no growth in the
# rate as the batch grows. Malformed replies occur at every size including
# one, so they are a reliability problem rather than a batching one and are
# handled by retry-then-bisect. 8 keeps prompts short and makes a bisect cheap.
JUDGE_BATCH_SIZE = 8


def lens_readout(
    item: JSpaceItem,
    model: Any,
    tokenizer: Any,
    lenses: dict[str, Any],
    *,
    top_k: int,
    primary_layer: int = PRIMARY_LAYER,
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

    # Adjudication happens at one fixed layer, so the dataset label matches the
    # layer the correction is evaluated at. Every other layer is still recorded,
    # because the readouts are free once the forward pass is done and changing
    # the designated layer later should not require recomputing anything.
    methods: dict[str, Any] = {}
    primary: dict[str, Any] = {}
    key = str(primary_layer)
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
        # Pick the variant that is best AT the primary layer, which need not be
        # the one that is best somewhere else in the stack.
        local = min(scored, key=lambda s: scored[s]["layers"][key]["target_rank"])
        at_layer = scored[local]["layers"][key]
        primary[name] = {
            "variant": local,
            "target_rank": at_layer["target_rank"],
            "in_top_k": at_layer["target_rank"] <= top_k,
            "top_tokens": [tok["decoded"] for tok in at_layer["top_tokens"]],
        }
    return {
        "probe_position": position,
        "probe_token": describe_tokens(tokenizer, [input_ids[position]])[0],
        "concept_variants": [s for s, _ in variants],
        "n_prefix_tokens": len(input_ids),
        "primary_layer": primary_layer,
        "primary": primary,
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
