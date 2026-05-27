#!/usr/bin/env python3
# sanhedrin-local.py — OpenAI-compatible Sanhedrin Executioner bridge.
# Drop-in replacement for the Haiku 4.5 subagent that sanhedrin.sh used to spawn.
#
# Reads draft from stdin, prints single-line verdict to stdout:
#   yes
#   no - [Sanhedrin Veto] [CLASS]: <reason under 120 chars>
#
# Architecture:
#   stdin (draft) -> Vestige /api/deep_reference (single semantic query)
#                 -> OpenAI-compatible chat endpoint (one-shot judgment)
#                 -> stdout (single-line verdict)
#
# Fail-open: if the endpoint is unreachable, print "yes" and exit 0 (don't break
# the Cognitive Sandwich on infra errors). The wrapping sanhedrin.sh maps
# "yes" to exit 0, so this preserves existing fail-open semantics.

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import sanhedrin_core
except Exception:
    sanhedrin_core = None


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


DASHBOARD_PORT = os.environ.get("VESTIGE_DASHBOARD_PORT") or "3927"
VESTIGE_BASE_URL = (
    os.environ.get("VESTIGE_BASE_URL") or f"http://127.0.0.1:{DASHBOARD_PORT}"
).rstrip("/")

SANHEDRIN_ENDPOINT = (
    os.environ.get("VESTIGE_SANHEDRIN_ENDPOINT")
    or os.environ.get("MLX_ENDPOINT")
    or ""
)
VESTIGE_ENDPOINT = (
    os.environ.get("VESTIGE_DEEP_REFERENCE_ENDPOINT")
    or f"{VESTIGE_BASE_URL}/api/deep_reference"
)
VESTIGE_HEALTH = (
    os.environ.get("VESTIGE_HEALTH_ENDPOINT") or f"{VESTIGE_BASE_URL}/api/health"
)
MODEL = (
    os.environ.get("VESTIGE_SANHEDRIN_MODEL")
    or os.environ.get("VESTIGE_SANDWICH_MODEL")
    or ""
)
SANHEDRIN_TIMEOUT = env_int("VESTIGE_SANHEDRIN_TIMEOUT", env_int("MLX_TIMEOUT", 45))
VESTIGE_TIMEOUT = env_int("VESTIGE_TIMEOUT", 5)
SANHEDRIN_BACKEND = (os.environ.get("VESTIGE_SANHEDRIN_BACKEND") or "").strip().lower()
THINK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
    re.DOTALL | re.IGNORECASE,
)


def post_json(url: str, body: dict, timeout: int):
    if not url:
        return None
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("VESTIGE_SANHEDRIN_API_KEY")
    if api_key and same_endpoint_origin(url, SANHEDRIN_ENDPOINT):
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=data, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def same_endpoint_origin(url: str, endpoint: str) -> bool:
    try:
        target = urllib.parse.urlsplit(url)
        expected = urllib.parse.urlsplit(endpoint)
    except ValueError:
        return False
    return (
        target.scheme == expected.scheme
        and target.netloc == expected.netloc
        and target.path == expected.path
    )


def use_backend_extensions() -> bool:
    if SANHEDRIN_BACKEND in {"mlx", "vllm"}:
        return True
    if SANHEDRIN_BACKEND in {"openai", "ollama", "llama.cpp", "llamacpp", "litellm"}:
        return False
    return MODEL.startswith("mlx-community/")


def sanhedrin_model_configured() -> bool:
    return bool(SANHEDRIN_ENDPOINT and MODEL)


def sanhedrin_body(
    messages: list[dict[str, str]],
    max_tokens: int,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    if stop:
        body["stop"] = stop
    if use_backend_extensions():
        body["top_k"] = 1
        body["seed"] = 42
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


TRUST_FLOOR = 0.55  # filter out low-trust memories that drive false-positive vetoes

CLAIM_MODE_ENV = "VESTIGE_SANHEDRIN_CLAIM_MODE"
OUTPUT_ENV = "VESTIGE_SANHEDRIN_OUTPUT"
STAGE_FILE_ENV = "VESTIGE_SANHEDRIN_STAGE_FILE"
MAX_CLAIMS = env_int("VESTIGE_SANHEDRIN_MAX_CLAIMS", 8)
MAX_CLAIM_CHARS = env_int("VESTIGE_SANHEDRIN_MAX_CLAIM_CHARS", 500)
MAX_EVIDENCE_CHARS = env_int("VESTIGE_SANHEDRIN_MAX_EVIDENCE_CHARS", 420)

CLAIM_CLASSES = {
    "TECHNICAL",
    "BIOGRAPHICAL",
    "FINANCIAL",
    "ACHIEVEMENT",
    "TIMELINE",
    "QUANTITATIVE",
    "ATTRIBUTION",
    "CAUSAL",
    "COMPARATIVE",
    "EXISTENTIAL",
    "VAGUE-QUANTIFIER",
    "UNVERIFIED-POSITIVE",
}
CRITICAL_ABSENCE_CLASSES = {
    "BIOGRAPHICAL",
    "FINANCIAL",
    "ACHIEVEMENT",
    "TIMELINE",
    "QUANTITATIVE",
    "ATTRIBUTION",
    "VAGUE-QUANTIFIER",
}
STRUCTURED_VERDICTS = {"SUPPORTED", "REFUTED", "REFUTED_BY_ABSENCE", "NEI"}
SEVERITY_ORDER = {
    "BIOGRAPHICAL": 0,
    "FINANCIAL": 1,
    "ACHIEVEMENT": 2,
    "ATTRIBUTION": 3,
    "TIMELINE": 4,
    "QUANTITATIVE": 5,
    "VAGUE-QUANTIFIER": 6,
    "UNVERIFIED-POSITIVE": 7,
    "TECHNICAL": 8,
    "EXISTENTIAL": 9,
    "CAUSAL": 10,
    "COMPARATIVE": 11,
}
USER_TERMS_RE = re.compile(
    r"\b(sam|sam's|the user|user's|you|your|yours|yourself)\b", re.IGNORECASE
)
HYPOTHETICAL_PREFIX_RE = re.compile(
    r"^\s*(if|suppose|imagine|hypothetically|assume|what if)\b",
    re.IGNORECASE,
)
SUBJECT_MODAL_PREFIX_RE = re.compile(
    r"^\s*(sam|sam's|the user|user's|you|your)\b\s+(would|could)\b",
    re.IGNORECASE,
)
TRAILING_MODAL_COMMENT_RE = re.compile(
    r"\s*,?\s+(which|that)\s+(would|could)\b.*$",
    re.IGNORECASE,
)
CURRENT_TURN_PREFIXES = [
    re.compile(r"^\s*(per your request|as requested)\s*,?\s*", re.IGNORECASE),
    re.compile(
        r"^\s*(you|sam|the user)\s+(asked for|requested)\s+maximum subagents\b[^,.;]*(?:,?\s*(and|so)\s*)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(you|sam|the user)\s+(asked|told|requested|wanted)\s+"
        r"(?:(me|us|codex|claude)\s+)?(to|for)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(your|sam's|the user's)\s+request\s+(was|is)\s+(to|for)\s+",
        re.IGNORECASE,
    ),
]
FIRST_PERSON_DISCOURSE_RE = re.compile(
    r"^\s*(i|we)\s+(reviewed|audited|checked|inspected|looked at|verified|"
    r"confirmed|found|updated|changed|implemented|fixed|patched|added|removed|"
    r"wired|ran|left)\b",
    re.IGNORECASE,
)
DISCOURSE_ACTION_PREFIX_RE = re.compile(
    r"^\s*(audit|review|check|inspect|look at|verify|confirm|implement|fix|"
    r"patch|add|remove|wire|run|use|go all in)\b",
    re.IGNORECASE,
)
EMBEDDED_USER_CLAIM_RE = re.compile(r"\b(sam|sam's|the user|user's)\b", re.IGNORECASE)
TECHNICAL_RE = re.compile(
    r"(/\w|[\w.-]+\.(py|rs|ts|tsx|js|jsx|json|md|toml|yaml|yml|sh)\b|"
    r"\b(api|endpoint|env|flag|model|server|hook|script|function|class|repo|"
    r"crate|mcp|http|json|sqlite|rust|python|typescript|command|config)\b|"
    r"\b[A-Z][A-Z0-9_]{2,}\b)",
    re.IGNORECASE,
)
BIOGRAPHICAL_RE = re.compile(
    r"\b(born|lives?|located|based in|works? at|employed|employer|school|"
    r"university|college|graduated|degree|founder|ceo|cto|student|job|role)\b",
    re.IGNORECASE,
)
FINANCIAL_RE = re.compile(
    r"(\$[\d,.]+|\b(revenue|funding|raised|earned|paid|payout|prize money|"
    r"salary|net worth|valuation|stock|shares?|portfolio|profit|loss)\b)",
    re.IGNORECASE,
)
ACHIEVEMENT_RE = re.compile(
    r"\b(won|winner|ranked|placed|scored|score|completed|finished|launched|"
    r"released|shipped|milestone|award|prize|accepted|published|graduated)\b",
    re.IGNORECASE,
)
TIMELINE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?|today|yesterday|tomorrow|last week|next week|"
    r"\d+\s+(days?|weeks?|months?|years?)\b)",
    re.IGNORECASE,
)
QUANTITATIVE_RE = re.compile(
    r"(\b\d+(?:\.\d+)?\s*(%|percent|x|times|stars?|users?|customers?|"
    r"submissions?|points?|gb|mb|ms|s|seconds?|minutes?|hours?)?\b|"
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|dozens?|hundreds?|"
    r"thousands?|many|several|few|most)\b)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"\$?\b[a-z0-9][a-z0-9.-]*\b", re.IGNORECASE)
STOP_CLAIM_TOKENS = {
    "about",
    "after",
    "also",
    "because",
    "been",
    "before",
    "claim",
    "from",
    "have",
    "into",
    "more",
    "sam",
    "that",
    "their",
    "there",
    "this",
    "user",
    "with",
    "your",
}
ATTRIBUTION_RE = re.compile(
    r"\b(said|told|asked|agreed|decided|approved|rejected|committed|authored|"
    r"wrote|built|implemented|requested|wanted|prefers?)\b",
    re.IGNORECASE,
)
VAGUE_QUANTIFIER_RE = re.compile(
    r"\b(a few|some|several|many|most|multiple)\b.*\b(wins?|won|prizes?|"
    r"money|customers?|deals?|submissions?|placements?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Claim:
    text: str
    claim_class: str
    source_index: int
    sam_critical: bool


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    preview: str
    trust: float
    role: str = "evidence"
    date: str = ""
    durable: bool = True
    source: str = "vestige"


@dataclass
class ClaimVerdict:
    claim: Claim
    status: str
    reason: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    durable_evidence_count: int = 0
    high_trust_evidence_count: int = 0


def env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def truncate_chars(text: str, max_chars: int, suffix: str = "...") -> str:
    """Truncate by Python characters, never UTF-8 bytes, and avoid dangling marks."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(suffix):
        return text[:max_chars]
    cut = text[: max_chars - len(suffix)].rstrip()
    while cut and unicodedata.combining(cut[-1]):
        cut = cut[:-1]
    return f"{cut}{suffix}"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fetch_evidence(draft: str) -> tuple[str, int]:
    """Single deep_reference call — returns (formatted evidence, count of high-trust memories).
    Only memories with trust >= TRUST_FLOOR are surfaced. If none qualify, returns ("", 0)
    and the caller should auto-pass without invoking the model.
    """
    try:
        with urllib.request.urlopen(VESTIGE_HEALTH, timeout=VESTIGE_TIMEOUT) as r:
            r.read()
    except Exception:
        if sanhedrin_core is not None:
            sanhedrin_core.record_fail_open("vestige_health_unavailable", VESTIGE_HEALTH)
        return "", 0

    query = draft[:1500]
    resp = post_json(VESTIGE_ENDPOINT, {"query": query, "depth": 12}, VESTIGE_TIMEOUT)
    if not isinstance(resp, dict):
        if sanhedrin_core is not None:
            sanhedrin_core.record_fail_open("deep_reference_unavailable", VESTIGE_ENDPOINT)
        return "", 0

    parts = []
    high_trust_count = 0
    confidence = resp.get("confidence", 0)

    rec = resp.get("recommended") or {}
    rec_trust = float(rec.get("trust_score", 0) or 0)
    if rec and rec_trust >= TRUST_FLOOR:
        rid = (rec.get("memory_id") or rec.get("id") or "")[:8]
        date = (rec.get("date") or "")[:10]
        prev = (rec.get("answer_preview") or rec.get("preview") or "")[:500]
        parts.append(f"RECOMMENDED [{rid}] trust={rec_trust:.2f} date={date}:\n{prev}")
        high_trust_count += 1

    contradictions = resp.get("contradictions") or []
    if contradictions:
        parts.append(f"\nCONTRADICTIONS DETECTED: {len(contradictions)} pair(s)")
        for c in contradictions[:3]:
            parts.append(f"  - {json.dumps(c)[:200]}")

    superseded = resp.get("superseded") or []
    if superseded:
        ht_super = [s for s in superseded if float(s.get("trust", 0) or 0) >= TRUST_FLOOR]
        if ht_super:
            parts.append(f"\nSUPERSEDED MEMORIES (trust>={TRUST_FLOOR}): {len(ht_super)}")
            for s in ht_super[:3]:
                sid = (s.get("id") or "")[:8]
                parts.append(f"  - [{sid}] {(s.get('preview') or '')[:200]}")

    evidence = resp.get("evidence") or []
    high_trust_evidence = [ev for ev in evidence if float(ev.get("trust", 0) or 0) >= TRUST_FLOOR]
    if high_trust_evidence:
        parts.append(f"\nHIGH-TRUST EVIDENCE (trust>={TRUST_FLOOR}, {min(len(high_trust_evidence), 5)} of {len(evidence)} total):")
        for ev in high_trust_evidence[:5]:
            eid = (ev.get("id") or "")[:8]
            role = ev.get("role", "?")
            trust = float(ev.get("trust", 0) or 0)
            prev = (ev.get("preview") or "").strip()[:300]
            parts.append(f"  [{eid}] role={role} trust={trust:.2f}\n    {prev}")
            high_trust_count += 1

    if high_trust_count == 0:
        return "", 0

    header = f"VESTIGE CONFIDENCE: {int(confidence * 100)}% | HIGH-TRUST MEMORIES: {high_trust_count}\n\n"
    return header + "\n".join(parts), high_trust_count


def split_candidate_claims(draft: str) -> list[str]:
    """Return sentence-ish draft fragments that can be classified as claims."""
    without_fences = re.sub(r"```.*?```", " ", draft[:16_384], flags=re.DOTALL)
    without_fences = re.sub(r"`[^`\n]+`", " ", without_fences)
    without_fences = re.sub(r'(^|[\s([{])"[^"\n]+"(?=([\s.,;:!?)}\]]|$))', r"\1 ", without_fences)
    fragments: list[str] = []
    for line in without_fences.splitlines():
        if line.lstrip().startswith(">"):
            continue
        line = re.sub(r"^\s*[-*+]\s+", "", line).strip()
        line = re.sub(r"^\s*\d+[.)]\s+", "", line).strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'])", line)
        fragments.extend(part.strip(" \t-") for part in parts if part.strip(" \t-"))
        if len(fragments) >= 512:
            return fragments[:512]
    if not fragments:
        compact = " ".join(without_fences.split())
        fragments = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'])", compact)
            if part.strip()
        ]
    return fragments[:512]


def normalize_asserted_fragment(text: str) -> str | None:
    text = " ".join(text.split()).strip()
    if not text:
        return None
    if re.search(r"\b(need|still need|let me|will|should)\s+to\s+verify\b", text, re.I):
        return None
    if re.fullmatch(
        r"(the user|user|sam|you)\s+(said|told|asked|wrote|noted|mentioned)\s*"
        r"(earlier|before|previously)?\.?",
        text,
        re.I,
    ):
        return None
    text = TRAILING_MODAL_COMMENT_RE.sub("", text).strip(" ,;:-")
    if HYPOTHETICAL_PREFIX_RE.search(text) or SUBJECT_MODAL_PREFIX_RE.search(text):
        return None

    for prefix in CURRENT_TURN_PREFIXES:
        stripped = prefix.sub("", text, count=1).strip(" ,;:-")
        if stripped == text:
            continue
        embedded = EMBEDDED_USER_CLAIM_RE.search(stripped)
        if embedded and embedded.start() > 0 and DISCOURSE_ACTION_PREFIX_RE.search(stripped):
            stripped = stripped[embedded.start() :].strip(" ,;:-")
        elif DISCOURSE_ACTION_PREFIX_RE.search(stripped) or FIRST_PERSON_DISCOURSE_RE.search(stripped):
            return None
        text = stripped
        break

    if FIRST_PERSON_DISCOURSE_RE.search(text):
        embedded = EMBEDDED_USER_CLAIM_RE.search(text)
        if embedded and embedded.start() > 0:
            text = text[embedded.start() :].strip(" ,;:-")
        else:
            return None

    text = TRAILING_MODAL_COMMENT_RE.sub("", text).strip(" ,;:-")
    if not text or HYPOTHETICAL_PREFIX_RE.search(text) or SUBJECT_MODAL_PREFIX_RE.search(text):
        return None
    return text


def classify_claim(text: str) -> str | None:
    """Classify a factual-shaped claim with conservative, testable heuristics."""
    if VAGUE_QUANTIFIER_RE.search(text):
        return "VAGUE-QUANTIFIER"
    if BIOGRAPHICAL_RE.search(text):
        return "BIOGRAPHICAL"
    if FINANCIAL_RE.search(text):
        return "FINANCIAL"
    if ACHIEVEMENT_RE.search(text):
        return "ACHIEVEMENT"
    if ATTRIBUTION_RE.search(text):
        return "ATTRIBUTION"
    if TECHNICAL_RE.search(text):
        return "TECHNICAL"
    if TIMELINE_RE.search(text):
        return "TIMELINE"
    if QUANTITATIVE_RE.search(text):
        return "QUANTITATIVE"
    if re.search(r"\b(exists?|there is|there are|contains?|includes?)\b", text, re.I):
        return "EXISTENTIAL"
    if re.search(r"\b(because|caused|causes|therefore|so that|as a result)\b", text, re.I):
        return "CAUSAL"
    if re.search(r"\b(better|best|faster|fastest|more than|less than|fewer than)\b", text, re.I):
        return "COMPARATIVE"
    return None


def is_sam_critical_claim(text: str, claim_class: str) -> bool:
    if claim_class not in CRITICAL_ABSENCE_CLASSES:
        return False
    return bool(USER_TERMS_RE.search(text))


def extract_check_worthy_claims(
    draft: str,
    max_claims: int = MAX_CLAIMS,
    max_claim_chars: int = MAX_CLAIM_CHARS,
) -> list[Claim]:
    claims: list[Claim] = []
    seen: set[str] = set()
    for idx, fragment in enumerate(split_candidate_claims(draft)):
        text = normalize_asserted_fragment(fragment)
        if not text:
            continue
        claim_class = classify_claim(text)
        if not claim_class:
            continue
        text = truncate_chars(text, max_claim_chars)
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            Claim(
                text=text,
                claim_class=claim_class,
                source_index=idx,
                sam_critical=is_sam_critical_claim(text, claim_class),
            )
        )
    return sorted(
        claims,
        key=lambda claim: (SEVERITY_ORDER.get(claim.claim_class, 99), claim.source_index),
    )[:max_claims]


def normalize_evidence_item(raw: Any, source: str = "vestige") -> EvidenceItem | None:
    if isinstance(raw, str):
        preview = raw.strip()
        if not preview:
            return None
        return EvidenceItem(
            id="stage",
            preview=truncate_chars(preview, MAX_EVIDENCE_CHARS),
            trust=1.0,
            role="staged",
            durable=False,
            source="stage",
        )
    if not isinstance(raw, dict):
        return None

    preview = (
        raw.get("preview")
        or raw.get("answer_preview")
        or raw.get("content")
        or raw.get("text")
        or raw.get("claim")
        or ""
    )
    preview = str(preview).strip()
    if not preview:
        return None
    trust = safe_float(raw.get("trust", raw.get("trust_score", 1.0 if source == "stage" else 0.0)))
    item_id = str(raw.get("memory_id") or raw.get("id") or source or "evidence")
    role = str(raw.get("role") or ("staged" if source == "stage" else "evidence"))
    date = str(raw.get("date") or raw.get("created_at") or "")[:32]
    return EvidenceItem(
        id=item_id,
        preview=truncate_chars(preview, MAX_EVIDENCE_CHARS),
        trust=trust,
        role=role,
        date=date,
        durable=(source != "stage"),
        source=source,
    )


def evidence_from_deep_reference(resp: dict[str, Any]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    rec = resp.get("recommended") or {}
    rec_item = normalize_evidence_item(rec, "vestige")
    if rec_item:
        items.append(rec_item)
    for raw in resp.get("evidence") or []:
        item = normalize_evidence_item(raw, "vestige")
        if item:
            items.append(item)
    for raw in resp.get("superseded") or []:
        item = normalize_evidence_item(raw, "vestige")
        if item:
            items.append(item)
    return dedupe_evidence(items)


def dedupe_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    deduped: list[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item.source, item.id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def load_staged_evidence(path: str | None) -> list[EvidenceItem]:
    """Read optional JSON-array staged evidence. It is non-durable by design."""
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    items: list[EvidenceItem] = []
    for idx, raw_item in enumerate(raw):
        item = normalize_evidence_item(raw_item, "stage")
        if item is None:
            continue
        if item.id == "stage":
            item = replace(item, id=f"stage:{idx}")
        items.append(item)
    return items


def claim_query(claim: Claim) -> str:
    return (
        f"Class: {claim.claim_class}\n"
        f"Claim: {claim.text}"
    )


def fetch_claim_evidence(claim: Claim) -> tuple[list[EvidenceItem], bool]:
    resp = post_json(VESTIGE_ENDPOINT, {"query": claim_query(claim), "depth": 12}, VESTIGE_TIMEOUT)
    if not isinstance(resp, dict):
        return [], False
    if resp.get("error") or resp.get("errors"):
        return [], False
    if str(resp.get("status") or "").strip().lower() in {
        "error",
        "failed",
        "failure",
        "unavailable",
        "timeout",
    }:
        return [], False
    if not any(
        key in resp
        for key in ("confidence", "evidence", "recommended", "reasoning", "query", "status")
    ):
        return [], False
    return evidence_from_deep_reference(resp), True


def high_trust(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return [item for item in items if item.trust >= TRUST_FLOOR]


def durable_high_trust(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return [item for item in items if item.durable and item.trust >= TRUST_FLOOR]


def salient_claim_tokens(text: str) -> set[str]:
    tokens = {token.lower().strip(".") for token in TOKEN_RE.findall(text)}
    return {
        token
        for token in tokens
        if len(token) >= 4 and token not in STOP_CLAIM_TOKENS
    }


def evidence_relevant_to_claim(claim: Claim, evidence: EvidenceItem) -> bool:
    claim_numbers = set(re.findall(r"\$?\d+(?:[,.]\d+)*(?:\.\d+)?", claim.text))
    if claim_numbers and any(num in evidence.preview for num in claim_numbers):
        return True
    claim_tokens = salient_claim_tokens(claim.text)
    if not claim_tokens:
        return True
    preview_tokens = salient_claim_tokens(evidence.preview)
    overlap = claim_tokens & preview_tokens
    threshold = 1 if claim.claim_class == "TECHNICAL" else 2
    return len(overlap) >= threshold


def relevant_durable_high_trust(claim: Claim, items: list[EvidenceItem]) -> list[EvidenceItem]:
    return [
        item
        for item in durable_high_trust(items)
        if evidence_relevant_to_claim(claim, item)
    ]


def format_claim_evidence(items: list[EvidenceItem], claim: Claim | None = None) -> str:
    if not items:
        return "(no relevant evidence retrieved)"
    lines = []
    durable_count = (
        len(relevant_durable_high_trust(claim, items))
        if claim is not None
        else len(durable_high_trust(items))
    )
    high_count = len(high_trust(items))
    lines.append(
        f"HIGH-TRUST EVIDENCE: {high_count} | DURABLE HIGH-TRUST EVIDENCE: {durable_count}"
    )
    stage_count = len([item for item in items if not item.durable])
    if stage_count:
        lines.append(
            "STAGED EVIDENCE PRESENT: non-durable overlay; do not count it as durable memory."
        )
    for item in high_trust(items)[:8]:
        durable = "durable" if item.durable else "staged"
        short_id = item.id[:12]
        lines.append(
            f"[{short_id}] {durable} role={item.role} trust={item.trust:.2f} date={item.date}\n"
            f"{item.preview}"
        )
    return "\n\n".join(lines)


SYSTEM_PROMPT = """You are the Sanhedrin Executioner. You judge whether a DRAFT contradicts Vestige memory evidence about the user. ONE LINE OF OUTPUT.

VALID CLASS TAGS (closed set — pick exactly one):
TECHNICAL | ACHIEVEMENT | FINANCIAL | BIOGRAPHICAL | TIMELINE | ATTRIBUTION | VAGUE-QUANTIFIER | UNVERIFIED-POSITIVE

DEFAULT POSTURE
- DEFAULT to `yes` (PASS) for TECHNICAL / TIMELINE / EXISTENTIAL claims unless you can cite a same-subject direct contradiction.
- DEFAULT to `no` (VETO, fail-closed) for these specific user-about claims when high-trust evidence is silent on the named entity:
    * Specific institution / employer / school / company the user is claimed to be at
    * Specific dollar amount won / earned / raised
    * Specific competition placement / score / prize received
    * Specific date the user did something specific (graduated, was hired, was born)
    * Vague-quantifier positive about the user ("a few wins", "some prize money", "most submissions placed top 10", "many customers", "several deals")

THREE FALSE-POSITIVE PROTECTIONS (these output `yes`)
1. SUBJECT-EQUALITY GATE: only same-subject claims are veto candidates. Memory about Vestige's internal codebase ≠ contradiction with external tools (Qwen, MCP-protocol-spec, MLX, Cursor). Memory about project X ≠ contradiction with project Y.
2. VERSION-DISCRIMINATOR RULE: version/generation tokens (M1/M2/M3/M4/M5, v0.5/v1.0, GPT-4/GPT-5, Qwen3.5/Qwen3.6) are subject discriminators. Different versions = different subjects = no contradiction by default.
3. AGREEMENT-IS-NOT-CONTRADICTION: if the memory preview AGREES with the draft claim, that's PASS not VETO.

INFERENCE BAN
- DO NOT use "implies", "implying", "suggests", "must mean", "would mean", "indicates", "therefore" in veto reasons.
- If you have to chain inferences from a memory to reach a contradiction, PASS.
- TIMELINE vetoes specifically: require an EXPLICIT date or duration in the cited memory that arithmetically excludes the draft's date. Vague phrases like "until I graduate" cannot ground a TIMELINE veto.

ARCHITECTURE-VS-COMPONENT RULE
- A memory describing OVERALL architecture (Thalamus+Sanhedrin triad, 4-layer biology) does NOT contradict a draft about an INTERNAL COMPONENT (subagent model, sidecar transport, bridge script). Different layers of the same stack are not contradictions.

OUTPUT FORMAT (exactly one line, no preamble, no explanation, no markdown)
- PASS: yes
- VETO: no - [Sanhedrin Veto] [CLASS]: <reason under 140 chars, cite memory id verbatim from evidence>

EIGHT WORKED EXAMPLES — STUDY THESE PATTERNS

[VETO — same-subject TECHNICAL contradiction]
Evidence: "Vestige is a 2-crate Rust workspace (vestige-core + vestige-mcp)" trust=0.62 [de43be5a]
Draft: "Edit the FastAPI router in vestige/main.py for Python extensions to Vestige"
Output: no - [Sanhedrin Veto] TECHNICAL: Draft says FastAPI/Python for Vestige, memory de43be5a says 2-crate Rust workspace.

[VETO — same-subject ACHIEVEMENT contradiction]
Evidence: "Final benchmark submission scored 36/50 on April 15, no payout" trust=0.71 [9cf2a764]
Draft: "The user won the benchmark with a perfect 50/50 and took the $25K grand prize"
Output: no - [Sanhedrin Veto] ACHIEVEMENT: Draft claims 50/50 win + $25K, memory 9cf2a764 shows 36/50 final, no payout.

[VETO — VAGUE-QUANTIFIER fail-closed]
Evidence: high-trust memories about the user's competition history, none enumerate any wins
Draft: "The user won a few competitions and earned some prize money"
Output: no - [Sanhedrin Veto] VAGUE-QUANTIFIER: Draft says "a few wins / some prize money", evidence enumerates zero wins, fail-closed.

[VETO — UNVERIFIED-POSITIVE fail-closed]
Evidence: high-trust memories about the user's identity/work, no example school or employer mention
Draft: "The user graduated from Example University in 2019 with a 3.94 GPA and worked at Example Labs"
Output: no - [Sanhedrin Veto] UNVERIFIED-POSITIVE: Specific Stanford/2019/Google Brain claims, evidence silent on all, fail-closed.

[PASS — SUBJECT-EQUALITY gate (external tool, not Vestige)]
Evidence: "Vestige is a 2-crate Rust workspace" trust=0.62
Draft: "Switched the Sanhedrin executioner to local Qwen3.6-35B-A3B via mlx_lm.server"
Output: yes

[PASS — VERSION-DISCRIMINATOR rule]
Evidence: "M5 Max ~900 GB/s bandwidth (planned hardware)" trust=0.62
Draft: "Memory bandwidth on the M3 Max is around 400 GB/s for the unified architecture"
Output: yes

[PASS — AGREEMENT-IS-NOT-CONTRADICTION]
Evidence: "The user's M3 Max MacBook Pro arrived 2026-04-20" trust=0.55
Draft: "The user's MacBook is an M3 Max"
Output: yes

[PASS — ARCHITECTURE-VS-COMPONENT]
Evidence: "Cognitive Sandwich = Thalamus preflight triad + Sanhedrin Stop council shipped 2026-04-20" trust=0.7
Draft: "Cognitive Sandwich's Sanhedrin originally used a Haiku 4.5 subagent for the Executioner role"
Output: yes

[PASS — AUXILIARY-SCRIPT consumer-vs-consumed]
Evidence: "Vestige is a 2-crate Rust workspace" trust=0.62
Draft: "I added a Python script (sanhedrin-local.py) at ~/.claude/hooks/ that calls Vestige's HTTP API for evidence fetch."
Reason: external script that CALLS Vestige is not the same subject as Vestige's internal implementation. The consumer is not the consumed.
Output: yes

[PASS — HYPOTHETICAL-MOOD]
Evidence: "Final benchmark score was 36/50 with no payout" trust=0.71
Draft: "If the user wins the benchmark 50/50 next time around, they could claim the $25K grand prize."
Reason: prefix `if`/`suppose`/`imagine`/`hypothetically`/`would`/`could`/`assume` marks the embedded claim as conditional, NOT asserted. Conditional claims about future or counterfactual states do not contradict factual memory.
Output: yes

HYPOTHETICAL-MOOD RULE: if a draft sentence is governed by `if`, `suppose`, `imagine`, `hypothetically`, `would`, `could`, `assume`, `what if`, the embedded claim is NOT being asserted as fact — PASS that claim regardless of memory state.

ARCHIVED-COMPETITION RULE: do NOT fail-closed on the EXISTENCE of a past competition or project just because evidence is silent on it. Fail-closed applies only to specific PLACEMENT, SCORE, PRIZE, INSTITUTION, or DOLLAR AMOUNT -- not to "the user participated in X."

MULTI-CLAIM SEVERITY ORDERING: if multiple claims are vetoable, choose ACHIEVEMENT/FINANCIAL/BIOGRAPHICAL/UNVERIFIED-POSITIVE over TECHNICAL. Specific fabrications about the user's life are more dangerous than tech-stack mismatches.

When in doubt on TECHNICAL/TIMELINE: PASS. When in doubt on a user-about ACHIEVEMENT/FINANCIAL/BIOGRAPHICAL claim with specific named entities not in evidence: VETO with UNVERIFIED-POSITIVE."""


CLAIM_SYSTEM_PROMPT = """You are the Sanhedrin Executioner in claim mode. Judge ONE extracted claim against the provided Vestige evidence.

Return exactly one JSON object, no markdown:
{
  "status": "SUPPORTED|REFUTED|REFUTED_BY_ABSENCE|NEI",
  "class": "<claim class>",
  "reason": "<short reason, under 140 chars>",
  "evidence_ids": ["<ids used>"]
}

Rules:
- SUPPORTED: high-trust evidence directly supports the claim.
- REFUTED: high-trust evidence directly contradicts the same-subject claim.
- REFUTED_BY_ABSENCE: use only when instructions say absence-fail-closed applies.
- NEI: not enough information, stale/noisy evidence, wrong subject, or inference required.
- Do not infer contradiction across different subjects, versions, projects, or architecture layers.
- Staged evidence is context only and is not durable Vestige memory.
- Reasons must not use implies, suggests, must mean, would mean, indicates, therefore, or this means.
"""


VALID_CLASSES = CLAIM_CLASSES
INFERENCE_VERBS = (
    "implies", "implying", "suggests", "must mean", "would mean",
    "indicates that", "therefore the", "this means",
)
VERDICT_RE = re.compile(
    r"^no - \[Sanhedrin Veto\] \[?([A-Z][A-Z\-]*)\]?: (.{1,180})$"
)


def validate_verdict(verdict: str) -> str:
    """Post-validate the model's verdict. Fail-open ('yes') on any malformation:
    - Length over 220 chars
    - Veto with class tag not in the closed set
    - Veto reason containing inference verbs
    - Veto not matching the canonical regex
    """
    v = verdict.strip()
    if not v:
        return "yes"
    low = v.lower()
    if low == "yes" or low.startswith("yes "):
        return "yes"
    if not low.startswith("no"):
        return "yes"
    if len(v) > 220:
        return "yes"  # runaway reasoning blob
    m = VERDICT_RE.match(v)
    if not m:
        return "yes"  # format break
    cls = m.group(1)
    reason = m.group(2)
    if cls not in VALID_CLASSES:
        return "yes"  # invented class tag
    reason_low = reason.lower()
    for verb in INFERENCE_VERBS:
        if verb in reason_low:
            return "yes"  # inference-chain veto, downgrade per ban
    return v


def judge(draft: str, evidence: str) -> str:
    if not sanhedrin_model_configured():
        if sanhedrin_core is not None:
            sanhedrin_core.record_fail_open(
                "model_not_configured",
                "Set VESTIGE_SANHEDRIN_ENDPOINT and VESTIGE_SANHEDRIN_MODEL, or choose a preset.",
            )
        return ""
    user_msg = (
        f"VESTIGE EVIDENCE (recommended + top trust-scored memories):\n"
        f"{evidence if evidence else '(no relevant evidence retrieved)'}\n\n"
        f"---\nDRAFT TO JUDGE:\n{draft}"
    )
    body = sanhedrin_body(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        2500,
        [
            "\n\nWait,", "\n\nActually,", "\n\nLet me", "\n\nHmm,",
            "\n\nOn second thought", "\n\nOh wait",
        ],
    )
    resp = post_json(SANHEDRIN_ENDPOINT, body, SANHEDRIN_TIMEOUT)
    if not isinstance(resp, dict):
        return ""
    try:
        msg = resp["choices"][0]["message"]
        raw = msg.get("content") or ""
        if not raw.strip():
            raw = msg.get("reasoning") or ""
    except (KeyError, IndexError, TypeError):
        return ""
    cleaned = THINK_RE.sub("", raw).strip()
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return ""
    last = lines[-1]
    low = last.lower()
    if low.startswith("yes") or low.startswith("no"):
        return validate_verdict(last)
    for ln in reversed(lines):
        l = ln.lower()
        if l.startswith("yes") or l.startswith("no"):
            return validate_verdict(ln)
    return ""


def absence_verdict(claim: Claim) -> ClaimVerdict:
    reason = (
        f"{claim.claim_class} claim about Sam has zero high-trust durable Vestige evidence."
    )
    return ClaimVerdict(
        claim=claim,
        status="REFUTED_BY_ABSENCE",
        reason=truncate_chars(reason, 140),
    )


def nei_verdict(
    claim: Claim,
    reason: str,
    evidence: list[EvidenceItem] | None = None,
) -> ClaimVerdict:
    evidence = evidence or []
    return ClaimVerdict(
        claim=claim,
        status="NEI",
        reason=truncate_chars(reason, 140),
        evidence_ids=[item.id for item in high_trust(evidence)[:3]],
        durable_evidence_count=len(relevant_durable_high_trust(claim, evidence)),
        high_trust_evidence_count=len(high_trust(evidence)),
    )


def supported_verdict(claim: Claim, evidence: list[EvidenceItem]) -> ClaimVerdict:
    return ClaimVerdict(
        claim=claim,
        status="SUPPORTED",
        reason="High-trust evidence supports or does not contradict the claim.",
        evidence_ids=[item.id for item in high_trust(evidence)[:3]],
        durable_evidence_count=len(relevant_durable_high_trust(claim, evidence)),
        high_trust_evidence_count=len(high_trust(evidence)),
    )


def parse_json_object(raw: str) -> dict[str, Any] | None:
    cleaned = THINK_RE.sub("", raw).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(cleaned[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def verdict_from_legacy_line(claim: Claim, raw: str, evidence: list[EvidenceItem]) -> ClaimVerdict | None:
    line = validate_verdict(raw)
    if line == "yes":
        return supported_verdict(claim, evidence)
    m = VERDICT_RE.match(line)
    if not m:
        return None
    reason = m.group(2)
    if not relevant_durable_high_trust(claim, evidence):
        return nei_verdict(claim, "Durable evidence required for refuted verdict.", evidence)
    return ClaimVerdict(
        claim=claim,
        status="REFUTED",
        reason=truncate_chars(reason, 140),
        evidence_ids=[item.id for item in high_trust(evidence)[:3]],
        durable_evidence_count=len(relevant_durable_high_trust(claim, evidence)),
        high_trust_evidence_count=len(high_trust(evidence)),
    )


def validate_structured_verdict(
    claim: Claim,
    data: dict[str, Any],
    evidence: list[EvidenceItem],
) -> ClaimVerdict:
    status = str(data.get("status") or "").strip().upper()
    if status not in STRUCTURED_VERDICTS:
        status = "NEI"
    claim_class = str(data.get("class") or claim.claim_class).strip().upper()
    if claim_class not in CLAIM_CLASSES:
        claim_class = claim.claim_class
    reason = truncate_chars(str(data.get("reason") or "").strip(), 140)
    if any(verb in reason.lower() for verb in INFERENCE_VERBS):
        return nei_verdict(claim, "Inference-chain verdict downgraded to NEI.", evidence)
    if status == "REFUTED_BY_ABSENCE":
        if not (claim.sam_critical and claim.claim_class in CRITICAL_ABSENCE_CLASSES):
            return nei_verdict(claim, "Absence veto does not apply to this claim.", evidence)
        if relevant_durable_high_trust(claim, evidence):
            return nei_verdict(claim, "Durable evidence exists; absence veto does not apply.", evidence)
    if status == "REFUTED" and not relevant_durable_high_trust(claim, evidence):
        return nei_verdict(claim, "Durable evidence required for refuted verdict.", evidence)
    if status == "SUPPORTED" and high_trust(evidence) and not durable_high_trust(evidence):
        return nei_verdict(claim, "Durable evidence required for supported verdict.", evidence)
    evidence_ids_raw = data.get("evidence_ids") or []
    evidence_ids = [
        str(eid) for eid in evidence_ids_raw[:5]
    ] if isinstance(evidence_ids_raw, list) else []
    if not reason:
        if status == "SUPPORTED":
            reason = "High-trust evidence supports or does not contradict the claim."
        elif status == "NEI":
            reason = "Not enough high-trust evidence to decide."
        elif status == "REFUTED_BY_ABSENCE":
            reason = absence_verdict(claim).reason
        else:
            reason = "High-trust evidence refutes the claim."
    return ClaimVerdict(
        claim=Claim(
            text=claim.text,
            claim_class=claim_class,
            source_index=claim.source_index,
            sam_critical=claim.sam_critical,
        ),
        status=status,
        reason=reason,
        evidence_ids=evidence_ids,
        durable_evidence_count=len(relevant_durable_high_trust(claim, evidence)),
        high_trust_evidence_count=len(high_trust(evidence)),
    )


def judge_claim_with_model(claim: Claim, evidence: list[EvidenceItem]) -> ClaimVerdict:
    if not sanhedrin_model_configured():
        if sanhedrin_core is not None:
            sanhedrin_core.record_fail_open(
                "model_not_configured",
                "Set VESTIGE_SANHEDRIN_ENDPOINT and VESTIGE_SANHEDRIN_MODEL, or choose a preset.",
            )
        return nei_verdict(claim, "Sanhedrin model not configured; fail-open for this claim.", evidence)
    user_msg = (
        f"CLAIM CLASS: {claim.claim_class}\n"
        f"SAM-CRITICAL: {'yes' if claim.sam_critical else 'no'}\n"
        f"ABSENCE-FAIL-CLOSED APPLIES: "
        f"{'yes' if claim.sam_critical and claim.claim_class in CRITICAL_ABSENCE_CLASSES else 'no'}\n"
        f"DURABLE HIGH-TRUST EVIDENCE COUNT: {len(relevant_durable_high_trust(claim, evidence))}\n\n"
        f"CLAIM:\n{claim.text}\n\n"
        f"EVIDENCE:\n{format_claim_evidence(evidence, claim)}"
    )
    body = sanhedrin_body(
        [
            {"role": "system", "content": CLAIM_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        700,
    )
    resp = post_json(SANHEDRIN_ENDPOINT, body, SANHEDRIN_TIMEOUT)
    if not isinstance(resp, dict):
        if sanhedrin_core is not None:
            sanhedrin_core.record_fail_open("model_unavailable", f"endpoint={SANHEDRIN_ENDPOINT}")
        return nei_verdict(claim, "Sanhedrin model unavailable; fail-open for this claim.", evidence)
    try:
        msg = resp["choices"][0]["message"]
        raw = msg.get("content") or msg.get("reasoning") or ""
    except (KeyError, IndexError, TypeError):
        if sanhedrin_core is not None:
            sanhedrin_core.record_fail_open("malformed_model_response", f"endpoint={SANHEDRIN_ENDPOINT}")
        return nei_verdict(claim, "Malformed Sanhedrin model response.", evidence)
    data = parse_json_object(raw)
    if data is not None:
        return validate_structured_verdict(claim, data, evidence)
    legacy = verdict_from_legacy_line(claim, raw, evidence)
    if legacy is not None:
        return legacy
    if sanhedrin_core is not None:
        sanhedrin_core.record_fail_open("unstructured_model_response", raw[:500])
    return nei_verdict(claim, "Sanhedrin model did not return structured JSON.", evidence)


def judge_claim(claim: Claim, evidence: list[EvidenceItem]) -> ClaimVerdict:
    if not sanhedrin_model_configured():
        if sanhedrin_core is not None:
            sanhedrin_core.record_fail_open(
                "model_not_configured",
                "Set VESTIGE_SANHEDRIN_ENDPOINT and VESTIGE_SANHEDRIN_MODEL, or choose a preset.",
            )
        return nei_verdict(claim, "Sanhedrin model not configured; fail-open for this claim.", evidence)
    durable_count = len(relevant_durable_high_trust(claim, evidence))
    high_count = len(high_trust(evidence))
    if claim.sam_critical and claim.claim_class in CRITICAL_ABSENCE_CLASSES and durable_count == 0:
        verdict = absence_verdict(claim)
        verdict.high_trust_evidence_count = high_count
        return verdict
    if high_count == 0:
        return nei_verdict(claim, "No high-trust evidence retrieved for this claim.", evidence)
    return judge_claim_with_model(claim, evidence)


def render_legacy_from_verdicts(verdicts: list[ClaimVerdict]) -> str:
    vetoes = [v for v in verdicts if v.status in {"REFUTED", "REFUTED_BY_ABSENCE"}]
    if not vetoes:
        return "yes"
    vetoes.sort(
        key=lambda v: (
            SEVERITY_ORDER.get(v.claim.claim_class, 99),
            v.claim.source_index,
        )
    )
    chosen = vetoes[0]
    reason = truncate_chars(chosen.reason or chosen.claim.text, 140)
    return f"no - [Sanhedrin Veto] [{chosen.claim.claim_class}]: {reason}"


def recompute_legacy_from_result(result: dict[str, Any]) -> str:
    vetoes = []
    for raw in result.get("verdicts", []):
        claim = raw.get("claim", {}) if isinstance(raw, dict) else {}
        status = str(raw.get("status", ""))
        if status not in {"REFUTED", "REFUTED_BY_ABSENCE"}:
            continue
        vetoes.append(
            (
                SEVERITY_ORDER.get(str(claim.get("claim_class", "")), 99),
                int(claim.get("source_index", 0) or 0),
                str(claim.get("claim_class", "TECHNICAL")),
                truncate_chars(str(raw.get("reason") or claim.get("text") or ""), 140),
            )
        )
    if not vetoes:
        return "yes"
    _, _, claim_class, reason = sorted(vetoes)[0]
    return f"no - [Sanhedrin Veto] [{claim_class}]: {reason}"


def apply_appeals_to_claim_mode_result(result: dict[str, Any]) -> dict[str, Any]:
    if sanhedrin_core is None:
        return result
    appeals = sanhedrin_core.load_appeals()
    changed = False
    for raw in result.get("verdicts", []):
        if not isinstance(raw, dict) or raw.get("status") not in {"REFUTED", "REFUTED_BY_ABSENCE"}:
            continue
        claim = raw.get("claim", {}) if isinstance(raw.get("claim"), dict) else {}
        text = str(claim.get("text") or "")
        if sanhedrin_core.is_appealed({"fingerprint": sanhedrin_core.claim_fingerprint(text)}, appeals):
            raw["status"] = "APPEALED"
            raw["reason"] = "Prior appeal suppresses this Sanhedrin veto."
            changed = True

    if changed:
        legacy = recompute_legacy_from_result(result)
        result["legacy_verdict"] = legacy
        result["decision"] = "yes" if legacy == "yes" else "no"
        result["verdict"] = result["decision"]
        result["passed"] = legacy == "yes"
        result["reason"] = "" if result["passed"] else legacy.split(" - ", 1)[-1]
    return result


def save_claim_mode_receipt(
    draft: str,
    result: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> None:
    if sanhedrin_core is None:
        return
    manifest = manifest or sanhedrin_core.new_manifest(draft)
    claims = []
    for idx, raw in enumerate(result.get("verdicts", []), start=1):
        if not isinstance(raw, dict):
            continue
        claim = raw.get("claim", {}) if isinstance(raw.get("claim"), dict) else {}
        text = str(claim.get("text") or "")
        claim_class = str(claim.get("claim_class") or "TECHNICAL")
        status = str(raw.get("status") or "NEI")
        evidence_ids = raw.get("evidence_ids") if isinstance(raw.get("evidence_ids"), list) else []
        if status == "SUPPORTED":
            decision = "pass"
            evidence_state = "supported"
            fix = "No change required."
        elif status == "APPEALED":
            decision = "appealed"
            evidence_state = "appealed"
            fix = "Prior appeal suppresses this veto fingerprint."
        elif status == "REFUTED_BY_ABSENCE":
            decision = "veto"
            evidence_state = "missing_precedent"
            fix = "Remove the unsupported user-specific claim or cite durable Vestige evidence first."
        elif status == "REFUTED":
            decision = "veto"
            evidence_state = "contradicted"
            fix = "Remove or qualify the contradicted claim using the cited Vestige precedent."
        else:
            decision = "pass_unverified"
            evidence_state = "not_enough_information"
            fix = "No blocking change required."
        claims.append(
            {
                "id": f"c{idx:03d}",
                "text": text,
                "fingerprint": sanhedrin_core.claim_fingerprint(text),
                "class": claim_class,
                "subject": "Sam" if bool(claim.get("sam_critical")) else "draft",
                "risk": "hard" if bool(claim.get("sam_critical")) else "normal",
                "evidence_state": evidence_state,
                "decision": decision,
                "precedent": [
                    {
                        "type": "vestige",
                        "summary": str(raw.get("reason") or status),
                        "evidence": ", ".join(str(eid) for eid in evidence_ids[:5]),
                        "durableCount": raw.get("durable_evidence_count"),
                        "highTrustCount": raw.get("high_trust_evidence_count"),
                    }
                ],
                "fix": fix,
                "appeal": {
                    "status": "appealed" if decision == "appealed" else "open",
                    "actions": ["stale", "wrong", "too_strict"],
                },
            }
        )

    manifest["claims"] = claims
    manifest["overall"] = "pass" if result.get("passed") else "veto"
    if any(claim["decision"] == "appealed" for claim in claims):
        manifest["overall"] = "pass_with_warnings" if result.get("passed") else manifest["overall"]
        manifest["verdictBar"] = "APPEALED"
        manifest["summary"] = "Prior appeal suppressed a Sanhedrin veto."
    elif result.get("passed"):
        manifest["verdictBar"] = "PASS" if not claims else "NOTE"
        manifest["summary"] = "Sanhedrin found no blocking claim issues."
    else:
        manifest["verdictBar"] = "VETO"
        manifest["summary"] = str(result.get("reason") or "Sanhedrin blocked a claim.")
    sanhedrin_core.save_manifest(manifest)


def save_legacy_receipt(manifest: dict[str, Any] | None, verdict: str, evidence: str = "") -> str:
    if sanhedrin_core is None or manifest is None:
        return verdict
    updated = sanhedrin_core.apply_model_verdict(manifest, verdict, evidence)
    sanhedrin_core.save_manifest(manifest)
    return updated


def claim_mode_result(draft: str) -> dict[str, Any]:
    claims = extract_check_worthy_claims(draft)
    staged = load_staged_evidence(os.environ.get(STAGE_FILE_ENV))
    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        evidence, ok = fetch_claim_evidence(claim)
        if not ok:
            if sanhedrin_core is not None:
                sanhedrin_core.record_fail_open("retrieval_unavailable", claim.text)
            verdicts.append(
                nei_verdict(
                    claim,
                    "Vestige retrieval unavailable; fail-open for this claim.",
                    staged,
                )
            )
            continue
        combined = dedupe_evidence(evidence + staged)
        verdicts.append(judge_claim(claim, combined))
    legacy_verdict = render_legacy_from_verdicts(verdicts)
    decision = "yes" if legacy_verdict == "yes" else "no"
    json_reason = "" if decision == "yes" else legacy_verdict.split(" - ", 1)[-1]
    return {
        "mode": "claim",
        "decision": decision,
        "verdict": decision,
        "reason": json_reason,
        "passed": legacy_verdict == "yes",
        "legacy_verdict": legacy_verdict,
        "claims_extracted": len(claims),
        "staged_evidence_count": len(staged),
        "verdicts": [asdict(v) for v in verdicts],
    }


def print_claim_mode_result(result: dict[str, Any]) -> None:
    if (os.environ.get(OUTPUT_ENV) or "").strip().lower() == "json":
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    else:
        print(result.get("legacy_verdict") or "yes")


def main() -> None:
    draft = sys.stdin.read().strip()
    if not draft:
        print("yes")
        return

    manifest = sanhedrin_core.new_manifest(draft) if sanhedrin_core is not None else None
    if sanhedrin_core is not None and manifest is not None:
        receipt_veto = sanhedrin_core.apply_receipt_lock(manifest)
        if receipt_veto:
            sanhedrin_core.save_manifest(manifest)
            print(f"no - [Sanhedrin Veto] [TECHNICAL]: {receipt_veto}")
            return

    if env_flag(CLAIM_MODE_ENV):
        result = apply_appeals_to_claim_mode_result(claim_mode_result(draft))
        save_claim_mode_receipt(draft, result, manifest)
        print_claim_mode_result(result)
        return

    evidence, high_trust_count = fetch_evidence(draft)

    # Auto-pass if no high-trust evidence — model can't legitimately veto
    # without something concrete to cite. Eliminates the common false-positive
    # mode where the model invents a contradiction from low-trust noise.
    if high_trust_count == 0:
        save_legacy_receipt(manifest, "yes", evidence)
        print("yes")
        return

    verdict = judge(draft, evidence)

    if not verdict:
        # Fail-open: server unreachable, malformed response, etc.
        if sanhedrin_core is not None:
            sanhedrin_core.record_fail_open("legacy_model_unavailable", f"endpoint={SANHEDRIN_ENDPOINT}")
        save_legacy_receipt(manifest, "yes", evidence)
        print("yes")
        return

    verdict = save_legacy_receipt(manifest, verdict, evidence)
    print(verdict)


if __name__ == "__main__":
    main()
