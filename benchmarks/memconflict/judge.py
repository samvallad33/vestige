"""Rule-based MemConflict judge.

This is a deliberate, faithful port of the *rule-based fallback judge* that
ships in the upstream MemConflict repository at the pinned revision:

    Evaluation/eval_scoring.py
      - normalize_text
      - extract_normalized_answer_variants
      - answers_match_strict
      - extract_meaningful_tokens  (+ FALLBACK_STOPWORDS)
      - compute_partial_credit_score
      - has_update_order_signal
      - has_conflict_recognition_signal
      - build_rule_based_result

WHY THE RULE-BASED JUDGE AND NOT THE LLM JUDGE
----------------------------------------------
Upstream's headline tables (Table 3 / Table 4 of arXiv:2605.20926) were
produced with an LLM judge (gpt-5.0-mini). We deliberately do NOT use an LLM
judge here, for two reasons:

  1. Reproducibility. An LLM judge makes the harness non-deterministic, costly,
     and dependent on a model snapshot that will drift. A stranger could not
     reproduce our numbers.
  2. Confound control. MemDelta (arXiv:2606.29914) shows that swapping the
     model in a memory evaluation moves the result by double-digit points and
     can reverse system rankings. Holding the judge fixed and deterministic is
     the entire point of the mandatory controls in this harness.

THE COST OF THAT CHOICE, STATED PLAINLY
---------------------------------------
Numbers produced by this judge are NOT comparable to the published Table 3
numbers. They are only comparable *across the arms in this harness*, which all
run through the identical judge. Never place our AA next to upstream's AA in
the same table or sentence.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

# --- verbatim port: normalize_text -----------------------------------------


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    normalized = str(text)
    normalized = normalized.replace("_", " ").replace("-", " ").lower()
    normalized = re.sub(r"[\"'`]", " ", normalized)
    normalized = re.sub("[^\\w\\s\u4e00-\u9fff]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_normalized_answer_variants(text: Any) -> List[str]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return []

    variants: List[str] = []
    separators = ["||", "\n", ";"]

    def add_variant(candidate: str) -> None:
        normalized = normalize_text(candidate)
        if normalized and normalized not in variants:
            variants.append(normalized)

    add_variant(raw_text)
    for separator in separators:
        if separator not in raw_text:
            continue
        for part in raw_text.split(separator):
            add_variant(part)
    return variants


def answers_match_strict(gold_answer: Any, model_answer: Any) -> bool:
    gold_variants = extract_normalized_answer_variants(gold_answer)
    model_variants = extract_normalized_answer_variants(model_answer)
    if not gold_variants or not model_variants:
        return False
    return any(g == m for g in gold_variants for m in model_variants)


FALLBACK_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "from", "of", "in", "on", "at", "for", "with", "by", "as",
    "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "it", "they", "them", "their", "he", "she", "his", "her", "you", "your",
    "user", "users", "now", "current", "currently", "recently", "about",
    "did", "does", "do", "has", "have", "had", "what", "where", "when", "which",
}


def extract_meaningful_tokens(text: Any) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens: List[str] = []
    for token in normalized.split():
        if len(token) <= 1:
            continue
        if token in FALLBACK_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def compute_partial_credit_score(gold_answer: Any, model_answer: Any) -> float:
    if answers_match_strict(gold_answer, model_answer):
        return 1.0

    gold_variants = extract_normalized_answer_variants(gold_answer)
    model_variants = extract_normalized_answer_variants(model_answer)
    if not gold_variants or not model_variants:
        return 0.0

    best_overlap_ratio = 0.0
    best_shared_count = 0
    for gold_variant in gold_variants:
        gold_tokens = set(extract_meaningful_tokens(gold_variant))
        if not gold_tokens:
            continue
        for model_variant in model_variants:
            model_tokens = set(extract_meaningful_tokens(model_variant))
            if not model_tokens:
                continue
            shared = gold_tokens & model_tokens
            shared_count = len(shared)
            overlap_ratio = shared_count / max(1, len(gold_tokens))
            if overlap_ratio > best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
            if shared_count > best_shared_count:
                best_shared_count = shared_count

            if gold_variant in model_variant and len(gold_variant) >= 8:
                return 0.5
            if model_variant in gold_variant and len(model_variant) >= 8:
                return 0.5

    if best_shared_count >= 2 or best_overlap_ratio >= 0.5:
        return 0.5
    return 0.0


def has_update_order_signal(model_answer: Any) -> bool:
    normalized = normalize_text(model_answer)
    if not normalized:
        return False
    change_markers = ["changed", "change", "updated", "update", "switched"]
    order_markers = [("from", "to"), ("previously", "now"), ("used to", "now"), ("before", "now")]
    has_change = any(m in normalized for m in change_markers)
    has_order = any(l in normalized and r in normalized for l, r in order_markers)
    return has_change and has_order


def has_conflict_recognition_signal(model_answer: Any) -> bool:
    normalized = normalize_text(model_answer)
    if not normalized:
        return False
    keywords = ["inconsisten", "conflict", "contradict", "cannot confirm", "uncertain", "mismatch"]
    return any(k in normalized for k in keywords)


# --- harness-side scoring ---------------------------------------------------


def score_question(question: Dict[str, Any], model_answer: str) -> Dict[str, Any]:
    """Score one (question, reader-output) pair with the rule-based judge.

    Returns a dict of metric_name -> value, plus which metrics are applicable
    for this conflict type. Metrics that do not apply are omitted entirely
    (never scored as 0), so averages are always taken over the correct
    denominator.
    """
    ctype = question.get("conflict_type", "unknown")
    gold = str(question.get("answer", "")).strip()
    aa = compute_partial_credit_score(gold, model_answer)

    out: Dict[str, Any] = {"conflict_type": ctype, "answer_accuracy": aa}

    if ctype == "dynamic_conflict":
        out["uocs"] = 1.0 if (aa >= 0.5 and has_update_order_signal(model_answer)) else 0.0
    elif ctype == "static_conflict":
        out["crs_lex"] = 1.0 if has_conflict_recognition_signal(model_answer) else 0.0
    elif ctype == "conditional_conflict":
        # Upstream binarises conditional AA at the 0.5 threshold.
        out["answer_accuracy"] = 1.0 if aa >= 0.5 else 0.0
    return out
