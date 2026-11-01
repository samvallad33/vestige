"""Unmodified MemForm MCQ parser from opanhw/memform-code, MIT license.

Pinned source commit0565117042a31a5f78576f89ff6b8a77c393ee09.
See UPSTREAM-LICENSE.txt; parser AST is checked against pinned source at use.
"""

from __future__ import annotations

import re


def parse_final_label_details(
    text: str, valid_labels: list[str]
) -> tuple[str | None, str | None]:
    """Robustly parse an MCQ label without using an LLM judge.

    Accepted examples include:
      FINAL_ANSWER: A
      Answer: A
      Final answer is A
      A
      A. Extend the timeout
      (A) Extend the timeout

    Explicit final-answer markers are preferred. Otherwise, only line-level answer-like
    forms are accepted, scanning from the end of the response to avoid picking labels
    mentioned in the model's reasoning.
    """
    mapping = {str(x).upper(): str(x) for x in valid_labels}
    label_alt = "|".join(re.escape(x) for x in sorted(mapping, key=len, reverse=True))

    # 1) Strong explicit markers anywhere in the response; use the last one.
    explicit_patterns = [
        rf"FINAL[_\s-]*ANSWER\s*(?:[:：=]|IS\b)\s*[\[(]?({label_alt})[\])]?(?=\s|[.。,:：;；!?]|$)",
        rf"ANSWER\s*(?:[:：=]|IS\b)\s*[\[(]?({label_alt})[\])]?(?=\s|[.。,:：;；!?]|$)",
        rf"(?:THE\s+)?(?:FINAL\s+)?CHOICE\s*(?:[:：=]|IS\b)\s*[\[(]?({label_alt})[\])]?(?=\s|[.。,:：;；!?]|$)",
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return mapping.get(str(matches[-1]).upper()), "explicit_marker"

    stripped = text.strip()
    if stripped.upper() in mapping:
        return mapping[stripped.upper()], "exact_label"

    # 2) Answer-like lines, from bottom to top. Handles `A. option text`, `(A) ...`, `A: ...`.
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    line_pattern = re.compile(
        rf"^(?:FINAL\s+ANSWER\s*[:：]?\s*)?[\[(]?({label_alt})[\])]?\s*(?:[.。,:：;；\-]|\s)\s*.+$",
        flags=re.IGNORECASE,
    )
    for line in reversed(lines):
        m = line_pattern.match(line)
        if m:
            return mapping.get(m.group(1).upper()), "label_prefixed_line"
        if line.upper() in mapping:
            return mapping[line.upper()], "standalone_label_line"

    # 3) Conservative natural-language fallback.
    natural = re.findall(
        rf"(?:THE\s+)?ANSWER\s+(?:IS|WOULD\s+BE)\s+[\[(]?({label_alt})[\])]?(?=\s|[.。,:：;；!?]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if natural:
        return mapping.get(natural[-1].upper()), "natural_language_answer"

    return None, None
