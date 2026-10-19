"""Check 2: every number in the report must trace to a tool result or to the prompt.

The agent is a reporter, not a calculator. Its only legitimate sources of numbers
are the tool results and the text it was shown (system prompt, user prompt, tool
descriptions). A number that appears nowhere in those is counted as invented,
even when the arithmetic behind it is right.

Grounding rule:

* written with two or more decimals (a percentage counts as two): some allowed
  value must round *or* truncate to it at that precision;
* written with fewer than two decimals: it must equal an allowed value exactly.
  At one decimal almost anything collides with a legitimate rounding (0.286 would
  ground an invented "0.3 threshold").

Sign is ignored: "a gap of 0.21 in favour of synthetic" quotes -0.205 faithfully.
"""

from __future__ import annotations

import math
import re
from typing import Any

from evals.checks.common import NUMBER_RE, CheckResult, iter_numbers, parse_number_token
from evals.checks.verdict import strip_verdict_block

_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_NUMBERED_HEADING = re.compile(r"^\s*#{1,6}\s*\d+(?:\.\d+)*[.)]?\s*", re.MULTILINE)
_TABLE_RULE = re.compile(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-*:?\s*\|?\s*$", re.MULTILINE)
_HORIZONTAL_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
_FENCE = re.compile(r"```.*?```", re.DOTALL)


def allowed_numbers(tool_results: list[Any], *prompt_texts: str) -> set[float]:
    """The allowed set: every number in every tool result plus every number in the prompts."""
    allowed: set[float] = set()
    for result in tool_results:
        allowed.update(iter_numbers(result))
    for text in prompt_texts:
        for token in NUMBER_RE.findall(text):
            allowed.add(parse_number_token(token)[0])
    return allowed


def clean_report(text: str) -> str:
    """Remove markdown furniture whose digits are not quoted figures."""
    text = strip_verdict_block(text)
    text = _FENCE.sub(" ", text)
    text = _TABLE_RULE.sub(" ", text)
    text = _HORIZONTAL_RULE.sub(" ", text)
    text = _NUMBERED_HEADING.sub(" ", text)
    text = _LIST_MARKER.sub(" ", text)
    return text


def _round_half_up(x: float, decimals: int) -> float:
    q = 10**decimals
    return math.floor(abs(x) * q + 0.5) / q


def _truncate(x: float, decimals: int) -> float:
    q = 10**decimals
    return math.floor(abs(x) * q + 1e-9) / q


def is_grounded(value: float, decimals: int, allowed: set[float]) -> bool:
    v = abs(value)
    for a in allowed:
        a = abs(a)
        if math.isclose(a, v, rel_tol=0, abs_tol=1e-9):
            return True
        if decimals >= 2 and (
            math.isclose(_round_half_up(a, decimals), v, abs_tol=1e-9)
            or math.isclose(round(a, decimals), v, abs_tol=1e-9)
            or math.isclose(_truncate(a, decimals), v, abs_tol=1e-9)
        ):
            return True
    return False


def check_numeric_faithfulness(
    report: str,
    tool_results: list[Any],
    *prompt_texts: str,
) -> CheckResult:
    allowed = allowed_numbers(tool_results, *prompt_texts)
    tokens = NUMBER_RE.findall(clean_report(report))
    ungrounded: list[str] = []
    for token in tokens:
        value, decimals = parse_number_token(token)
        if not is_grounded(value, decimals, allowed):
            ungrounded.append(token)
    total = len(tokens)
    return CheckResult(
        name="numeric_faithfulness",
        passed=not ungrounded,
        details={
            "numbers_total": total,
            "numbers_grounded": total - len(ungrounded),
            "ungrounded": ungrounded,
        },
    )
