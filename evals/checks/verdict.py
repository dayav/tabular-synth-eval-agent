"""Check 3: the verdict block at the end of the report matches the known answer.

Also home to :func:`derive_expected`, which applies the system prompt's threshold
rules to raw tool output. The golden ``expected.json`` files are written by hand
from the construction of each case; an offline regression test asserts that they
agree with ``derive_expected``, so a stale answer key is caught before a model is
blamed for it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from evals.checks.common import CheckResult

VERDICT_START = re.compile(r"\{\s*\"verdict\"")
FENCED_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

JS_THRESHOLD = 0.1
COHEN_D_THRESHOLD = 0.2
UTILITY_GAP_THRESHOLD = 0.05

AXES = ("categorical", "numerical", "utility", "overall")


def _candidates(text: str) -> list[tuple[int, int, dict[str, Any]]]:
    """Every parsable ``{"verdict": ...}`` object in the text as ``(start, end, obj)``.

    Fenced blocks are tried first with their fence span; unfenced objects are
    found by raw-decoding from each ``{"verdict"`` occurrence.
    """
    found: list[tuple[int, int, dict[str, Any]]] = []
    decoder = json.JSONDecoder()
    for m in FENCED_BLOCK.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            found.append((m.start(), m.end(), obj))
    for m in VERDICT_START.finditer(text):
        try:
            obj, end = decoder.raw_decode(text, m.start())  # end is an absolute index
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            span = (m.start(), end)
            if not any(s <= span[0] and span[1] <= e for s, e, _ in found):
                found.append((span[0], span[1], obj))
    return sorted(found, key=lambda t: t[0])


def parse_verdict_block(text: str) -> dict[str, Any] | None:
    """Return the last verdict object in the report, or ``None``."""
    cands = _candidates(text)
    return cands[-1][2] if cands else None


def strip_verdict_block(text: str) -> str:
    """The report without its last verdict block (the prose meant for humans)."""
    cands = _candidates(text)
    if not cands:
        return text
    start, end, _ = cands[-1]
    return text[:start] + text[end:]


def _flags(per_column: dict[str, float | None], threshold: float, exclude: str) -> list[str]:
    return [
        col
        for col, v in per_column.items()
        if col != exclude and v is not None and v > threshold
    ]


def derive_expected(
    js_result: dict[str, Any],
    d_result: dict[str, Any],
    tstr_result: dict[str, Any],
    target_column: str,
    *,
    js_threshold: float = JS_THRESHOLD,
    d_threshold: float = COHEN_D_THRESHOLD,
    gap_threshold: float = UTILITY_GAP_THRESHOLD,
) -> dict[str, Any]:
    """Apply the prompt's verdict rules to raw tool output."""
    cat_flags = _flags(js_result["per_column"], js_threshold, target_column)
    num_flags = _flags(d_result["per_column"], d_threshold, target_column)
    gap = float(tstr_result["utility_gap"])
    categorical = "poor" if cat_flags else "good"
    numerical = "poor" if num_flags else "good"
    utility = "poor" if gap > gap_threshold else "good"
    if utility == "poor":
        overall = "poor"
    elif categorical == "good" and numerical == "good":
        overall = "excellent"
    else:
        overall = "acceptable"
    return {
        "verdict": {
            "categorical": categorical,
            "numerical": numerical,
            "utility": utility,
            "overall": overall,
        },
        "flagged_columns": sorted(cat_flags + num_flags),
    }


def _mentioned_in_prose(column: str, prose: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(column)}(?![\w])", prose) is not None


def check_verdict(report: str, expected: dict[str, Any]) -> CheckResult:
    """Compare the report's verdict block and prose with the expected answer."""
    block = parse_verdict_block(report)
    problems: list[str] = []
    if block is None:
        return CheckResult(
            name="verdict", passed=False, details={"problems": ["no verdict block found"], "got": None}
        )
    got_verdict = block.get("verdict") or {}
    for axis in AXES:
        if got_verdict.get(axis) != expected["verdict"][axis]:
            problems.append(f"{axis}: got {got_verdict.get(axis)!r}, expected {expected['verdict'][axis]!r}")
    got_flags = set(block.get("flagged_columns") or [])
    exp_flags = set(expected["flagged_columns"])
    if got_flags - exp_flags:
        problems.append(f"flagged without evidence: {sorted(got_flags - exp_flags)}")
    if exp_flags - got_flags:
        problems.append(f"missed flags: {sorted(exp_flags - got_flags)}")
    prose = strip_verdict_block(report)
    for col in sorted(exp_flags):
        if not _mentioned_in_prose(col, prose):
            problems.append(f"{col} is flagged in the block but never named in the prose")
    return CheckResult(
        name="verdict",
        passed=not problems,
        details={"problems": problems, "got": block},
    )
