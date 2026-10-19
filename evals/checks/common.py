"""Shared pieces for the checkers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

# Numbers written in prose, tables or JSON: optional sign (ASCII or Unicode minus),
# digits with optional thousands separators, optional decimals, optional percent.
# The look-behind keeps digits inside identifiers (age_x10, v2, sha1) out.
NUMBER_RE = re.compile(
    r"(?<![\w.])[-\u2212]?\d[\d,]*(?:\.\d+)?"
    r"(?:%|x|\u00d7|\s?(?:percentage\s+points?|percent|pp)\b)?(?!\w)"
)
_PERCENT_WORDS = re.compile(r"\s?(?:percentage\s+points?|percent|pp)$")


@dataclass
class CheckResult:
    """Outcome of one mechanical check on one run."""

    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, **self.details}


def iter_numbers(obj: Any) -> Iterator[float]:
    """Yield every number found recursively in a tool result (dict/list/scalars)."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_numbers(v)
    elif isinstance(obj, str):
        yield from numbers_in_text(obj)


def numbers_in_text(text: str) -> Iterator[float]:
    """Yield the numeric value of every number written in ``text``."""
    for token in NUMBER_RE.findall(text):
        value, _ = parse_number_token(token)
        yield value


def parse_number_token(token: str) -> tuple[float, int]:
    """Turn a matched token into ``(value, decimals)``.

    ``"79%"`` and ``"79 percentage points"`` become ``(0.79, 2)``; ``"1,234.5"`` becomes ``(1234.5, 1)``;
    ``"−0.205"`` becomes ``(-0.205, 3)``.
    """
    t = token.replace("−", "-").replace(",", "")
    percent = t.endswith("%") or bool(_PERCENT_WORDS.search(t))
    t = _PERCENT_WORDS.sub("", t)
    if t.endswith(("%", "x", "\u00d7")):
        t = t[:-1]
    decimals = len(t.split(".")[1]) if "." in t else 0
    value = float(t)
    if percent:
        value /= 100.0
        decimals += 2
    return value, decimals


def unique_sorted(values: Iterable[float]) -> list[float]:
    return sorted(set(values))
