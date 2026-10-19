"""Checker rules for the verdict block: fenced and unfenced, last block wins,
prose mention required, flags must match exactly.

Provenance: checker rules.
"""

from __future__ import annotations

import json

from evals.checks.verdict import check_verdict, parse_verdict_block, strip_verdict_block

EXPECTED = {
    "verdict": {"categorical": "good", "numerical": "poor", "utility": "good", "overall": "acceptable"},
    "flagged_columns": ["age"],
}
BLOCK = json.dumps(EXPECTED)


def test_fenced_block() -> None:
    report = "The age column is off.\n\n```json\n" + BLOCK + "\n```"
    assert parse_verdict_block(report) == EXPECTED
    assert check_verdict(report, EXPECTED).passed
    assert "verdict" not in strip_verdict_block(report)


def test_unfenced_block() -> None:
    report = "The age column is off.\n\n" + BLOCK
    assert parse_verdict_block(report) == EXPECTED
    assert check_verdict(report, EXPECTED).passed


def test_last_block_wins() -> None:
    draft = json.dumps({"verdict": {**EXPECTED["verdict"], "overall": "poor"}, "flagged_columns": ["age"]})
    report = "age is off.\n\n```json\n" + draft + "\n```\n\nCorrection:\n\n```json\n" + BLOCK + "\n```"
    assert parse_verdict_block(report) == EXPECTED
    assert check_verdict(report, EXPECTED).passed


def test_prose_mention_required() -> None:
    report = "Something is off.\n\n```json\n" + BLOCK + "\n```"
    check = check_verdict(report, EXPECTED)
    assert not check.passed
    assert any("never named in the prose" in p for p in check.details["problems"])


def test_flags_must_match_exactly() -> None:
    extra = json.dumps({"verdict": EXPECTED["verdict"], "flagged_columns": ["age", "income"]})
    report = "age and income are off.\n\n```json\n" + extra + "\n```"
    check = check_verdict(report, EXPECTED)
    assert not check.passed
    assert any("flagged without evidence" in p for p in check.details["problems"])
    none = json.dumps({"verdict": EXPECTED["verdict"], "flagged_columns": []})
    check = check_verdict("age is off.\n\n" + none, EXPECTED)
    assert any("missed flags" in p for p in check.details["problems"])


def test_wrong_axis_fails() -> None:
    wrong = json.dumps({"verdict": {**EXPECTED["verdict"], "utility": "poor"}, "flagged_columns": ["age"]})
    check = check_verdict("age is off.\n\n" + wrong, EXPECTED)
    assert not check.passed
    assert any(p.startswith("utility:") for p in check.details["problems"])


def test_missing_block_fails() -> None:
    check = check_verdict("Just prose about age.", EXPECTED)
    assert not check.passed and check.details["got"] is None
