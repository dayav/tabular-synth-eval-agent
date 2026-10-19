"""Checker rules: rounding, truncation, percentages, sign, thousands separators,
markdown exemptions, coarse-number exactness, recomputed figures.

Provenance: checker rules, plus the recomputed-mean transcript recorded on
2026-10-15 (the model averaged its own rounded values and wrote a mean one unit
in the last place away from the tool's).
"""

from __future__ import annotations

from evals.cases import Case
from evals.checks.numeric_faithfulness import check_numeric_faithfulness, clean_report
from regression.helpers import load_fixture, prompts_for

RESULTS = [{"per_column": {"gender": 0.17429, "city": 0.0441}, "mean": 0.1092, "n_columns": 2},
           {"tstr_accuracy": 0.787, "trtr_accuracy": 0.992, "utility_gap": -0.205, "n_test_samples": 400, "income": 42000}]
PROMPT = "Flag above 0.1. Poor above 0.05. Split 80/20."


def ungrounded(text: str) -> list[str]:
    return check_numeric_faithfulness(text, RESULTS, PROMPT).details["ungrounded"]


def test_rounding_at_two_or_more_decimals() -> None:
    assert ungrounded("gender 0.174 and 0.17 and 0.1743") == []


def test_truncation_is_accepted() -> None:
    assert ungrounded("TSTR 0.78") == []


def test_percentages_are_two_decimals() -> None:
    assert ungrounded("accuracy 79% then 78.7%") == []
    assert ungrounded("accuracy 81%") == ["81%"]


def test_sign_is_ignored() -> None:
    assert ungrounded("gap −0.205, gap -0.205, gap 0.205, gap 0.21") == []


def test_thousands_separators() -> None:
    assert ungrounded("income 42,000 and 42000") == []


def test_markdown_furniture_is_exempt() -> None:
    text = "## 1. Categorical\n\n1. first\n2. second\n\n| a | b |\n|---|---|\n| gender | 0.174 |\n\n---\n"
    assert ungrounded(text) == []
    assert "1. first" not in clean_report(text)


def test_coarse_numbers_must_match_exactly() -> None:
    assert ungrounded("2 categorical columns") == []          # n_columns is 2
    assert ungrounded("0.1 is the threshold, 0.3 is not") == ["0.3"]
    assert ungrounded("7 columns in total") == ["7"]
    assert ungrounded("a factor of 10") == ["10"]
    assert ungrounded("scaled 10x") == ["10x"]


def test_identifiers_are_not_numbers() -> None:
    assert ungrounded("file age_x10 under eval_abcq, see v2 and sha1") == []


def test_fraction_is_reported() -> None:
    details = check_numeric_faithfulness("0.174 and 0.3 and 79%", RESULTS, PROMPT).details
    assert details["numbers_total"] == 3 and details["numbers_grounded"] == 2


def test_replayed_recomputed_mean_is_ungrounded(cases: list[Case]) -> None:
    fx = load_fixture("clean_synthetic_recomputed_mean.json")
    case = next(c for c in cases if c.name == fx["case"])
    results = [tc["result"] for tc in fx["tool_calls"]]
    check = check_numeric_faithfulness(fx["report"], results, *prompts_for(case, fx["prompt"]))
    assert not check.passed
    assert len(check.details["ungrounded"]) == 1, check.details
    assert check.details["ungrounded"][0].startswith("0.038"), check.details
