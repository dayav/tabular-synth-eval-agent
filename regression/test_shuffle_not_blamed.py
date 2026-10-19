"""On the shuffled case, blaming income is a guess dressed up as a finding.

Provenance: constructed from the oracle report. A permutation keeps every
marginal, so no tool result points at income; the utility drop is real but its
cause is not visible. A report that flags income fails the verdict check; the
honest report, which says the cause is not visible, passes.
"""

from __future__ import annotations

import json
from typing import Any

from evals.cases import Case
from evals.checks.verdict import check_verdict
from regression.helpers import oracle_report, verdict_block


def _case(cases: list[Case], name: str) -> Case:
    return next(c for c in cases if c.name == name)


def test_flagging_income_fails(cases: list[Case], tool_outputs: dict[str, dict[str, Any]]) -> None:
    case = _case(cases, "income_shuffled")
    honest = oracle_report(case, tool_outputs[case.name])
    expected_block = verdict_block(case.expected["verdict"], [])
    assert expected_block in honest
    blamed = honest.replace(expected_block, verdict_block(case.expected["verdict"], ["income"]))
    blamed = blamed.replace(
        "no single column can be blamed", "the income column is the likely cause"
    )
    check = check_verdict(blamed, case.expected)
    assert not check.passed
    assert any("flagged without evidence: ['income']" in p for p in check.details["problems"]), check.details


def test_honest_report_passes(cases: list[Case], tool_outputs: dict[str, dict[str, Any]]) -> None:
    case = _case(cases, "income_shuffled")
    check = check_verdict(oracle_report(case, tool_outputs[case.name]), case.expected)
    assert check.passed, check.details


def test_leak_case_is_the_same_story(cases: list[Case], tool_outputs: dict[str, dict[str, Any]]) -> None:
    case = _case(cases, "score_leaks_target")
    honest = oracle_report(case, tool_outputs[case.name])
    assert json.loads('{"a": 1}')  # keep json imported for symmetry with other fixtures
    blamed = honest.replace(
        verdict_block(case.expected["verdict"], []), verdict_block(case.expected["verdict"], ["score"])
    )
    assert not check_verdict(blamed, case.expected).passed
    assert check_verdict(honest, case.expected).passed
