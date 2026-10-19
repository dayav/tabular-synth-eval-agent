"""Harness sanity: the oracle report passes all three checks on every case, an
empty report fails, and missing, duplicated or mistargeted tool calls fail.

Provenance: harness design. If a change to the checkers makes the oracle fail,
the checkers broke, not the model.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent import ToolCall
from evals.cases import Case
from evals.checks.numeric_faithfulness import check_numeric_faithfulness
from evals.checks.tool_sequence import check_tool_sequence
from evals.checks.verdict import check_verdict
from regression.helpers import TOOL_NAMES, fake_tool_calls, oracle_report, prompts_for


@pytest.mark.parametrize("name", ["identical", "clean_synthetic", "income_shuffled", "score_leaks_target", "age_x10", "phd_missing"])
def test_oracle_passes_all_three_checks(name: str, cases: list[Case], tool_outputs: dict[str, dict[str, Any]]) -> None:
    case = next(c for c in cases if c.name == name)
    out = tool_outputs[name]
    calls = fake_tool_calls(case, out)
    report = oracle_report(case, out)
    seq = check_tool_sequence(calls, TOOL_NAMES, case.real_path, case.synth_path, case.target_column)
    num = check_numeric_faithfulness(report, [tc.result for tc in calls], *prompts_for(case))
    ver = check_verdict(report, case.expected)
    assert seq.passed, seq.details
    assert num.passed, num.details
    assert ver.passed, ver.details


def test_empty_report_fails(cases: list[Case]) -> None:
    case = cases[0]
    assert not check_verdict("", case.expected).passed


def test_missing_duplicated_and_mistargeted_calls_fail(cases: list[Case], tool_outputs: dict[str, dict[str, Any]]) -> None:
    case = next(c for c in cases if c.name == "clean_synthetic")
    calls = fake_tool_calls(case, tool_outputs[case.name])
    args = (TOOL_NAMES, case.real_path, case.synth_path, case.target_column)

    assert check_tool_sequence(calls, *args).passed

    missing = calls[:2]
    assert not check_tool_sequence(missing, *args).passed

    duplicated = calls + [calls[0]]
    assert not check_tool_sequence(duplicated, *args).passed

    wrong_target = calls[:2] + [ToolCall("tstr_xgboost", {**calls[2].input, "target_column": "income"}, calls[2].result)]
    assert not check_tool_sequence(wrong_target, *args).passed

    wrong_file = [ToolCall(c.name, {**c.input, "synth_csv_path": str(case.real_path)}, c.result) for c in calls]
    assert not check_tool_sequence(wrong_file, *args).passed

    errored = calls[:2] + [ToolCall("tstr_xgboost", calls[2].input, {"error": "KeyError: 'target'"}, is_error=True)]
    assert not check_tool_sequence(errored, *args).passed
