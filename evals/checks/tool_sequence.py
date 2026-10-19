"""Check 1: did the agent call the right tools, on the right files, with the right target?

The three tools are independent, so their order is not asserted; the check grades
the outcome, not the path.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from evals.checks.common import CheckResult


def _same_file(a: Any, b: Path) -> bool:
    try:
        return Path(str(a)).resolve() == b.resolve()
    except OSError:
        return False


def check_tool_sequence(
    tool_calls: list[Any],
    expected_tools: list[str],
    real_path: Path,
    synth_path: Path,
    target_column: str,
) -> CheckResult:
    """``tool_calls`` are ``AgentRun.tool_calls`` (or anything with name/input/is_error)."""
    counts = Counter(tc.name for tc in tool_calls)
    problems: list[str] = []

    for tool in expected_tools:
        n = counts.get(tool, 0)
        if n == 0:
            problems.append(f"{tool} was never called (its section of the report is unsupported)")
        elif n > 1:
            problems.append(f"{tool} was called {n} times")
    for tool, n in counts.items():
        if tool not in expected_tools:
            problems.append(f"unexpected tool {tool} called {n} time(s)")

    for tc in tool_calls:
        if not _same_file(tc.input.get("real_csv_path"), real_path):
            problems.append(f"{tc.name}: real_csv_path={tc.input.get('real_csv_path')!r} is not the case's real file")
        if not _same_file(tc.input.get("synth_csv_path"), synth_path):
            problems.append(f"{tc.name}: synth_csv_path={tc.input.get('synth_csv_path')!r} is not the case's synthetic file")
        if tc.name == "tstr_xgboost" and tc.input.get("target_column") != target_column:
            problems.append(f"tstr_xgboost: target_column={tc.input.get('target_column')!r}, expected {target_column!r}")
        if getattr(tc, "is_error", False):
            problems.append(f"{tc.name} returned an error: {tc.result}")

    return CheckResult(
        name="tool_sequence",
        passed=not problems,
        details={"calls": [tc.name for tc in tool_calls], "problems": problems},
    )
