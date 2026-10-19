"""A threshold that is in neither prompt is an invented number.

Provenance: constructed from the oracle report. The system prompt gives three
thresholds (0.1, 0.2, 0.05). A report that adds "under 0.3 is still acceptable"
must fail numeric faithfulness with 0.3 as the only ungrounded number, even
though every other figure is quoted correctly.
"""

from __future__ import annotations

from typing import Any

from evals.cases import Case
from evals.checks.numeric_faithfulness import check_numeric_faithfulness
from regression.helpers import oracle_report, prompts_for


def test_invented_threshold_is_the_only_ungrounded_number(
    cases: list[Case], tool_outputs: dict[str, dict[str, Any]]
) -> None:
    case = next(c for c in cases if c.name == "clean_synthetic")
    out = tool_outputs[case.name]
    results = [out["js"], out["d"], out["tstr"]]
    report = oracle_report(case, out).replace(
        "\n## Numerical fidelity",
        "\nA JS distance under 0.3 is still acceptable in practice.\n\n## Numerical fidelity",
        1,
    )
    check = check_numeric_faithfulness(report, results, *prompts_for(case))
    assert not check.passed
    assert check.details["ungrounded"] == ["0.3"], check.details


def test_real_distance_does_not_ground_a_coarse_threshold(
    cases: list[Case], tool_outputs: dict[str, dict[str, Any]]
) -> None:
    """The phd_missing education distance (about 0.29) must not ground an invented '0.3'."""
    case = next(c for c in cases if c.name == "phd_missing")
    out = tool_outputs[case.name]
    assert 0.25 < out["js"]["per_column"]["education"] < 0.35
    report = oracle_report(case, out) + "\n\nAnything under 0.3 would have been tolerable."
    check = check_numeric_faithfulness(report, [out["js"], out["d"], out["tstr"]], *prompts_for(case))
    assert check.details["ungrounded"] == ["0.3"], check.details
