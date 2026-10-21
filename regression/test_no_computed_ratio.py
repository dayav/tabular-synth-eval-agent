"""A ratio the model computed itself is an invented number, even when it is right.

Provenance: recorded live on 2026-10-18 (eval run 20261018_235325_claude-sonnet-4-6,
transcript age_x10_2). claude-sonnet-4-6 reported age's Cohen's d of 4.909 correctly
and then wrote that it was "more than 24 times the flagging threshold of 0.2".
The arithmetic is right (4.909 / 0.2 = 24.5); the number is still the model's, not
a tool's, and the prompt forbids computing new figures. The checker flagged it;
this test keeps it flagged.
"""

from __future__ import annotations

from evals.cases import Case
from evals.checks.numeric_faithfulness import check_numeric_faithfulness
from evals.checks.verdict import check_verdict
from regression.helpers import load_fixture, prompts_for, replay_results


def test_computed_ratio_is_the_only_ungrounded_number(cases: list[Case]) -> None:
    fx = load_fixture("age_x10_computed_ratio.json")
    case = next(c for c in cases if c.name == fx["case"])
    results = replay_results(fx)
    check = check_numeric_faithfulness(fx["report"], results, *prompts_for(case, fx["prompt"]))
    assert not check.passed
    assert check.details["ungrounded"] == ["24"], check.details
    # everything else about the run was right: the verdict check passes on the same report
    assert check_verdict(fx["report"], case.expected).passed
