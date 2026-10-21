"""Golden consistency: every expected.json agrees with the threshold rules applied
to real tool output.

Provenance: harness design. The answer keys are written from how each synthetic
file was built; this test applies ``derive_expected`` to what the tools actually
return, so a change to the generator, the tools or the thresholds is caught
before a model is blamed for a stale key.
"""

from __future__ import annotations

from typing import Any

import pytest

from evals.cases import Case
from evals.checks.verdict import derive_expected

CASE_NAMES = [
    "identical", "clean_synthetic", "income_shuffled", "score_leaks_target", "age_x10", "phd_missing",
    "identical_privacy", "clean_privacy",
]


def test_all_cases_exist(cases: list[Case]) -> None:
    assert sorted(c.name for c in cases) == sorted(CASE_NAMES)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_expected_matches_derived(name: str, cases: list[Case], tool_outputs: dict[str, dict[str, Any]]) -> None:
    case = next(c for c in cases if c.name == name)
    out = tool_outputs[name]
    derived = derive_expected(out["js"], out["d"], out["tstr"], case.target_column, out.get("privacy"))
    assert derived["verdict"] == case.expected["verdict"], (derived, case.expected["notes"])
    assert derived["flagged_columns"] == sorted(case.expected["flagged_columns"])


def test_cases_tell_the_intended_story(tool_outputs: dict[str, dict[str, Any]]) -> None:
    """The constructions must produce the effects the note relies on."""
    clean, shuffled, leaky = tool_outputs["clean_synthetic"], tool_outputs["income_shuffled"], tool_outputs["score_leaks_target"]
    # a permutation keeps the marginal: same Cohen's d on income as the control
    assert shuffled["d"]["per_column"]["income"] == pytest.approx(clean["d"]["per_column"]["income"])
    # ... and destroys the joint: utility collapses
    assert shuffled["tstr"]["utility_gap"] > 0.1
    # the leaky score keeps mean and spread (d under 0.2) but ruins utility
    assert leaky["d"]["per_column"]["score"] < 0.2
    assert leaky["tstr"]["utility_gap"] > 0.1
    # the loud cases are loud
    assert tool_outputs["age_x10"]["d"]["per_column"]["age"] > 0.8
    assert tool_outputs["phd_missing"]["js"]["per_column"]["education"] > 0.1
    # privacy: the copy is maximally identifiable, the honest control sits near 0.5 with no copies
    copy, honest = tool_outputs["identical_privacy"]["privacy"], tool_outputs["clean_privacy"]["privacy"]
    assert copy["identifiability"] == 1.0 and copy["exact_match_fraction"] == 1.0
    assert 0.35 < honest["identifiability"] < 0.65 and honest["exact_matches"] == 0
    # ... and epsilon_any on the honest control is high, which is why it is not thresholded
    assert honest["epsilon_any"] > 0.5
