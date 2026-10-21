"""Thresholds come from the metric documentation, not from the prompt or from memory.

Provenance: phase 5 design. The system prompt no longer carries any threshold; the agent
must retrieve them with lookup_metric_guidance. This file pins four things: the docs,
the agent constants and the checker constants agree; the prompt contains no threshold
number; a report that quotes a threshold without a lookup result fails numeric
faithfulness while the same report with the lookup result passes; and the tool-sequence
check requires at least one lookup and allows several.
"""

from __future__ import annotations

from typing import Any

import agent
from evals.cases import Case
from evals.checks import verdict as verdict_rules
from evals.checks.common import NUMBER_RE, parse_number_token
from evals.checks.numeric_faithfulness import check_numeric_faithfulness
from evals.checks.tool_sequence import check_tool_sequence
from guidance import DOCS_DIR, lookup, thresholds
from regression.helpers import TOOL_NAMES, fake_tool_calls, guidance_calls, oracle_report, prompts_for


def test_docs_agent_and_checker_agree() -> None:
    t = thresholds()
    assert agent.JS_THRESHOLD == verdict_rules.JS_THRESHOLD == t["js_distance"]["threshold"]
    assert agent.COHEN_D_THRESHOLD == verdict_rules.COHEN_D_THRESHOLD == t["cohens_d"]["threshold"]
    assert agent.UTILITY_GAP_THRESHOLD == verdict_rules.UTILITY_GAP_THRESHOLD == t["utility_gap"]["threshold"]
    assert agent.IDENTIFIABILITY_THRESHOLD == verdict_rules.IDENTIFIABILITY_THRESHOLD == t["identifiability"]["threshold"]
    assert agent.EXACT_MATCH_THRESHOLD == verdict_rules.EXACT_MATCH_THRESHOLD == t["identifiability"]["exact_match_threshold"]
    assert {p.stem for p in DOCS_DIR.glob("*.md")} >= {"js_distance", "cohens_d", "utility_gap", "identifiability", "verdict_rules"}


def test_system_prompt_carries_no_threshold() -> None:
    numbers = {parse_number_token(tok)[0] for tok in NUMBER_RE.findall(agent.SYSTEM_PROMPT)}
    for value in (0.1, 0.2, 0.05, 0.7, 0.5):
        assert value not in numbers, f"{value} is in the system prompt; thresholds must come from the docs"


def test_threshold_without_lookup_is_invented(cases: list[Case], tool_outputs: dict[str, dict[str, Any]]) -> None:
    case = next(c for c in cases if c.name == "clean_synthetic")
    out = tool_outputs[case.name]
    report = oracle_report(case, out)  # quotes 0.1, 0.2 and 0.05 with their doc sources
    metric_results = [out["js"], out["d"], out["tstr"]]
    without = check_numeric_faithfulness(report, metric_results, *prompts_for(case))
    assert not without.passed
    # 0.1 and 0.2 are invented without a lookup. 0.05 happens to be grounded on this case by
    # truncation of score's Cohen's d (0.0567): the known two-decimal collision, section 5.2.
    assert {"0.1", "0.2"} <= set(without.details["ungrounded"]), without.details
    with_lookup = check_numeric_faithfulness(
        report, metric_results + [tc.result for tc in guidance_calls(with_privacy=False)], *prompts_for(case)
    )
    assert with_lookup.passed, with_lookup.details


def test_lookup_is_required_and_may_repeat(cases: list[Case], tool_outputs: dict[str, dict[str, Any]]) -> None:
    case = next(c for c in cases if c.name == "clean_synthetic")
    calls = fake_tool_calls(case, tool_outputs[case.name])
    args = (TOOL_NAMES, case.real_path, case.synth_path, case.target_column)
    assert check_tool_sequence(calls, *args).passed
    assert sum(1 for c in calls if c.name == "lookup_metric_guidance") == 3  # one per metric, repeats allowed
    no_lookup = [c for c in calls if c.name != "lookup_metric_guidance"]
    check = check_tool_sequence(no_lookup, *args)
    assert not check.passed and any("lookup_metric_guidance was never called" in p for p in check.details["problems"])


def test_lookup_returns_the_right_section() -> None:
    for metric, query in (("js_distance", "JS distance threshold"), ("cohens_d", "Cohen's d threshold"),
                          ("utility_gap", "utility gap threshold"), ("identifiability", "identifiability threshold")):
        hit = lookup(query, top_k=1)["results"][0]
        assert hit["metric"] == metric and hit["section"] == "Threshold", (query, hit)
        assert hit["threshold"] == thresholds()[metric]["threshold"]
