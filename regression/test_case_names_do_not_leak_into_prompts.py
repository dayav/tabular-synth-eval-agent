"""The model reads file paths, so the prompt must carry no answer key.

Provenance: recorded (smoke run, 2026-10-15). Prompted under its real path, the
agent wrote that age had been scaled "by a factor of 10" and cited the directory
name ``age_x10``. The number was right and it was still a hallucination: nothing
in the tool results says 10. The checker flagged it, which is how the leak was
noticed. The runner now prompts on neutral copies.
"""

from __future__ import annotations

from pathlib import Path

from evals.cases import Case, neutral_copy
from evals.checks.numeric_faithfulness import check_numeric_faithfulness
from regression.helpers import load_fixture, prompts_for


def test_neutral_copy_prompt_contains_no_case_name(cases: list[Case], tmp_path: Path) -> None:
    names = [c.name for c in cases]
    for case in cases:
        copy = neutral_copy(case, tmp_path)
        prompt = case.prompt(real=copy.real_path, synth=copy.synth_path)
        for name in names:
            assert name not in prompt, f"{name!r} leaked into the prompt: {prompt}"
        for fragment in ("shuffle", "leak", "x10", "missing", "identical", "clean"):
            assert fragment not in prompt.lower()
        assert not any(ch.isdigit() for ch in copy.directory.name), copy.directory.name
        copy.cleanup()


def test_replayed_path_leak_transcript_has_ungrounded_ten(cases: list[Case]) -> None:
    fx = load_fixture("age_x10_path_leak.json")
    case = next(c for c in cases if c.name == fx["case"])
    results = [tc["result"] for tc in fx["tool_calls"]]
    check = check_numeric_faithfulness(fx["report"], results, *prompts_for(case, fx["prompt"]))
    assert not check.passed
    assert "10" in check.details["ungrounded"], check.details
