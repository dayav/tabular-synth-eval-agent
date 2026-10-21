"""Helpers for the regression suite: the oracle report and fake runs.

The oracle report quotes the tool results verbatim and applies the prompt's
rules. It is the reference answer that must pass every check; if a change to the
checkers makes it fail, the checkers broke, not the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent import SYSTEM_PROMPT, ToolCall
from evals.cases import Case
from evals.checks.verdict import derive_expected
from evals.run_eval import TOOL_DESCRIPTIONS

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TOOL_NAMES = ["js_divergence_categorical", "cohen_d_numerical", "tstr_xgboost"]
TOOL_NAMES_WITH_PRIVACY = [*TOOL_NAMES, "identifiability_heom"]


def prompts_for(case: Case, user_prompt: str | None = None) -> tuple[str, str, str]:
    """The three prompt texts whose numbers are allowed in a report."""
    return SYSTEM_PROMPT, user_prompt or case.prompt(), TOOL_DESCRIPTIONS


def fake_tool_calls(case: Case, outputs: dict[str, Any], *, real: Path | None = None, synth: Path | None = None) -> list[ToolCall]:
    real = real or case.real_path
    synth = synth or case.synth_path
    paths = {"real_csv_path": str(real), "synth_csv_path": str(synth)}
    calls = [
        ToolCall("js_divergence_categorical", dict(paths), outputs["js"]),
        ToolCall("cohen_d_numerical", dict(paths), outputs["d"]),
        ToolCall("tstr_xgboost", {**paths, "target_column": case.target_column}, outputs["tstr"]),
    ]
    if "privacy" in outputs:
        calls.append(ToolCall("identifiability_heom", dict(paths), outputs["privacy"]))
    return calls


def verdict_block(verdict: dict[str, str], flagged: list[str]) -> str:
    return "```json\n" + json.dumps({"verdict": verdict, "flagged_columns": flagged}) + "\n```"


def oracle_report(case: Case, outputs: dict[str, Any]) -> str:
    """A report built directly from the tool results that must pass every check."""
    js, d, tstr = outputs["js"], outputs["d"], outputs["tstr"]
    priv = outputs.get("privacy")
    target = case.target_column
    derived = derive_expected(js, d, tstr, target, priv)
    v, flagged = derived["verdict"], derived["flagged_columns"]

    headline = (f"**Overall verdict: {v['overall']}.** Categorical fidelity is {v['categorical']}, "
                f"numerical fidelity is {v['numerical']}, utility is {v['utility']}")
    headline += f", privacy is {v['privacy']}." if "privacy" in v else "."
    lines = [headline, ""]
    lines.append("## Categorical fidelity (Jensen-Shannon distance, flagged above 0.1)")
    lines.append("JS distance measures how far apart two category frequency tables are; 0 means identical.")
    for col, val in js["per_column"].items():
        mark = " **flagged**" if col in flagged else ""
        lines.append(f"- {col}: {val}{mark}")
    lines.append("")
    lines.append("## Numerical fidelity (Cohen's d, flagged above 0.2)")
    lines.append("Cohen's d is the difference in means in units of the pooled standard deviation.")
    for col, val in d["per_column"].items():
        if col == target:
            lines.append(f"- {col}: {val} (this is the label column, not a feature; not assessed)")
        else:
            mark = " **flagged**" if col in flagged else ""
            lines.append(f"- {col}: {val}{mark}")
    lines.append("")
    lines.append("## Utility (train on synthetic, test on real; poor above 0.05)")
    lines.append(
        f"TSTR accuracy {tstr['tstr_accuracy']} against a TRTR baseline of {tstr['trtr_accuracy']} "
        f"on {tstr['n_test_samples']} held-out real rows gives a utility gap of {tstr['utility_gap']}."
    )
    if tstr["utility_gap"] <= 0:
        lines.append("A gap at or below zero means the synthetic data lost no utility.")
    elif v["utility"] == "poor" and not flagged:
        lines.append(
            "Utility is poor although no column crosses a fidelity threshold: the cause is not "
            "visible in the univariate fidelity metrics, so no single column can be blamed."
        )
    if priv is not None:
        lines.append("")
        lines.append("## Privacy (HEOM nearest neighbours; poor above 0.7 identifiability or 0.05 exact matches)")
        lines.append(
            f"Identifiability {priv['identifiability']}: the fraction of real records whose nearest synthetic "
            f"record is closer than their nearest other real record (an honest sample sits near 0.5). "
            f"Exact matches {priv['exact_matches']} of {priv['n_synthetic']} synthetic rows "
            f"(fraction {priv['exact_match_fraction']}). epsilon_any {priv['epsilon_any']} is supporting evidence only."
        )
    lines.append("")
    lines.append(verdict_block(v, flagged))
    return "\n".join(lines)


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))
