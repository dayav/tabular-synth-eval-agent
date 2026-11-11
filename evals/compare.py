"""Tabulate every saved eval session.

    python -m evals.compare            # print the sessions table
    python -m evals.compare --write    # also write results/agent_evals.md and the README block

Reads the committed ``evals/results/*/summary.json`` and the answer keys under
``evals/cases/`` only: no model call, no API key. Every figure in the table is
copied or summed from those files, so the table cannot drift from the results.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evals import REPO_ROOT
from evals.cases import CASES_DIR
from evals.run_eval import RESULTS_DIR

OUT_PATH = REPO_ROOT / "results" / "agent_evals.md"
README_PATH = REPO_ROOT / "README.md"
README_START, README_END = "<!-- EVAL TABLE START -->", "<!-- EVAL TABLE END -->"

# Sessions saved before summary.json recorded the agent under test, keyed by the
# timestamp in the folder name. Taken from the "Measured" entries in CHANGELOG.md.
LEGACY_AGENT = {
    "20261018_235325": "3 tools, thresholds in the prompt",
    "20261021_000623": "4 tools (privacy added), thresholds in the prompt",
    "20261021_010217": "5 tools, thresholds retrieved from the docs",
}
# Clarity scored under the first judge prompt, which showed the judge the tool results
# but not the agent's instructions (notes/eval_harness.md, section 10). These two
# folders were never re-scored, so their clarity is not comparable with the other rows.
FIRST_JUDGE_PROMPT = frozenset({"20261018_235325_claude-haiku-5-5", "20261018_235325_claude-sonnet-5-5"})

HEADER = (
    "| session (UTC) | agent under test | model | cases × runs | right tools | numbers traceable to a tool result "
    "| trials with no invented number | right verdict | all three checks | clarity (judge, 1–5) |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def load_sessions(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    """Every ``summary.json`` under ``results_dir``, oldest first, with its folder name added."""
    sessions = []
    for path in sorted(results_dir.glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        s["folder"] = path.parent.name
        sessions.append(s)
    return sessions


def agent_label(s: dict[str, Any]) -> str:
    """Which agent a session measured: recorded in the summary, or looked up for the older folders."""
    agent = s.get("agent")
    if agent:
        return f"{agent['n_tools']} tools at `{agent['git_revision']}`"
    return LEGACY_AGENT.get(s["folder"][:15], "not recorded")


def _frac(passed: int, n: int) -> str:
    if not n:
        return "n/a"
    shown = "100" if passed == n else f"{100 * passed / n:.1f}"
    return f"{shown}% ({passed}/{n})"


def _counts(s: dict[str, Any]) -> dict[str, tuple[int, int]]:
    n = s["n_scored"]
    nf = s["numeric_faithfulness"]
    return {
        "tools": (s["tool_selection"]["passed"], n),
        "numbers": (nf["numbers_grounded"], nf["numbers_total"]),
        "no_invented": (s["runs_with_no_invented_number"]["passed"], n),
        "verdict": (s["verdict"]["passed"], n),
        "all_three": (s["all_three_checks"]["passed"], n),
    }


def sessions_table(sessions: list[dict[str, Any]]) -> str:
    """One row per session and a totals row."""
    if not sessions:
        return HEADER + "\n| (no results yet) |"
    rows = []
    totals = {k: [0, 0] for k in ("tools", "numbers", "no_invented", "verdict", "all_three")}
    for s in sessions:
        counts = _counts(s)
        for key, (passed, n) in counts.items():
            totals[key][0] += passed
            totals[key][1] += n
        clarity = s["clarity"]["mean"]
        clarity_cell = "n/a" if clarity is None else f"{clarity:.2f}" + (" †" if s["folder"] in FIRST_JUDGE_PROMPT else "")
        rows.append(
            f"| {s['timestamp'][:10]} | {agent_label(s)} | {s['model_requested']} "
            f"| {len(s['cases'])} × {s['runs_per_case']} "
            f"| {_frac(*counts['tools'])} | {_frac(*counts['numbers'])} | {_frac(*counts['no_invented'])} "
            f"| {_frac(*counts['verdict'])} | {_frac(*counts['all_three'])} | {clarity_cell} |"
        )
    rows.append(
        f"| **all sessions** | | | **{totals['tools'][1]} trials** "
        f"| {_frac(*totals['tools'])} | {_frac(*totals['numbers'])} | {_frac(*totals['no_invented'])} "
        f"| {_frac(*totals['verdict'])} | {_frac(*totals['all_three'])} | |"
    )
    return HEADER + "\n" + "\n".join(rows)


def per_case_table(sessions: list[dict[str, Any]], cases_dir: Path = CASES_DIR) -> str:
    """Trials pooled over every session, one row per golden case, with the case's answer key."""
    pooled: dict[str, dict[str, int]] = {}
    for s in sessions:
        for name, pc in s["per_case"].items():
            row = pooled.setdefault(name, {"n": 0, "tool_sequence": 0, "numeric_faithfulness": 0, "verdict": 0})
            for key in row:
                row[key] += pc[key]
    lines = [
        "| case | what the synthetic file is | right overall verdict | trials | right tools | no invented number | right verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in sorted(pooled.items()):
        key_path = cases_dir / name / "expected.json"
        expected = json.loads(key_path.read_text(encoding="utf-8")) if key_path.exists() else {}
        n = row["n"]
        lines.append(
            f"| `{name}` | {expected.get('description', '')} | {expected.get('verdict', {}).get('overall', '')} | {n} "
            f"| {row['tool_sequence']}/{n} | {row['numeric_faithfulness']}/{n} | {row['verdict']}/{n} |"
        )
    return "\n".join(lines)


def render_block(sessions: list[dict[str, Any]]) -> str:
    """The table and the sentences computed from it, as spliced into the README."""
    n_trials = sum(s["n_scored"] for s in sessions)
    n_lost = sum(s["n_failed"] for s in sessions)
    by_agent: dict[str, int] = {}
    for s in sessions:
        by_agent[agent_label(s)] = by_agent.get(agent_label(s), 0) + s["n_scored"]
    floors = sorted({(s["n_scored"], s["noise_floor"]) for s in sessions if s["noise_floor"]}, reverse=True)
    last = max(s["timestamp"][:10] for s in sessions) if sessions else "none"
    lines = [
        f"{len(sessions)} sessions, {n_trials} scored trials, last session {last}. A trial is one agent run on one "
        f"golden case. {n_lost} trials were lost to API errors, truncation or refusals.",
        "",
        sessions_table(sessions),
        "",
        "Trials by agent version: " + "; ".join(f"{label}: {n}" for label, n in by_agent.items()) + ".",
    ]
    if floors:
        lines.append("")
        lines.append(
            "Noise floor on a pass rate (1/√n): "
            + ", ".join(f"about ±{100 * floor:.0f} points at {n} trials" for n, floor in floors)
            + "."
        )
    if any(s["folder"] in FIRST_JUDGE_PROMPT for s in sessions):
        lines.append("")
        lines.append(
            "† Scored under the first judge prompt, which did not show the judge the agent's instructions; "
            "not comparable with the undaggered rows."
        )
    return "\n".join(lines)


def render_file(sessions: list[dict[str, Any]]) -> str:
    """The whole of ``results/agent_evals.md``."""
    return "\n".join(
        [
            "# Agent eval results",
            "",
            "Generated by `python -m evals.compare --write` (`make results`) from the committed "
            "[evals/results/](../evals/results/) summaries and the answer keys under [evals/cases/](../evals/cases/). "
            "Do not edit by hand. What the checks are and how to read the numbers is in the "
            "[README](../README.md#the-agent-graded) and [notes/eval_harness.md](../notes/eval_harness.md).",
            "",
            "## Every session",
            "",
            render_block(sessions),
            "",
            "## Every golden case, trials pooled over all sessions",
            "",
            per_case_table(sessions),
            "",
            "## Source folders",
            "",
            *[f"- [evals/results/{s['folder']}/](../evals/results/{s['folder']}/summary.md)" for s in sessions],
            "",
        ]
    )


def splice_readme(block: str, readme: Path = README_PATH) -> None:
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise SystemExit("README has no eval table markers")
    pattern = re.escape(README_START) + r".*?" + re.escape(README_END)
    readme.write_text(
        re.sub(pattern, lambda _: f"{README_START}\n{block}\n{README_END}", text, flags=re.DOTALL), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write results/agent_evals.md and the README block")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)

    sessions = load_sessions(args.results_dir)
    print(sessions_table(sessions))
    if args.write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(render_file(sessions), encoding="utf-8")
        splice_readme(render_block(sessions))
        print(f"\nwrote {OUT_PATH.relative_to(REPO_ROOT)} and the README block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
