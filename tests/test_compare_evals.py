"""The eval results table: totals are summed from the summaries, and the README block is replaced in place."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals import compare


def _summary(model: str, n: int, *, invented_in: int = 0, agent: dict[str, Any] | None = None) -> dict[str, Any]:
    ok = {"passed": n, "n": n, "rate": 1.0}
    s: dict[str, Any] = {
        "model_requested": model,
        "timestamp": "2026-11-01T10:00:00+00:00",
        "cases": ["identical", "made_up_case"],
        "runs_per_case": n // 2,
        "n_scored": n,
        "n_failed": 0,
        "noise_floor": 0.5,
        "tool_selection": ok,
        "numeric_faithfulness": {"numbers_grounded": 10 * n - invented_in, "numbers_total": 10 * n, "rate": 0.0},
        "runs_with_no_invented_number": {"passed": n - invented_in, "n": n, "rate": 0.0},
        "verdict": ok,
        "all_three_checks": {"passed": n - invented_in, "n": n, "rate": 0.0},
        "clarity": {"mean": None, "n_judged": 0},
        "per_case": {
            "identical": {"n": n // 2, "tool_sequence": n // 2, "numeric_faithfulness": n // 2 - invented_in, "verdict": n // 2},
            "made_up_case": {"n": n // 2, "tool_sequence": n // 2, "numeric_faithfulness": n // 2, "verdict": n // 2},
        },
    }
    if agent:
        s["agent"] = agent
    return s


def _write(results_dir: Path, folder: str, summary: dict[str, Any]) -> None:
    (results_dir / folder).mkdir(parents=True)
    (results_dir / folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_table_sums_sessions_and_labels_the_agent(tmp_path: Path) -> None:
    _write(tmp_path, "20261018_235325_model-a", _summary("model-a", 4, invented_in=1))
    _write(tmp_path, "20261101_100000_model-b", _summary("model-b", 2, agent={"git_revision": "abc1234", "n_tools": 5}))
    sessions = compare.load_sessions(tmp_path)

    assert [compare.agent_label(s) for s in sessions] == ["3 tools, thresholds in the prompt", "5 tools at `abc1234`"]
    table = compare.sessions_table(sessions)
    assert "| 75.0% (3/4) | 100% (4/4) | 75.0% (3/4) | n/a |" in table  # one invented number fails that trial only
    assert "| **all sessions** | | | **6 trials** | 100% (6/6) | 98.3% (59/60) | 83.3% (5/6) | 100% (6/6) | 83.3% (5/6) | |" in table

    block = compare.render_block(sessions)
    assert block.startswith("2 sessions, 6 scored trials, last session 2026-11-01.")
    assert "3 tools, thresholds in the prompt: 4; 5 tools at `abc1234`: 2." in block

    per_case = compare.per_case_table(sessions)
    assert "| `identical` | The synthetic file is an exact copy of the real file. | excellent | 3 | 3/3 | 2/3 | 3/3 |" in per_case
    assert "| `made_up_case` |  |  | 3 | 3/3 | 3/3 | 3/3 |" in per_case  # a case with no answer key on disk still gets a row


def test_readme_block_is_replaced_between_the_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{compare.README_START}\nold table\n{compare.README_END}\nafter\n", encoding="utf-8")
    compare.splice_readme("new \\1 table", readme)  # a backslash in the block must survive the substitution
    assert readme.read_text(encoding="utf-8") == (
        f"before\n{compare.README_START}\nnew \\1 table\n{compare.README_END}\nafter\n"
    )
