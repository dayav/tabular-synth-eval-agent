"""Print one row per saved results folder: ``python -m evals.compare``."""

from __future__ import annotations

import json
from pathlib import Path

from evals.run_eval import RESULTS_DIR, _pct


def main(results_dir: Path = RESULTS_DIR) -> None:
    rows = []
    for summary_path in sorted(results_dir.glob("*/summary.json")):
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            "| {d} | {m} | {n} | {ts} | {nf} | {nr} | {vd} | {a3} | {cl} | {ti}k / {to}k |".format(
                d=summary_path.parent.name,
                m=s["model_requested"],
                n=s["n_scored"],
                ts=_pct(s["tool_selection"]["rate"]),
                nf=_pct(s["numeric_faithfulness"]["rate"]),
                nr=_pct(s["runs_with_no_invented_number"]["rate"]),
                vd=_pct(s["verdict"]["rate"]),
                a3=_pct(s["all_three_checks"]["rate"]),
                cl=f"{s['clarity']['mean']:.2f}" if s["clarity"]["mean"] is not None else "n/a",
                ti=round(s["agent_tokens"]["input"] / 1000),
                to=round(s["agent_tokens"]["output"] / 1000),
            )
        )
    header = (
        "| results folder | model | n | tool-selection | numeric faithfulness | no invented number "
        "| verdict | all three | clarity | agent tokens in / out |\n|---|---|---|---|---|---|---|---|---|---|"
    )
    print(header)
    print("\n".join(rows) if rows else "| (no results yet) |")


if __name__ == "__main__":
    main()
