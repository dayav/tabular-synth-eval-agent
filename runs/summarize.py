#!/usr/bin/env python
"""Print one row per logged run: ``python runs/summarize.py [--runs DIR] [--json]``.

Reads only ``meta.json``, ``usage.json`` and the verdict block of ``report.md``,
so it has no dependency on the agent code and works on any machine the folders
were copied to.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

VERDICT_RE = re.compile(r"\{\s*\"verdict\"")


def overall_verdict(report: str) -> str:
    """The ``overall`` value of the last verdict block in a report, or ``-``."""
    decoder = json.JSONDecoder()
    found = "-"
    for m in VERDICT_RE.finditer(report):
        try:
            obj, _ = decoder.raw_decode(report, m.start())
        except json.JSONDecodeError:
            continue
        found = str(obj.get("verdict", {}).get("overall", "-"))
    return found


def load_run(folder: Path) -> dict[str, Any] | None:
    meta_path, usage_path = folder / "meta.json", folder / "usage.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    usage = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.exists() else {}
    report = (folder / "report.md").read_text(encoding="utf-8") if (folder / "report.md").exists() else ""
    files = meta.get("inputs", {}).get("files", [])
    synth = next((f for f in files if f.get("role") == "synthetic"), {})
    tokens = usage.get("usage", {})
    cost = usage.get("cost", {})
    return {
        "run_id": meta.get("run_id", folder.name),
        "started_at": meta.get("started_at", ""),
        "model": meta.get("model_served") or meta.get("model_requested", ""),
        "status": meta.get("status", ""),
        "synthetic": Path(synth.get("path", "")).name if synth else "",
        "rows": synth.get("rows"),
        "tools": len(meta.get("tools_called", [])),
        "turns": meta.get("turns"),
        "latency_s": meta.get("latency_s"),
        "api_latency_s": usage.get("api_latency_s"),
        "tool_latency_s": usage.get("tool_latency_s"),
        "tokens_in": tokens.get("input_tokens"),
        "tokens_out": tokens.get("output_tokens"),
        "cache_read": tokens.get("cache_read_input_tokens"),
        "cost_usd": cost.get("total_usd") if cost.get("priced") else None,
        "pricing_date": cost.get("pricing_date"),
        "verdict": overall_verdict(report),
    }


def _fmt(v: Any, spec: str = "") -> str:
    if v is None or v == "":
        return "-"
    return format(v, spec) if spec else str(v)


def table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| run | started (UTC) | model | status | synthetic file | rows | tools | turns "
        "| latency s (api / tools) | tokens in / out | cache read | est. cost USD | verdict |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['run_id']} | {r['started_at'].replace('T', ' ').replace('+00:00', '')} | {r['model']} | {r['status']} "
            f"| {r['synthetic'] or '-'} | {_fmt(r['rows'])} | {_fmt(r['tools'])} | {_fmt(r['turns'])} "
            f"| {_fmt(r['latency_s'], '.1f')} ({_fmt(r['api_latency_s'], '.1f')} / {_fmt(r['tool_latency_s'], '.1f')}) "
            f"| {_fmt(r['tokens_in'])} / {_fmt(r['tokens_out'])} | {_fmt(r['cache_read'])} "
            f"| {_fmt(r['cost_usd'], '.4f')} | {r['verdict']} |"
        )
    priced = [r for r in rows if r["cost_usd"] is not None]
    if priced:
        total = sum(r["cost_usd"] for r in priced)
        dates = sorted({r["pricing_date"] for r in priced if r["pricing_date"]})
        lines.append("")
        lines.append(
            f"{len(rows)} run(s); estimated total {total:.4f} USD over {len(priced)} priced run(s), "
            f"list prices as of {', '.join(dates)}."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path(__file__).resolve().parent, help="folder holding run folders")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args(argv)
    rows = [r for r in (load_run(d) for d in sorted(args.runs.iterdir()) if d.is_dir()) if r]
    if args.json:
        print(json.dumps(rows, indent=1))
    elif rows:
        print(table(rows))
    else:
        print(f"no runs under {args.runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
