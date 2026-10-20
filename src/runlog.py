"""Run logging: one folder per agent run with inputs, trace, latency, tokens, cost and report.

Layout of ``<root>/<run_id>/``::

    meta.json    who, when, what: model requested and served, prompt, input files
                 (path, bytes, sha256, rows, columns), target column, status, timings
    trace.json   one entry per API step: latency, tokens, request id, stop reason,
                 and the tool calls that step triggered with their own latency and result
    usage.json   token totals and the estimated cost with the pricing date it rests on
    report.md    the report text the agent returned

``runs/summarize.py`` prints a table across run folders. ``run_to_dict`` is the
shared serialiser the eval harness uses for its transcripts, so a transcript
and a run folder describe a run in the same words.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pricing import estimate_cost

if TYPE_CHECKING:
    from agent import AgentRun

DEFAULT_RUNS_DIR = Path("runs")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_csv(path: str | Path, role: str) -> dict[str, Any]:
    """Size, hash and shape of one input file. Never raises: a missing file is recorded as such."""
    p = Path(path)
    info: dict[str, Any] = {"role": role, "path": str(p)}
    if not p.exists():
        info["error"] = "file not found"
        return info
    info["bytes"] = p.stat().st_size
    info["sha256"] = _sha256(p)
    try:
        import pandas as pd

        frame = pd.read_csv(p)
        info["rows"] = int(frame.shape[0])
        info["columns"] = int(frame.shape[1])
        info["column_names"] = [str(c) for c in frame.columns]
    except Exception as e:  # noqa: BLE001 - metadata must not fail the run
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def input_metadata(run: AgentRun) -> dict[str, Any]:
    """Input files and target column, taken from the tool calls the agent actually made."""
    seen: dict[str, dict[str, Any]] = {}
    target: str | None = None
    for tc in run.tool_calls:
        for key, role in (("real_csv_path", "real"), ("synth_csv_path", "synthetic"), ("test_csv_path", "test")):
            path = tc.input.get(key)
            if path and str(path) not in seen:
                seen[str(path)] = describe_csv(str(path), role)
        if target is None and tc.input.get("target_column"):
            target = str(tc.input["target_column"])
    return {"files": list(seen.values()), "target_column": target}


def git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def run_to_dict(run: AgentRun) -> dict[str, Any]:
    """Serialise a run without the raw SDK message objects."""
    cost = estimate_cost(
        run.model_served or run.model_requested,
        input_tokens=run.usage.input_tokens,
        output_tokens=run.usage.output_tokens,
        cache_read_input_tokens=run.usage.cache_read_input_tokens,
        cache_creation_input_tokens=run.usage.cache_creation_input_tokens,
    )
    return {
        "run_id": run.run_id,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "latency_s": run.latency_s,
        "model_requested": run.model_requested,
        "model_served": run.model_served,
        "stop_reason": run.stop_reason,
        "status": "error" if run.error else "ok",
        "error": run.error,
        "turns": len(run.steps),
        "prompt": run.user_prompt,
        "steps": [asdict(s) for s in run.steps],
        "tool_calls": [asdict(tc) for tc in run.tool_calls],
        "usage": asdict(run.usage),
        "cost": cost.to_dict(),
        "report": run.text,
    }


def write_run(run: AgentRun, root: str | Path = DEFAULT_RUNS_DIR) -> Path:
    """Write ``<root>/<run.run_id>/{meta,trace,usage}.json + report.md``. Returns the folder."""
    folder = Path(root) / run.run_id
    folder.mkdir(parents=True, exist_ok=True)
    d = run_to_dict(run)

    meta = {
        k: d[k]
        for k in (
            "run_id", "started_at", "finished_at", "latency_s", "model_requested", "model_served",
            "stop_reason", "status", "error", "turns", "prompt",
        )
    }
    meta["inputs"] = input_metadata(run)
    meta["tools_called"] = [tc.name for tc in run.tool_calls]
    meta["git_revision"] = git_revision()
    meta["system_prompt_sha256"] = hashlib.sha256(run.system_prompt.encode("utf-8")).hexdigest()[:12]

    # trace: steps with the tool calls each one triggered
    calls_by_turn: dict[int, list[dict[str, Any]]] = {}
    for tc in d["tool_calls"]:
        calls_by_turn.setdefault(int(tc["turn"]), []).append(tc)
    trace = [{**step, "tool_calls": calls_by_turn.get(int(step["turn"]), [])} for step in d["steps"]]

    usage = {"usage": d["usage"], "cost": d["cost"], "latency_s": d["latency_s"],
             "tool_latency_s": round(sum(tc.latency_s for tc in run.tool_calls), 3),
             "api_latency_s": round(sum(s.latency_s for s in run.steps), 3)}

    (folder / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n", encoding="utf-8")
    (folder / "trace.json").write_text(json.dumps(trace, indent=1, default=str) + "\n", encoding="utf-8")
    (folder / "usage.json").write_text(json.dumps(usage, indent=1) + "\n", encoding="utf-8")
    (folder / "report.md").write_text(run.text, encoding="utf-8")
    return folder
