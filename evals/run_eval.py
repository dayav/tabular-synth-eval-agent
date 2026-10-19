"""Run the agent on every golden case N times and score each run.

    python -m evals.run_eval --model claude-sonnet-4-6 --runs 3 [--judge claude-opus-5-5 | --no-judge]

Writes ``evals/results/<timestamp>_<model>/`` with ``summary.json``,
``summary.md``, one transcript per run under ``transcripts/`` and plumbing
failures (API errors, truncation, refusals, turn limit) in ``errors.jsonl``.
Plumbing failures are excluded from the metrics but counted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import evals  # noqa: F401  (puts src/ on sys.path)
from agent import DEFAULT_MODEL, SYSTEM_PROMPT, TOOLS, AgentRun, get_client, run_agent  # noqa: E402
from evals.cases import Case, load_cases, neutral_copy
from evals.checks.judge import DEFAULT_JUDGE_MODEL, RUBRIC_HASH, judge_report
from evals.checks.numeric_faithfulness import check_numeric_faithfulness
from evals.checks.tool_sequence import check_tool_sequence
from evals.checks.verdict import check_verdict

RESULTS_DIR = Path(__file__).resolve().parent / "results"
AGENT_CONTEXT = SYSTEM_PROMPT + "\n\nTool descriptions:\n" + "\n".join(
    f"- {t['name']}: {t['description']}" for t in TOOLS
)
TOOL_DESCRIPTIONS = "\n".join(
    str(t["description"])
    + " "
    + " ".join(str(p.get("description", "")) for p in t["input_schema"].get("properties", {}).values())
    for t in TOOLS
)


def classify_failure(run: AgentRun | None, exc: BaseException | None) -> str | None:
    """Return a plumbing-failure class, or ``None`` when the run is scorable."""
    if exc is not None:
        return "api_error"
    assert run is not None
    if run.stop_reason == "max_tokens":
        return "truncated"
    if run.stop_reason == "refusal":
        return "refusal"
    if run.stop_reason == "max_turns":
        return "turn_limit"
    if not run.text.strip():
        return "empty_response"
    return None


def score_one(
    case: Case,
    run_index: int,
    *,
    model: str,
    judge_model: str | None,
    tmp_parent: Path,
) -> dict[str, Any]:
    copy = neutral_copy(case, tmp_parent)
    started = time.perf_counter()
    run: AgentRun | None = None
    exc: BaseException | None = None
    try:
        prompt = case.prompt(real=copy.real_path, synth=copy.synth_path)
        run = run_agent(prompt, model=model)
    except Exception as e:  # noqa: BLE001
        exc = e
    latency = time.perf_counter() - started

    record: dict[str, Any] = {
        "case": case.name,
        "run_index": run_index,
        "model_requested": model,
        "latency_s": round(latency, 2),
    }
    failure = classify_failure(run, exc)
    if failure is not None:
        record["failure"] = failure
        record["error"] = "".join(traceback.format_exception_only(exc)) if exc else (run.stop_reason if run else "")
        if run is not None:
            record["usage"] = asdict(run.usage)
            record["report"] = run.text
        copy.cleanup()
        return record

    assert run is not None
    tool_results = [tc.result for tc in run.tool_calls]
    checks = {
        "tool_sequence": check_tool_sequence(
            run.tool_calls, case.expected_tools, copy.real_path, copy.synth_path, case.target_column
        ),
        "numeric_faithfulness": check_numeric_faithfulness(
            run.text, tool_results, run.system_prompt, run.user_prompt, TOOL_DESCRIPTIONS
        ),
        "verdict": check_verdict(run.text, case.expected),
    }
    record.update(
        {
            "model_served": run.model_served,
            "stop_reason": run.stop_reason,
            "prompt": run.user_prompt,
            "tool_calls": [
                {"name": tc.name, "input": tc.input, "result": tc.result, "is_error": tc.is_error}
                for tc in run.tool_calls
            ],
            "report": run.text,
            "usage": asdict(run.usage),
            "checks": {k: v.to_dict() for k, v in checks.items()},
        }
    )
    if judge_model:
        try:
            record["judge"] = judge_report(
                run.text, tool_results, client=get_client(), model=judge_model,
                agent_context=AGENT_CONTEXT,
            ).to_dict()
        except Exception as e:  # noqa: BLE001
            record["judge_error"] = "".join(traceback.format_exception_only(e))
    copy.cleanup()
    return record


def summarise(records: list[dict[str, Any]], *, model: str, judge_model: str | None, runs: int) -> dict[str, Any]:
    scored = [r for r in records if "failure" not in r]
    failed = [r for r in records if "failure" in r]
    n = len(scored)

    def rate(key: str) -> tuple[int, int]:
        return sum(1 for r in scored if r["checks"][key]["passed"]), n

    ts_pass, _ = rate("tool_sequence")
    nf_runs_pass, _ = rate("numeric_faithfulness")
    vd_pass, _ = rate("verdict")
    all3 = sum(1 for r in scored if all(c["passed"] for c in r["checks"].values()))
    numbers_total = sum(r["checks"]["numeric_faithfulness"]["numbers_total"] for r in scored)
    numbers_grounded = sum(r["checks"]["numeric_faithfulness"]["numbers_grounded"] for r in scored)
    judged = [r["judge"] for r in scored if "judge" in r]
    clarity = sum(j["mean"] for j in judged) / len(judged) if judged else None

    failures: dict[str, int] = {}
    for r in failed:
        failures[r["failure"]] = failures.get(r["failure"], 0) + 1

    per_case: dict[str, dict[str, Any]] = {}
    for r in scored:
        pc = per_case.setdefault(
            r["case"], {"n": 0, "tool_sequence": 0, "numeric_faithfulness": 0, "verdict": 0, "ungrounded": []}
        )
        pc["n"] += 1
        for k in ("tool_sequence", "numeric_faithfulness", "verdict"):
            pc[k] += int(r["checks"][k]["passed"])
        pc["ungrounded"].extend(r["checks"]["numeric_faithfulness"]["ungrounded"])

    return {
        "model_requested": model,
        "models_served": sorted({r.get("model_served", "") for r in scored}),
        "judge_model": judge_model,
        "rubric_hash": RUBRIC_HASH if judge_model else None,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "cases": sorted({r["case"] for r in records}),
        "runs_per_case": runs,
        "n_scored": n,
        "n_failed": len(failed),
        "failures": failures,
        "noise_floor": round(1 / math.sqrt(n), 3) if n else None,
        "tool_selection": {"passed": ts_pass, "n": n, "rate": ts_pass / n if n else None},
        "numeric_faithfulness": {
            "numbers_grounded": numbers_grounded,
            "numbers_total": numbers_total,
            "rate": numbers_grounded / numbers_total if numbers_total else None,
        },
        "runs_with_no_invented_number": {"passed": nf_runs_pass, "n": n, "rate": nf_runs_pass / n if n else None},
        "verdict": {"passed": vd_pass, "n": n, "rate": vd_pass / n if n else None},
        "all_three_checks": {"passed": all3, "n": n, "rate": all3 / n if n else None},
        "clarity": {"mean": clarity, "n_judged": len(judged)},
        "agent_tokens": {
            "input": sum(r["usage"]["input_tokens"] for r in records if "usage" in r),
            "output": sum(r["usage"]["output_tokens"] for r in records if "usage" in r),
        },
        "judge_tokens": {
            "input": sum(j["input_tokens"] for j in judged),
            "output": sum(j["output_tokens"] for j in judged),
        },
        "mean_latency_s": round(sum(r["latency_s"] for r in scored) / n, 1) if n else None,
        "per_case": per_case,
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def summary_markdown(s: dict[str, Any]) -> str:
    noise = s["noise_floor"]
    lines = [
        f"# Eval summary: {s['model_requested']}",
        "",
        f"Run {s['timestamp']} on {len(s['cases'])} cases x {s['runs_per_case']} runs; "
        f"{s['n_scored']} scored, {s['n_failed']} plumbing failures {s['failures'] or ''}. "
        f"Served: {', '.join(s['models_served'])}. Judge: {s['judge_model'] or 'none'}"
        + (f" (rubric {s['rubric_hash']})" if s["rubric_hash"] else "")
        + ".",
        "",
        "| model | tool-selection | numeric faithfulness | runs with no invented number | verdict | all three checks | clarity (1-5) | agent tokens in / out |",
        "|---|---|---|---|---|---|---|---|",
        "| {m} | {ts} ({tsp}/{n}) | {nf} ({nfg}/{nft}) | {nr} ({nrp}/{n}) | {vd} ({vdp}/{n}) | {a3} | {cl} | {ti}k / {to}k |".format(
            m=s["model_requested"],
            ts=_pct(s["tool_selection"]["rate"]),
            tsp=s["tool_selection"]["passed"],
            n=s["n_scored"],
            nf=_pct(s["numeric_faithfulness"]["rate"]),
            nfg=s["numeric_faithfulness"]["numbers_grounded"],
            nft=s["numeric_faithfulness"]["numbers_total"],
            nr=_pct(s["runs_with_no_invented_number"]["rate"]),
            nrp=s["runs_with_no_invented_number"]["passed"],
            vd=_pct(s["verdict"]["rate"]),
            vdp=s["verdict"]["passed"],
            a3=_pct(s["all_three_checks"]["rate"]),
            cl=f"{s['clarity']['mean']:.2f}" if s["clarity"]["mean"] is not None else "n/a",
            ti=round(s["agent_tokens"]["input"] / 1000),
            to=round(s["agent_tokens"]["output"] / 1000),
        ),
        "",
        f"Noise floor on a pass rate at n={s['n_scored']}: about +/-{100 * noise:.0f} points." if noise else "",
        "",
        "## Per case",
        "",
        "| case | n | tool-selection | no invented number | verdict | ungrounded numbers |",
        "|---|---|---|---|---|---|",
    ]
    for name, pc in sorted(s["per_case"].items()):
        lines.append(
            f"| {name} | {pc['n']} | {pc['tool_sequence']}/{pc['n']} | {pc['numeric_faithfulness']}/{pc['n']} "
            f"| {pc['verdict']}/{pc['n']} | {', '.join(pc['ungrounded']) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def rejudge(results_dir: Path, judge_model: str, workers: int) -> dict[str, Any]:
    """Re-score the judge on saved transcripts (after a rubric or judge change) and rewrite the summary.

    The agent runs are not repeated, so the mechanical checks are unchanged.
    """
    paths = sorted((results_dir / "transcripts").glob("*.json"))
    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    errors_path = results_dir / "errors.jsonl"
    failed = [json.loads(line) for line in errors_path.read_text(encoding="utf-8").splitlines()] if errors_path.exists() else []
    client = get_client()

    def one(i: int) -> None:
        rec = records[i]
        tool_results = [tc["result"] for tc in rec["tool_calls"]]
        rec.pop("judge_error", None)
        try:
            rec["judge"] = judge_report(rec["report"], tool_results, client=client, model=judge_model, agent_context=AGENT_CONTEXT).to_dict()
        except Exception as e:  # noqa: BLE001
            rec["judge_error"] = "".join(traceback.format_exception_only(e))
        paths[i].write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, range(len(records))))

    old = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    summary = summarise(records + failed, model=old["model_requested"], judge_model=judge_model, runs=old["runs_per_case"])
    summary["timestamp"] = old["timestamp"]
    summary["rejudged_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (results_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--judge", default=DEFAULT_JUDGE_MODEL, help="judge model id")
    ap.add_argument("--no-judge", action="store_true", help="mechanical checks only")
    ap.add_argument("--cases", nargs="*", help="subset of case names")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--rejudge", type=Path, metavar="RESULTS_DIR", help="re-score the judge on saved transcripts, no agent runs")
    args = ap.parse_args(argv)

    judge_model = None if args.no_judge else args.judge
    if args.rejudge:
        summary = rejudge(args.rejudge, args.judge, args.workers)
        print(summary_markdown(summary))
        return 0
    if judge_model == args.model:
        print(f"warning: judge and agent are the same model ({args.model}); self-preference bias", file=sys.stderr)

    cases = load_cases(names=args.cases)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out / f"{stamp}_{re.sub(r'[^A-Za-z0-9.-]+', '-', args.model)}"
    (out_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    jobs = [(c, i) for c in cases for i in range(args.runs)]
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="evalrun_") as tmp, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(score_one, c, i, model=args.model, judge_model=judge_model, tmp_parent=Path(tmp)): (c, i)
            for c, i in jobs
        }
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            name = f"{rec['case']}_{rec['run_index']}"
            if "failure" in rec:
                with (out_dir / "errors.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
                print(f"  {name}: FAILED ({rec['failure']})")
            else:
                (out_dir / "transcripts" / f"{name}.json").write_text(
                    json.dumps(rec, indent=1, default=str), encoding="utf-8"
                )
                marks = " ".join(
                    f"{k}={'ok' if v['passed'] else 'FAIL'}" for k, v in rec["checks"].items()
                )
                print(f"  {name}: {marks}")

    summary = summarise(records, model=args.model, judge_model=judge_model, runs=args.runs)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    md = summary_markdown(summary)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"results in {out_dir}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
