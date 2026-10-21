#!/usr/bin/env python
"""Compare CTGAN, TVAE and LLM generation on the same metrics, with the same agent.

    python scripts/compare_generators.py [--datasets adult cardio credit] [--rows 500]
        [--generator-model claude-sonnet-5-5] [--agent-model claude-sonnet-5-5 | --no-agent]
        [--regenerate]

For every dataset: a reference sample of ``--rows`` real training rows and a held-out
sample of real test rows are drawn once (seeded). Each generator's output is cut to
``--rows`` rows (CTGAN and TVAE from ``data/generators/<ds>/``; the LLM output is generated
from the schema and marginals of the *reference sample* and cached under
``results/generators/llm_<ds>_<model>.csv``). The four tools then score every generator
against the same reference and test files, and the evaluation agent, if enabled, writes its
verdict for each. Results go to ``results/generators/comparison.{json,md}`` and the Markdown
table is spliced into the README between the generator markers.

Fairness notes the table carries: CTGAN and TVAE were trained on the full training set
(tens of thousands of rows); the LLM saw only the schema and marginal statistics of the
reference sample. All three are evaluated on the same ``--rows``-row footing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

from generate import generate_with_llm  # noqa: E402
from tools import cohen_d_numerical, identifiability_heom, js_divergence_categorical, tstr_xgboost  # noqa: E402

DATA_DIR = ROOT / "data" / "generators"
OUT_DIR = ROOT / "results" / "generators"
TARGETS = {"adult": "income", "cardio": "cardio", "credit": "default.payment.next.month"}
EXCLUDE = {"adult": [], "cardio": [], "credit": ["ID"]}
GENERATORS = ("ctgan", "tvae", "llm")
README_START, README_END = "<!-- GENERATOR TABLE START -->", "<!-- GENERATOR TABLE END -->"


def _write(frame: pd.DataFrame, path: Path) -> str:
    frame.to_csv(path, index=False)
    return str(path)


def evaluate_pair(real_csv: str, synth_csv: str, test_csv: str, target: str) -> dict[str, Any]:
    js = js_divergence_categorical(real_csv, synth_csv)
    d = cohen_d_numerical(real_csv, synth_csv)
    ts = tstr_xgboost(real_csv, synth_csv, target, test_csv_path=test_csv)
    pr = identifiability_heom(real_csv, synth_csv)
    d_cols = {k: v for k, v in d["per_column"].items() if k != target and v is not None}
    js_cols = {k: v for k, v in js["per_column"].items() if v is not None}
    return {
        "js_mean": js["mean"], "js_max": max(js_cols.values()) if js_cols else None, "js_flagged": sum(v > 0.1 for v in js_cols.values()),
        "d_mean": (sum(d_cols.values()) / len(d_cols)) if d_cols else None, "d_max": max(d_cols.values()) if d_cols else None,
        "d_flagged": sum(v > 0.2 for v in d_cols.values()),
        "tstr": ts["tstr_accuracy"], "trtr": ts["trtr_accuracy"], "utility_gap": ts["utility_gap"],
        "identifiability": pr["identifiability"], "epsilon_any": pr["epsilon_any"], "exact_match_fraction": pr["exact_match_fraction"],
        "n_categorical": js["n_columns"], "n_numerical": len(d_cols),
    }


def run_agent_verdict(real_csv: str, synth_csv: str, target: str, model: str) -> dict[str, Any]:
    from agent import run_agent
    from api import parse_verdict

    prompt = (
        f"Evaluate the synthetic dataset at {synth_csv} against the real dataset at {real_csv}. "
        f"The target column for utility evaluation is '{target}'. "
        "Also assess the privacy of the synthetic data: the disclosure risk of real records."
    )
    run = run_agent(prompt, model=model, log_dir=None)
    verdict, flagged = parse_verdict(run.text)
    return {"verdict": verdict.model_dump() if verdict else None, "flagged_columns": flagged, "model_served": run.model_served,
            "tokens": {"input": run.usage.input_tokens, "output": run.usage.output_tokens}, "report": run.text}


def compare_dataset(ds: str, rows: int, generator_model: str, agent_model: str | None, regenerate: bool, seed: int, workdir: Path) -> dict[str, Any]:
    folder = DATA_DIR / ds
    target = TARGETS[ds]
    train = pd.read_csv(folder / "train.csv").drop(columns=EXCLUDE[ds], errors="ignore").dropna()
    test = pd.read_csv(folder / "test.csv").drop(columns=EXCLUDE[ds], errors="ignore").dropna()
    ref = train.sample(n=min(rows, len(train)), random_state=seed).reset_index(drop=True)
    held = test.sample(n=min(2000, len(test)), random_state=seed).reset_index(drop=True)
    real_csv = _write(ref, workdir / f"{ds}_real.csv")
    test_csv = _write(held, workdir / f"{ds}_test.csv")

    out: dict[str, Any] = {"dataset": ds, "target": target, "rows": len(ref), "test_rows": len(held),
                           "train_rows_available": len(train), "columns": list(ref.columns), "generators": {}}
    for gen in GENERATORS:
        if gen == "llm":
            cached = OUT_DIR / f"llm_{ds}_{generator_model}.csv"
            meta_path = cached.with_suffix(".json")
            if not cached.exists() or regenerate:
                print(f"  generating {rows} rows for {ds} with {generator_model} ...")
                meta = generate_with_llm(real_csv, str(cached), rows, target_column=target, model=generator_model)
                meta_path.write_text(json.dumps({k: v for k, v in meta.items() if k != "schema_summary"}, indent=1))
            gen_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            synth = pd.read_csv(cached)
        else:
            gen_meta = {"source": f"data/generators/{ds}/{gen}.csv (thesis pipeline baseline, trained on the full training set)"}
            full = pd.read_csv(folder / f"{gen}.csv").drop(columns=EXCLUDE[ds], errors="ignore").dropna()
            synth = full.sample(n=min(rows, len(full)), random_state=seed).reset_index(drop=True)
        synth = synth[ref.columns]
        synth_csv = _write(synth, workdir / f"{ds}_{gen}.csv")
        print(f"  scoring {ds}/{gen} ({len(synth)} rows)")
        entry: dict[str, Any] = {"rows": len(synth), "meta": gen_meta, "metrics": evaluate_pair(real_csv, synth_csv, test_csv, target)}
        if agent_model:
            print(f"  agent verdict for {ds}/{gen} with {agent_model} ...")
            entry["agent"] = run_agent_verdict(real_csv, synth_csv, target, agent_model)
        out["generators"][gen] = entry
    return out


def _f(v: Any, spec: str = ".3f") -> str:
    return "-" if v is None else format(v, spec)


def markdown_table(results: list[dict[str, Any]], generator_model: str, agent_model: str | None, when: str) -> str:
    lines = [
        f"Measured {when}: {len(results)} datasets, {results[0]['rows']} rows per generator, same real reference sample and the same held-out test rows for every generator. "
        f"CTGAN and TVAE outputs are the thesis pipeline baselines (trained on the full training sets); the LLM ({generator_model}) saw only the schema and marginal statistics of the reference sample."
        + (f" Agent verdicts by {agent_model}." if agent_model else ""),
        "",
        "| dataset | generator | JS mean / max (flagged) | Cohen's d mean / max (flagged) | TSTR / TRTR | utility gap | identifiability | exact matches | agent overall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        for gen in GENERATORS:
            e = r["generators"].get(gen)
            if not e:
                continue
            m = e["metrics"]
            v = (e.get("agent") or {}).get("verdict") or {}
            lines.append(
                f"| {r['dataset']} | {gen.upper() if gen != 'llm' else 'LLM'} | {_f(m['js_mean'])} / {_f(m['js_max'])} ({m['js_flagged']}/{m['n_categorical']}) "
                f"| {_f(m['d_mean'])} / {_f(m['d_max'])} ({m['d_flagged']}/{m['n_numerical']}) | {_f(m['tstr'])} / {_f(m['trtr'])} | {_f(m['utility_gap'], '+.3f')} "
                f"| {_f(m['identifiability'], '.2f')} | {_f(m['exact_match_fraction'], '.1%')} | {v.get('overall', '-')} |"
            )
    lines += [
        "",
        "Read with care: three public datasets, a few hundred rows per generator, one draw each; this is a baseline, not a claim about LLMs and tabular data. "
        "Thresholds: JS above 0.1 and Cohen's d above 0.2 per column are flagged; utility gap above 0.05 is poor; identifiability above 0.7 or exact matches above 5% is poor. "
        "The LLM cannot copy rows it never saw, so its identifiability is a measure of how closely marginals alone place points near real records.",
    ]
    return "\n".join(lines)


def splice_readme(table: str, readme: Path = ROOT / "README.md") -> None:
    text = readme.read_text(encoding="utf-8")
    if README_START not in text:
        raise SystemExit("README has no generator table markers")
    new = re.sub(re.escape(README_START) + r".*?" + re.escape(README_END), f"{README_START}\n{table}\n{README_END}", text, flags=re.DOTALL)
    readme.write_text(new, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(TARGETS))
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--generator-model", default="claude-sonnet-5-5")
    ap.add_argument("--agent-model", default="claude-sonnet-5-5")
    ap.add_argument("--no-agent", action="store_true")
    ap.add_argument("--regenerate", action="store_true", help="regenerate the cached LLM output")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args(argv)
    agent_model = None if args.no_agent else args.agent_model
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="compare_") as tmp:
        for ds in args.datasets:
            if not (DATA_DIR / ds / "train.csv").exists():
                raise SystemExit(f"missing data for {ds}: run scripts/fetch_generator_outputs.py first")
            print(f"{ds}:")
            results.append(compare_dataset(ds, args.rows, args.generator_model, agent_model, args.regenerate, args.seed, Path(tmp)))
    when = datetime.now(UTC).strftime("%Y-%m-%d")
    table = markdown_table(results, args.generator_model, agent_model, when)
    (OUT_DIR / "comparison.json").write_text(json.dumps({"measured": when, "rows": args.rows, "generator_model": args.generator_model,
                                                          "agent_model": agent_model, "results": results}, indent=1, default=str))
    (OUT_DIR / "comparison.md").write_text(table + "\n")
    print()
    print(table)
    if not args.no_readme:
        splice_readme(table)
        print("README updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
