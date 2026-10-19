"""Reconstruct the two recorded-failure fixtures (see README.md in this folder)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

from evals.cases import load_case  # noqa: E402
from regression.helpers import oracle_report  # noqa: E402
from tools import cohen_d_numerical, js_divergence_categorical, tstr_xgboost  # noqa: E402


def outputs_for(case):  # type: ignore[no-untyped-def]
    r, s = str(case.real_path), str(case.synth_path)
    return {
        "js": js_divergence_categorical(r, s),
        "d": cohen_d_numerical(r, s),
        "tstr": tstr_xgboost(r, s, case.target_column),
    }


def transcript(case, outputs, prompt, report, provenance):  # type: ignore[no-untyped-def]
    paths = {"real_csv_path": prompt_paths[0], "synth_csv_path": prompt_paths[1]}
    return {
        "case": case.name,
        "provenance": provenance,
        "prompt": prompt,
        "tool_calls": [
            {"name": "js_divergence_categorical", "input": dict(paths), "result": outputs["js"], "is_error": False},
            {"name": "cohen_d_numerical", "input": dict(paths), "result": outputs["d"], "is_error": False},
            {"name": "tstr_xgboost", "input": {**paths, "target_column": case.target_column}, "result": outputs["tstr"], "is_error": False},
        ],
        "report": report,
    }


# 1. The path leak: the case was prompted under its real path and the model read the
#    directory name. "10" appears in no tool result.
case = load_case(ROOT / "evals/cases/age_x10")
out = outputs_for(case)
prompt_paths = (str(case.real_path), str(case.synth_path))
leak_sentence = (
    f"The age column has been scaled by a factor of 10 (the synthetic file lives under "
    f"`age_x10`), which explains the very large effect size of {out['d']['per_column']['age']}."
)
report = oracle_report(case, out).replace(
    "## Utility", leak_sentence + "\n\n## Utility", 1
)
(HERE / "age_x10_path_leak.json").write_text(
    json.dumps(
        transcript(
            case, out, case.prompt(), report,
            "reconstructed on 2026-10-18 from the smoke-run failure recorded on 2026-10-15 "
            "(claude-sonnet-4-6 cited the directory name age_x10 as evidence of a x10 scaling)",
        ),
        indent=1,
    ),
    encoding="utf-8",
)

# 2. The recomputed mean: the model rounded the per-column values to three decimals and
#    averaged those, producing a figure one unit in the last place away from the tool's mean.
case = load_case(ROOT / "evals/cases/clean_synthetic")
out = outputs_for(case)
prompt_paths = (str(case.real_path), str(case.synth_path))
per_col = [v for v in out["d"]["per_column"].values() if v is not None]
recomputed = round(sum(round(v, 3) for v in per_col) / len(per_col), 4)
true_mean = out["d"]["mean"]
assert abs(recomputed - round(true_mean, 4)) > 1e-9, (recomputed, true_mean)
mean_sentence = f"Mean Cohen's d across the numerical columns: {recomputed}."
report = oracle_report(case, out).replace("\n## Utility", f"{mean_sentence}\n\n## Utility", 1)
(HERE / "clean_synthetic_recomputed_mean.json").write_text(
    json.dumps(
        transcript(
            case, out, case.prompt(), report,
            "reconstructed on 2026-10-18 from the full-run failure recorded on 2026-10-15 "
            f"(claude-sonnet-4-6 wrote a mean of its own rounding; the tool's mean is {true_mean})",
        ),
        indent=1,
    ),
    encoding="utf-8",
)
print("fixtures written:", recomputed, "vs tool mean", true_mean)
