"""Generate the golden cases: one real dataset and six synthetic files whose
correct verdict is known by construction.

Run ``make cases`` (or ``python -m evals.generate_cases``). Everything is seeded,
so the output is byte-for-byte reproducible.

The real dataset is drawn from a known process:

* ``gender``, ``education``, ``city`` from fixed proportions
  (education: 50% bachelor, 35% master, 15% phd)
* ``age``, ``income``, ``score`` from normal distributions, independent of each other
* ``target = 1`` when ``1.2 * z(income) + 0.6 * z(score) + noise > 0``

Income is independent of the other columns on purpose: if ``age`` or
``education`` could stand in for it, shuffling it would not remove any real
signal and the shuffle case would prove nothing.

The ``expected.json`` written next to each case is derived from how the file was
built, never from what the tools returned. ``regression/test_expected_matches_tool_output.py``
checks that the two agree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CASES_DIR = Path(__file__).resolve().parent / "cases"
N_ROWS = 2000
SEED = 42
TARGET = "target"
TOOLS = ["js_divergence_categorical", "cohen_d_numerical", "tstr_xgboost", "lookup_metric_guidance"]
TOOLS_WITH_PRIVACY = ["js_divergence_categorical", "cohen_d_numerical", "tstr_xgboost", "identifiability_heom", "lookup_metric_guidance"]


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def draw_population(rng: np.random.Generator, n: int = N_ROWS) -> pd.DataFrame:
    """One draw of ``n`` rows from the known generative process."""
    gender = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])
    education = rng.choice(["bachelor", "master", "phd"], size=n, p=[0.50, 0.35, 0.15])
    city = rng.choice(["paris", "london", "berlin", "madrid"], size=n, p=[0.35, 0.30, 0.20, 0.15])
    age = np.clip(np.round(rng.normal(38, 10, size=n)), 18, 70).astype(int)
    income = np.clip(np.round(rng.normal(55_000, 15_000, size=n), -2), 15_000, None).astype(int)
    score = np.round(rng.normal(70, 10, size=n), 1)
    logit = 1.2 * _z(income.astype(float)) + 0.6 * _z(score) + rng.normal(0, 1, size=n)
    target = (logit > 0).astype(int)
    return pd.DataFrame(
        {
            "gender": gender,
            "education": education,
            "city": city,
            "age": age,
            "income": income,
            "score": score,
            TARGET: target,
        }
    )


def build_cases(rng: np.random.Generator) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    """Return ``{case_name: (synthetic_frame, expected_json)}`` for one real draw."""
    clean = draw_population(rng)

    def expected(
        description: str,
        construction: str,
        verdict: dict[str, str],
        flagged: list[str],
        notes: str,
    ) -> dict[str, Any]:
        return {
            "description": description,
            "construction": construction,
            "ground_truth_source": (
                "known by construction of the synthetic file (see evals/generate_cases.py)"
            ),
            "target_column": TARGET,
            "tools": TOOLS,
            "verdict": verdict,
            "flagged_columns": flagged,
            "notes": notes,
        }

    good_all = {"categorical": "good", "numerical": "good", "utility": "good", "overall": "excellent"}

    cases: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}

    cases["clean_synthetic"] = (
        clean,
        expected(
            "A fresh draw from the same process as the real file: the control.",
            "synth = draw_population(rng)  # same process, different random draw",
            good_all,
            [],
            "What 'no corruption' scores look like. Every JS distance and Cohen's d is "
            "sampling noise, the utility gap is within a few hundredths of zero.",
        ),
    )

    shuffled = clean.copy()
    shuffled["income"] = rng.permutation(shuffled["income"].to_numpy())
    cases["income_shuffled"] = (
        shuffled,
        expected(
            "The clean synthetic draw with the income column permuted within itself.",
            "synth = clean; synth.income = permutation(synth.income)",
            {"categorical": "good", "numerical": "good", "utility": "poor", "overall": "poor"},
            [],
            "A permutation keeps the marginal distribution, so Cohen's d on income is the "
            "same as on the clean control. It destroys the joint distribution with the "
            "label, so TSTR drops. Nothing in the tool output points at income: the report "
            "must NOT name it as the cause.",
        ),
    )

    leaky = clean.copy()
    leaky["score"] = np.where(leaky[TARGET] == 1, 80.0, 60.0)
    cases["score_leaks_target"] = (
        leaky,
        expected(
            "The clean synthetic draw with score replaced by 70 +/- 10 according to the label.",
            "synth = clean; synth.score = 80 where target == 1 else 60",
            {"categorical": "good", "numerical": "good", "utility": "poor", "overall": "poor"},
            [],
            "Same mean and spread as the real score column, so Cohen's d stays under 0.2, "
            "but the two classes are perfectly separable. A model trained on it learns a "
            "rule that is only weakly true in real data and loses accuracy. The report "
            "must NOT name score as the cause: no tool result points at it.",
        ),
    )

    scaled = clean.copy()
    scaled["age"] = scaled["age"] * 10
    cases["age_x10"] = (
        scaled,
        expected(
            "The clean synthetic draw with age multiplied by 10.",
            "synth = clean; synth.age = synth.age * 10",
            {"categorical": "good", "numerical": "poor", "utility": "good", "overall": "acceptable"},
            ["age"],
            "Cohen's d on age is huge. Trees split on absolute thresholds and age carries "
            "no signal about the label, so utility barely moves: fidelity poor, utility "
            "good, overall acceptable.",
        ),
    )

    missing = clean.copy()
    missing.loc[missing["education"] == "phd", "education"] = "master"
    cases["phd_missing"] = (
        missing,
        expected(
            "The clean synthetic draw with every phd relabelled master.",
            "synth = clean; synth.education[education == 'phd'] = 'master'",
            {"categorical": "poor", "numerical": "good", "utility": "good", "overall": "acceptable"},
            ["education"],
            "One categorical level disappears, so the JS distance on education is well "
            "above 0.1. Education carries no signal about the label, so utility is fine.",
        ),
    )

    return cases


def main(out_dir: Path = CASES_DIR) -> None:
    rng = np.random.default_rng(SEED)
    real = draw_population(rng)

    identical_expected = {
        "description": "The synthetic file is an exact copy of the real file.",
        "construction": "synth = real",
        "ground_truth_source": (
            "known by construction of the synthetic file (see evals/generate_cases.py)"
        ),
        "target_column": TARGET,
        "tools": TOOLS,
        "verdict": {"categorical": "good", "numerical": "good", "utility": "good", "overall": "excellent"},
        "flagged_columns": [],
        "notes": (
            "Every JS distance and Cohen's d is exactly 0. The utility gap is strongly "
            "negative because the synthetic-trained model has seen the held-out real rows; "
            "a gap at or below zero is good utility, and the report must not call it a problem."
        ),
    }
    all_cases = {"identical": (real.copy(), identical_expected), **build_cases(rng)}

    # Privacy cases: the same files as two existing cases, with the prompt asking for privacy.
    # The verdicts gain a privacy axis; overall follows the rule "poor if utility or privacy is poor".
    all_cases["identical_privacy"] = (
        real.copy(),
        {
            **identical_expected,
            "description": "Exact copy of the real file, with the prompt asking about privacy.",
            "ask_privacy": True,
            "tools": TOOLS_WITH_PRIVACY,
            "verdict": {"categorical": "good", "numerical": "good", "utility": "good", "privacy": "poor", "overall": "poor"},
            "notes": (
                "Every synthetic row is a verbatim real row: identifiability 1.0, exact_match_fraction 1.0. "
                "Maximum disclosure risk. Fidelity and utility look perfect, which is exactly why a privacy "
                "axis is needed: the report must call this poor overall."
            ),
        },
    )
    clean_frame, clean_expected = all_cases["clean_synthetic"]
    all_cases["clean_privacy"] = (
        clean_frame,
        {
            **clean_expected,
            "description": "A fresh draw from the same process, with the prompt asking about privacy: the privacy control.",
            "ask_privacy": True,
            "tools": TOOLS_WITH_PRIVACY,
            "verdict": {"categorical": "good", "numerical": "good", "utility": "good", "privacy": "good", "overall": "excellent"},
            "notes": (
                "An honest sample: identifiability near 0.5 (under the 0.7 threshold), no exact matches. "
                "epsilon_any is around 0.65 and must not be read as a problem: its honest-sample level is high."
            ),
        },
    )

    for name, (synth, exp) in all_cases.items():
        case_dir = out_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        real.to_csv(case_dir / "real.csv", index=False)
        synth.to_csv(case_dir / "synth.csv", index=False)
        (case_dir / "expected.json").write_text(json.dumps(exp, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {case_dir}")


if __name__ == "__main__":
    main()
