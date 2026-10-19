"""A utility gap at or below zero is good utility, and the identical case is excellent.

Provenance: observed while building the cases. With synth == real, the
synthetic-trained model has seen the 20% of real rows the tool holds out, so it
scores higher than the real-trained baseline and the gap is strongly negative.
An earlier draft of the rules treated only a gap *near* zero as good.
"""

from __future__ import annotations

from typing import Any

from evals.cases import Case
from evals.checks.verdict import derive_expected


def test_negative_gap_is_good_utility() -> None:
    js = {"per_column": {"g": 0.0}, "mean": 0.0, "n_columns": 1}
    d = {"per_column": {"a": 0.0, "target": 0.0}, "mean": 0.0, "n_columns": 2}
    tstr = {"tstr_accuracy": 0.99, "trtr_accuracy": 0.78, "utility_gap": -0.21,
            "target_column": "target", "n_test_samples": 400, "n_classes": 2}
    derived = derive_expected(js, d, tstr, "target")
    assert derived["verdict"] == {
        "categorical": "good", "numerical": "good", "utility": "good", "overall": "excellent"
    }
    assert derived["flagged_columns"] == []


def test_identical_case_has_negative_gap_and_is_excellent(
    cases: list[Case], tool_outputs: dict[str, dict[str, Any]]
) -> None:
    case = next(c for c in cases if c.name == "identical")
    out = tool_outputs["identical"]
    assert out["tstr"]["utility_gap"] < 0
    assert all(v == 0.0 for v in out["js"]["per_column"].values())
    assert all(v == 0.0 for v in out["d"]["per_column"].values())
    assert case.expected["verdict"]["overall"] == "excellent"
    assert case.expected["verdict"]["utility"] == "good"
