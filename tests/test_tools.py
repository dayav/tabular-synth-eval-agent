"""Known-answer unit tests for the three deterministic tools in src/tools.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tools import cohen_d_numerical, js_divergence_categorical, tstr_xgboost  # noqa: E402


def _write(tmp_path: Path, name: str, frame: pd.DataFrame) -> str:
    p = tmp_path / name
    frame.to_csv(p, index=False)
    return str(p)


def test_js_identical_is_zero_and_disjoint_is_one(tmp_path: Path) -> None:
    real = _write(tmp_path, "r.csv", pd.DataFrame({"c": ["a", "a", "b"], "n": [1, 2, 3]}))
    same = _write(tmp_path, "s.csv", pd.DataFrame({"c": ["b", "a", "a"], "n": [3, 2, 1]}))
    disjoint = _write(tmp_path, "d.csv", pd.DataFrame({"c": ["x", "y", "y"], "n": [1, 2, 3]}))
    assert js_divergence_categorical(real, same)["per_column"]["c"] == 0.0
    assert js_divergence_categorical(real, disjoint)["per_column"]["c"] == pytest.approx(1.0)


def test_js_known_value_and_summary(tmp_path: Path) -> None:
    # real: p = (0.5, 0.5); synth: q = (1, 0). JS divergence (base 2) = 1 - 0.5*log2(... ) -> distance sqrt(0.3113)
    real = _write(tmp_path, "r.csv", pd.DataFrame({"c": ["a", "b"], "d": ["u", "v"]}))
    synth = _write(tmp_path, "s.csv", pd.DataFrame({"c": ["a", "a"], "d": ["u", "v"]}))
    result = js_divergence_categorical(real, synth)
    m = np.array([0.75, 0.25])
    expected_div = 0.5 * sum(p * math.log2(p / mi) for p, mi in zip([0.5, 0.5], m, strict=True)) + 0.5 * (1.0 * math.log2(1.0 / m[0]))
    assert result["per_column"]["c"] == pytest.approx(math.sqrt(expected_div))
    assert result["per_column"]["d"] == 0.0
    assert result["n_columns"] == 2
    assert result["mean"] == pytest.approx((result["per_column"]["c"] + 0.0) / 2)


def test_cohen_d_known_value(tmp_path: Path) -> None:
    # means 3 and 5, sample variances 2.5 each -> pooled std sqrt(2.5), d = 2/sqrt(2.5)
    real = _write(tmp_path, "r.csv", pd.DataFrame({"x": [1, 2, 3, 4, 5]}))
    synth = _write(tmp_path, "s.csv", pd.DataFrame({"x": [3, 4, 5, 6, 7]}))
    result = cohen_d_numerical(real, synth)
    assert result["per_column"]["x"] == pytest.approx(2 / math.sqrt(2.5))
    assert result["n_columns"] == 1


def test_cohen_d_is_absolute_and_zero_for_identical(tmp_path: Path) -> None:
    real = _write(tmp_path, "r.csv", pd.DataFrame({"x": [3, 4, 5, 6, 7]}))
    synth = _write(tmp_path, "s.csv", pd.DataFrame({"x": [1, 2, 3, 4, 5]}))
    assert cohen_d_numerical(real, synth)["per_column"]["x"] == pytest.approx(2 / math.sqrt(2.5))
    assert cohen_d_numerical(real, real)["per_column"]["x"] == 0.0


def test_cohen_d_constant_column_is_undefined(tmp_path: Path) -> None:
    real = _write(tmp_path, "r.csv", pd.DataFrame({"x": [2, 2, 2], "y": [1, 2, 3]}))
    synth = _write(tmp_path, "s.csv", pd.DataFrame({"x": [2, 2, 2], "y": [1, 2, 3]}))
    result = cohen_d_numerical(real, synth)
    assert result["per_column"]["x"] is None
    assert result["mean"] == 0.0  # mean over the defined scores only


def _separable(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "noise": rng.normal(0, 1, n), "y": (x > 0).astype(int)})


def test_tstr_identical_synthetic_scores_at_least_the_baseline(tmp_path: Path) -> None:
    real = _write(tmp_path, "r.csv", _separable(400, 0))
    result = tstr_xgboost(real, real, "y")
    assert result["trtr_accuracy"] > 0.9
    assert result["tstr_accuracy"] >= result["trtr_accuracy"]
    assert result["utility_gap"] == pytest.approx(result["trtr_accuracy"] - result["tstr_accuracy"])
    assert result["n_test_samples"] == 80  # 20% of 400
    assert result["n_classes"] == 2
    assert result["target_column"] == "y"


def test_tstr_useless_synthetic_loses_utility(tmp_path: Path) -> None:
    frame = _separable(400, 1)
    useless = frame.copy()
    useless["y"] = np.random.default_rng(2).permutation(useless["y"].to_numpy())
    real = _write(tmp_path, "r.csv", frame)
    synth = _write(tmp_path, "s.csv", useless)
    result = tstr_xgboost(real, synth, "y")
    assert result["utility_gap"] > 0.2


def test_tstr_is_deterministic_and_accepts_string_labels(tmp_path: Path) -> None:
    frame = _separable(300, 3)
    frame["y"] = np.where(frame["y"] == 1, "yes", "no")
    real = _write(tmp_path, "r.csv", frame)
    a = tstr_xgboost(real, real, "y")
    b = tstr_xgboost(real, real, "y")
    assert a == b


def test_tstr_explicit_test_set(tmp_path: Path) -> None:
    real = _write(tmp_path, "r.csv", _separable(300, 4))
    test = _write(tmp_path, "t.csv", _separable(50, 5))
    result = tstr_xgboost(real, real, "y", test_csv_path=test)
    assert result["n_test_samples"] == 50
