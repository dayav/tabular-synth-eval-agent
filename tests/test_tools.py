"""Known-answer unit tests for the three deterministic tools in src/tools.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tools import cohen_d_numerical, identifiability_heom, js_divergence_categorical, tstr_xgboost  # noqa: E402


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


# --------------------------------------------------------------------------- privacy: identifiability_heom
def _mixed(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "colour": rng.choice(["red", "green", "blue"], n),
        "x": rng.normal(0, 1, n),
        "y": rng.uniform(0, 100, n),
    })


def test_identifiability_exact_copy_is_maximal(tmp_path: Path) -> None:
    real = _write(tmp_path, "r.csv", _mixed(300, 0))
    result = identifiability_heom(real, real)
    assert result["identifiability"] == 1.0
    assert result["epsilon_any"] == 1.0
    assert result["exact_match_fraction"] == 1.0 and result["exact_matches"] == 300
    assert result["median_synth_to_real_distance"] == 0.0
    assert result["numerical_columns"] == ["x", "y"] and result["categorical_columns"] == ["colour"]
    assert result["k"] == 2 and result["n_real"] == 300 and result["n_synthetic"] == 300


def test_identifiability_honest_sample_sits_near_one_half(tmp_path: Path) -> None:
    real = _write(tmp_path, "r.csv", _mixed(1500, 1))
    synth = _write(tmp_path, "s.csv", _mixed(1500, 2))
    result = identifiability_heom(real, synth)
    assert 0.4 < result["identifiability"] < 0.6
    assert result["exact_matches"] == 0


def test_identifiability_far_away_sample_is_zero(tmp_path: Path) -> None:
    frame = _mixed(200, 3)
    far = frame.copy()
    far["x"] += 50  # fifty standard deviations away on one axis
    real = _write(tmp_path, "r.csv", frame)
    synth = _write(tmp_path, "s.csv", far)
    result = identifiability_heom(real, synth)
    assert result["identifiability"] == 0.0 and result["epsilon_any"] == 0.0


def test_heom_distance_units(tmp_path: Path) -> None:
    """A category mismatch counts 1; a numeric difference counts its fraction of the real range."""
    real = _write(tmp_path, "r.csv", pd.DataFrame({"c": ["a", "b", "a", "b"], "v": [0.0, 10.0, 0.0, 10.0]}))
    # synthetic rows: same category, numeric off by half the range -> distance 0.5 ; different category -> 1
    synth = _write(tmp_path, "s.csv", pd.DataFrame({"c": ["a", "b"], "v": [5.0, 10.0]}))
    result = identifiability_heom(real, synth)
    # real radii are 0 (duplicates), so nothing is "inside" a ball; distances are reported via the median
    assert result["epsilon_any"] == 0.0
    assert result["median_synth_to_real_distance"] == pytest.approx(0.25)  # median of [0.5, 0.0]


def test_identifiability_k_and_excluded_columns_and_missing(tmp_path: Path) -> None:
    frame = _mixed(200, 4)
    frame.loc[0, "x"] = np.nan
    frame["id"] = range(200)
    real = _write(tmp_path, "r.csv", frame)
    synth = _write(tmp_path, "s.csv", frame.assign(id=range(1000, 1200)))
    with_id = identifiability_heom(real, synth)
    without_id = identifiability_heom(real, synth, exclude_columns=["id"])
    assert "id" not in without_id["numerical_columns"] and "id" in with_id["numerical_columns"]
    assert without_id["exact_match_fraction"] == 1.0  # identical once the shifted id is ignored
    assert without_id["n_rows_dropped_for_missing_values"] == 2  # the NaN row, in both files
    k3 = identifiability_heom(real, synth, k=3, exclude_columns=["id"])
    assert k3["k"] == 3 and k3["median_real_knn_radius"] >= without_id["median_real_knn_radius"]
    with pytest.raises(ValueError):
        identifiability_heom(real, synth, k=1)


def test_identifiability_size_matched_subsample(tmp_path: Path) -> None:
    real = _write(tmp_path, "r.csv", _mixed(300, 5))
    synth = _write(tmp_path, "s.csv", _mixed(900, 6))
    result = identifiability_heom(real, synth)
    assert result["n_synthetic_used_for_identifiability"] == 300 and result["n_synthetic"] == 900
