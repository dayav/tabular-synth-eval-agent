"""The comparison script's scoring and table rendering, on the golden case files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import compare_generators as cg  # noqa: E402
import numpy as np  # noqa: E402

from evals.generate_cases import draw_population  # noqa: E402


def test_evaluate_pair_on_golden_files(tmp_path: Path) -> None:
    real = str(ROOT / "evals/cases/identical/real.csv")
    test = tmp_path / "test.csv"
    draw_population(np.random.default_rng(99), n=1000).to_csv(test, index=False)  # a fresh held-out draw
    m = cg.evaluate_pair(real, str(ROOT / "evals/cases/clean_synthetic/synth.csv"), str(test), "target")
    assert m["n_categorical"] == 3 and m["n_numerical"] == 3  # target excluded from the d summary
    assert m["js_flagged"] == 0 and m["d_flagged"] == 0 and abs(m["utility_gap"]) < 0.1
    assert 0.3 < m["identifiability"] < 0.7 and m["exact_match_fraction"] == 0.0
    copy = cg.evaluate_pair(real, real, str(test), "target")
    assert copy["identifiability"] == 1.0 and copy["exact_match_fraction"] == 1.0


def test_markdown_table_and_readme_splice(tmp_path: Path) -> None:
    results = [{"dataset": "toy", "target": "t", "rows": 10, "test_rows": 5, "train_rows_available": 10, "columns": ["a"], "generators": {
        "ctgan": {"rows": 10, "meta": {}, "metrics": {"js_mean": 0.1, "js_max": 0.2, "js_flagged": 1, "d_mean": 0.05, "d_max": 0.1, "d_flagged": 0,
                                                      "tstr": 0.7, "trtr": 0.75, "utility_gap": 0.05, "identifiability": 0.5, "epsilon_any": 0.6,
                                                      "exact_match_fraction": 0.0, "n_categorical": 2, "n_numerical": 3},
                  "agent": {"verdict": {"overall": "acceptable"}}},
    }}]
    table = cg.markdown_table(results, "claude-x", "claude-y", "2026-10-21")
    assert "| toy | CTGAN | 0.100 / 0.200 (1/2) |" in table and "acceptable" in table and "claude-y" in table
    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- GENERATOR TABLE START -->\nOLDTABLE\n<!-- GENERATOR TABLE END -->\nafter\n")
    cg.splice_readme(table, readme)
    text = readme.read_text()
    assert "OLDTABLE" not in text and "| toy | CTGAN" in text and text.startswith("before") and text.endswith("after\n")
