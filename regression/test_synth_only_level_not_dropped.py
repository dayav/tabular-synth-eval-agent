"""A category level present only in the synthetic file is a real difference.

Provenance: observed tool bug. ``js_divergence_categorical`` reindexed on the
real file's levels only; the synthetic-only level was dropped and, because
``scipy.spatial.distance.jensenshannon`` renormalises its inputs, the distance
came out as exactly 0.0 for a visibly different column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools import js_divergence_categorical


def test_synth_only_level_is_counted(tmp_path: Path) -> None:
    real = tmp_path / "real.csv"
    synth = tmp_path / "synth.csv"
    pd.DataFrame({"colour": ["a", "a", "b", "b"], "y": [0, 1, 0, 1]}).to_csv(real, index=False)
    pd.DataFrame({"colour": ["a", "b", "c", "c"], "y": [0, 1, 0, 1]}).to_csv(synth, index=False)
    result = js_divergence_categorical(str(real), str(synth))
    js = result["per_column"]["colour"]
    assert js is not None and js > 0.3, result


def test_identical_levels_still_zero(tmp_path: Path) -> None:
    real = tmp_path / "real.csv"
    synth = tmp_path / "synth.csv"
    frame = pd.DataFrame({"colour": ["a", "a", "b", "c"], "y": [0, 1, 0, 1]})
    frame.to_csv(real, index=False)
    frame.sample(frac=1, random_state=1).to_csv(synth, index=False)
    result = js_divergence_categorical(str(real), str(synth))
    assert result["per_column"]["colour"] == 0.0
