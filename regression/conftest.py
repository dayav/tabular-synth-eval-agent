"""Shared fixtures: load the golden cases once and run the tools on them once.

The regression suite never calls the model. Tool outputs are deterministic
(seeded split), so computing them once per session is cheap and exact.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evals.cases import Case, load_cases  # noqa: E402
from tools import cohen_d_numerical, identifiability_heom, js_divergence_categorical, tstr_xgboost  # noqa: E402


@pytest.fixture(scope="session")
def cases() -> list[Case]:
    return load_cases()


@pytest.fixture(scope="session")
def tool_outputs(cases: list[Case]) -> dict[str, dict[str, Any]]:
    """``{case_name: {"js": ..., "d": ..., "tstr": ...}}`` from the real tools."""
    out: dict[str, dict[str, Any]] = {}
    for c in cases:
        r, s = str(c.real_path), str(c.synth_path)
        out[c.name] = {
            "js": js_divergence_categorical(r, s),
            "d": cohen_d_numerical(r, s),
            "tstr": tstr_xgboost(r, s, c.target_column),
        }
        if "identifiability_heom" in c.expected_tools:
            out[c.name]["privacy"] = identifiability_heom(r, s)
    return out
