"""Offline tests for the LLM generation baseline, with a scripted client."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate import generate_with_llm, json_schema_for, summarise_schema, validate_row  # noqa: E402


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "city": ["paris", "london", "paris", "berlin"] * 10,
        "age": list(range(20, 60)),
        "score": [x / 3 for x in range(40)],
        "flag": [0, 1] * 20,
        "target": [0, 0, 1, 1] * 10,
    })


def test_schema_summary_kinds_and_stats() -> None:
    specs = {s["name"]: s for s in summarise_schema(_frame(), target_column="target")}
    assert specs["city"]["kind"] == "categorical" and set(specs["city"]["levels"]) == {"paris", "london", "berlin"}
    assert specs["city"]["frequencies"]["paris"] == pytest.approx(0.5)
    assert specs["age"]["kind"] == "integer" and specs["age"]["min"] == 20 and specs["age"]["max"] == 59
    assert specs["score"]["kind"] == "float" and "q50" in specs["score"]["quantiles"]
    assert specs["flag"]["kind"] == "categorical" and specs["flag"]["integer_levels"] == [0, 1]  # low-cardinality int
    assert specs["target"]["integer_levels"] == [0, 1]
    assert "city" not in {s["name"] for s in summarise_schema(_frame(), exclude=["city"])}


def test_json_schema_is_strict() -> None:
    specs = summarise_schema(_frame(), target_column="target")
    schema = json_schema_for(specs)
    row = schema["properties"]["rows"]["items"]
    assert row["additionalProperties"] is False and set(row["required"]) == {"city", "age", "score", "flag", "target"}
    assert row["properties"]["city"]["enum"] and row["properties"]["flag"]["enum"] == [0, 1]
    assert "minItems" not in schema["properties"]["rows"]  # the API rejects array-size constraints


def test_validate_row_reasons() -> None:
    specs = summarise_schema(_frame(), target_column="target")
    ok = {"city": "paris", "age": 30, "score": 5.5, "flag": 1, "target": 0}
    assert validate_row(ok, specs) is None
    assert validate_row({**ok, "city": "rome"}, specs) == "bad_level:city"
    assert validate_row({**ok, "age": 30.5}, specs) == "not_integer:age"
    assert validate_row({**ok, "age": 500}, specs) == "out_of_range:age"
    assert validate_row({**ok, "flag": 2}, specs) == "bad_level:flag"
    assert validate_row({**ok, "score": "high"}, specs) == "not_numeric:score"
    assert validate_row({k: v for k, v in ok.items() if k != "target"}, specs) == "missing:target"


class ScriptedClient:
    """Returns batches: the first has one bad row, the rest are clean."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.calls = 0
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        rows = [{"city": "paris", "age": 25 + i, "score": 1.0 * i, "flag": i % 2, "target": i % 2} for i in range(self.batch_size)]
        if self.calls == 1:
            rows[0]["city"] = "rome"  # rejected
        text = json.dumps({"rows": rows})
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], model="claude-fake-1",
                               usage=SimpleNamespace(input_tokens=100, output_tokens=200))


def test_generate_with_llm_batches_validates_and_writes(tmp_path: Path) -> None:
    real = tmp_path / "real.csv"
    _frame().to_csv(real, index=False)
    out = tmp_path / "synth.csv"
    client = ScriptedClient(batch_size=5)
    result = generate_with_llm(str(real), str(out), n_rows=12, target_column="target", batch_size=5, client=client, model="claude-fake-1")
    assert result["rows_written"] == 12 and result["rows_requested"] == 12
    assert result["rows_rejected"] == 1 and result["rejection_reasons"] == {"bad_level:city": 1}
    assert result["batches"] == 3 and client.calls == 3  # 5+5+5 minus one bad row = 14 >= 12
    assert result["input_tokens"] == 300 and result["output_tokens"] == 600
    assert result["estimated_cost_usd"] is None  # fake model is unpriced
    frame = pd.read_csv(out)
    assert list(frame.columns) == ["city", "age", "score", "flag", "target"] and len(frame) == 12
    assert set(frame["city"]) == {"paris"}


def test_generate_gives_up_after_attempt_budget(tmp_path: Path) -> None:
    real = tmp_path / "real.csv"
    _frame().to_csv(real, index=False)

    class AllBad(ScriptedClient):
        def create(self, **kwargs: Any) -> Any:
            r = super().create(**kwargs)
            text = json.dumps({"rows": [{"city": "rome", "age": 1, "score": 0.0, "flag": 0, "target": 0}] * self.batch_size})
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], model="claude-fake-1", usage=r.usage)

    client = AllBad(batch_size=4)
    result = generate_with_llm(str(real), str(tmp_path / "s.csv"), n_rows=8, batch_size=4, max_attempts_factor=2, client=client, model="claude-fake-1")
    assert result["rows_written"] == 0 and result["batches"] == 4  # 2 batches minimum x factor 2
