"""Offline tests for run logging: a fake API client drives the real loop."""

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

import agent  # noqa: E402
from agent import AgentRun, ApiStep, ToolCall, Usage, run_agent  # noqa: E402
from pricing import PRICING_DATE, estimate_cost  # noqa: E402
from runlog import describe_csv, run_to_dict, write_run  # noqa: E402

sys.path.insert(0, str(ROOT / "runs"))
import summarize  # noqa: E402


# --------------------------------------------------------------------------- fake API
class _Usage(SimpleNamespace):
    pass


def _response(blocks: list[Any], stop_reason: str, in_tok: int, out_tok: int, cache_read: int = 0) -> Any:
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        model="claude-fake-1",
        usage=_Usage(input_tokens=in_tok, output_tokens=out_tok,
                     cache_read_input_tokens=cache_read, cache_creation_input_tokens=0),
        _request_id="req_test",
    )


def _tool_use(name: str, inp: dict[str, Any], id_: str) -> Any:
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=id_)


def _text(t: str) -> Any:
    return SimpleNamespace(type="text", text=t)


class FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("boom: fake API exhausted")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = FakeMessages(responses)


@pytest.fixture
def csvs(tmp_path: Path) -> tuple[Path, Path]:
    frame = pd.DataFrame({"c": ["a", "b", "a", "b"] * 5, "x": range(20), "target": [0, 1] * 10})
    real, synth = tmp_path / "real.csv", tmp_path / "synth.csv"
    frame.to_csv(real, index=False)
    frame.sample(frac=1, random_state=1).to_csv(synth, index=False)
    return real, synth


def _scripted_client(real: Path, synth: Path) -> FakeClient:
    paths = {"real_csv_path": str(real), "synth_csv_path": str(synth)}
    return FakeClient([
        _response([_tool_use("js_divergence_categorical", dict(paths), "t1"),
                   _tool_use("cohen_d_numerical", dict(paths), "t2")], "tool_use", 1000, 50),
        _response([_tool_use("tstr_xgboost", {**paths, "target_column": "target"}, "t3")], "tool_use", 1500, 40, cache_read=800),
        _response([_text("Overall verdict: excellent.\n\n```json\n"
                         '{"verdict": {"categorical": "good", "numerical": "good", "utility": "good", "overall": "excellent"}, "flagged_columns": []}'
                         "\n```")], "end_turn", 2000, 300),
    ])


# --------------------------------------------------------------------------- loop instrumentation
def test_run_agent_records_steps_tools_usage_and_writes_folder(
    csvs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real, synth = csvs
    client = _scripted_client(real, synth)
    monkeypatch.setattr(agent, "get_client", lambda: client)

    run = run_agent("evaluate", model="claude-fake-1", log_dir=tmp_path / "runs")

    assert run.stop_reason == "end_turn" and run.error is None
    assert [s.turn for s in run.steps] == [0, 1, 2]
    assert [s.n_tool_calls for s in run.steps] == [2, 1, 0]
    assert all(s.request_id == "req_test" and s.latency_s >= 0 for s in run.steps)
    assert run.usage.input_tokens == 4500 and run.usage.output_tokens == 390
    assert run.usage.cache_read_input_tokens == 800
    assert [tc.name for tc in run.tool_calls] == ["js_divergence_categorical", "cohen_d_numerical", "tstr_xgboost"]
    assert [tc.turn for tc in run.tool_calls] == [0, 0, 1]
    assert all(tc.latency_s >= 0 and not tc.is_error for tc in run.tool_calls)
    assert run.model_served == "claude-fake-1"
    assert run.latency_s >= 0 and run.started_at and run.finished_at

    folder = run.log_dir
    assert folder is not None and folder.parent == tmp_path / "runs" and folder.name == run.run_id
    assert {p.name for p in folder.iterdir()} == {"meta.json", "trace.json", "usage.json", "report.md"}

    meta = json.loads((folder / "meta.json").read_text())
    assert meta["status"] == "ok" and meta["turns"] == 3 and meta["model_served"] == "claude-fake-1"
    assert meta["inputs"]["target_column"] == "target"
    roles = {f["role"]: f for f in meta["inputs"]["files"]}
    assert roles["real"]["rows"] == 20 and roles["real"]["columns"] == 3
    assert roles["synthetic"]["sha256"] != roles["real"]["sha256"]
    assert meta["tools_called"] == ["js_divergence_categorical", "cohen_d_numerical", "tstr_xgboost"]

    trace = json.loads((folder / "trace.json").read_text())
    assert len(trace) == 3
    assert [len(t["tool_calls"]) for t in trace] == [2, 1, 0]
    assert trace[1]["tool_calls"][0]["input"]["target_column"] == "target"
    assert "utility_gap" in trace[1]["tool_calls"][0]["result"]

    usage = json.loads((folder / "usage.json").read_text())
    assert usage["usage"]["input_tokens"] == 4500
    assert usage["cost"]["priced"] is False  # fake model has no list price
    assert usage["api_latency_s"] >= 0 and usage["tool_latency_s"] >= 0

    assert (folder / "report.md").read_text().startswith("Overall verdict: excellent.")


def test_failed_run_is_logged_then_reraised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient([_response([_tool_use("cohen_d_numerical", {"real_csv_path": "nope.csv", "synth_csv_path": "nope.csv"}, "t1")], "tool_use", 10, 5)])
    monkeypatch.setattr(agent, "get_client", lambda: client)
    with pytest.raises(RuntimeError, match="boom"):
        run_agent("evaluate", model="claude-fake-1", log_dir=tmp_path / "runs")
    folders = list((tmp_path / "runs").iterdir())
    assert len(folders) == 1
    meta = json.loads((folders[0] / "meta.json").read_text())
    assert meta["status"] == "error" and "boom" in meta["error"]
    assert meta["turns"] == 1
    trace = json.loads((folders[0] / "trace.json").read_text())
    assert trace[0]["tool_calls"][0]["is_error"] is True  # the missing file errored inside the tool


def test_log_dir_none_writes_nothing(csvs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real, synth = csvs
    monkeypatch.setattr(agent, "get_client", lambda: _scripted_client(real, synth))
    monkeypatch.chdir(tmp_path)
    run = run_agent("evaluate", model="claude-fake-1", log_dir=None)
    assert run.log_dir is None and not (tmp_path / "runs").exists()


def test_max_turns_is_recorded(csvs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real, synth = csvs
    paths = {"real_csv_path": str(real), "synth_csv_path": str(synth)}
    client = FakeClient([_response([_tool_use("cohen_d_numerical", dict(paths), f"t{i}")], "tool_use", 10, 5) for i in range(2)])
    monkeypatch.setattr(agent, "get_client", lambda: client)
    run = run_agent("evaluate", max_turns=2, model="claude-fake-1", log_dir=tmp_path)
    assert run.stop_reason == "max_turns" and len(run.steps) == 2


# --------------------------------------------------------------------------- cost
def test_estimate_cost_known_and_unknown_model() -> None:
    est = estimate_cost("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=100_000)
    assert est.priced and est.pricing_date == PRICING_DATE
    assert est.input_usd == pytest.approx(3.0) and est.output_usd == pytest.approx(1.5)
    assert est.total_usd == pytest.approx(4.5)
    cached = estimate_cost("claude-sonnet-5-5", input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000)
    assert cached.cache_read_usd == pytest.approx(0.20)
    unknown = estimate_cost("claude-made-up", input_tokens=10, output_tokens=10)
    assert not unknown.priced and unknown.total_usd == 0.0 and "claude-made-up" in unknown.note


def test_pricing_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    table = tmp_path / "p.json"
    table.write_text(json.dumps({"claude-fake-1": {"input": 1.0, "output": 2.0}}))
    monkeypatch.setenv("PRICING_JSON", str(table))
    est = estimate_cost("claude-fake-1", input_tokens=500_000, output_tokens=500_000)
    assert est.priced and est.total_usd == pytest.approx(1.5)


# --------------------------------------------------------------------------- serialiser and summarizer
def _fake_run(real: Path, synth: Path) -> AgentRun:
    run = AgentRun(text="Verdict: poor.\n\n" + json.dumps({"verdict": {"overall": "poor"}, "flagged_columns": []}),
                   model_requested="claude-sonnet-4-6", model_served="claude-sonnet-4-6", user_prompt="p",
                   stop_reason="end_turn", started_at="2026-10-19T00:00:00+00:00", finished_at="2026-10-19T00:00:09+00:00", latency_s=9.0)
    step = ApiStep(turn=0, latency_s=4.0, input_tokens=100, output_tokens=10, cache_read_input_tokens=0,
                   cache_creation_input_tokens=0, stop_reason="end_turn", request_id="r", n_tool_calls=1)
    run.steps.append(step)
    run.usage = Usage(100, 10, 0, 0)
    run.tool_calls.append(ToolCall("tstr_xgboost", {"real_csv_path": str(real), "synth_csv_path": str(synth), "target_column": "target"}, {"utility_gap": 0.2}, latency_s=2.5, turn=0))
    return run


def test_run_to_dict_and_summarizer(csvs: tuple[Path, Path], tmp_path: Path) -> None:
    real, synth = csvs
    run = _fake_run(real, synth)
    d = run_to_dict(run)
    assert d["turns"] == 1 and d["cost"]["priced"] and d["cost"]["total_usd"] == pytest.approx(100 * 3 / 1e6 + 10 * 15 / 1e6)
    assert "messages" not in d

    folder = write_run(run, tmp_path / "runs")
    rows = [summarize.load_run(folder)]
    assert rows[0] is not None
    assert rows[0]["verdict"] == "poor" and rows[0]["tools"] == 1 and rows[0]["cost_usd"] == pytest.approx(d["cost"]["total_usd"])
    assert rows[0]["tool_latency_s"] == pytest.approx(2.5) and rows[0]["api_latency_s"] == pytest.approx(4.0)
    text = summarize.table([r for r in rows if r])
    assert run.run_id in text and "poor" in text and "list prices as of" in text
    assert summarize.main(["--runs", str(tmp_path / "runs")]) == 0


def test_describe_csv_handles_missing_file(tmp_path: Path) -> None:
    info = describe_csv(tmp_path / "missing.csv", "real")
    assert info["error"] == "file not found" and "sha256" not in info
