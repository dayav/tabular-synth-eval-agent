"""API tests: the real FastAPI app and the real agent loop, with a fake model."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "tests"):
    sys.path.insert(0, str(p))

from fake_client import EchoClient  # noqa: E402

import agent  # noqa: E402
import api  # noqa: E402


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(api, "ALLOWED_MODELS", ("claude-fake-1", "claude-sonnet-4-6"))
    monkeypatch.setattr(api, "API_DEFAULT_MODEL", "claude-fake-1")
    monkeypatch.setattr(agent, "get_client", lambda: EchoClient())
    return TestClient(api.app)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


def _frames(n: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    real = pd.DataFrame({"c": ["a", "b"] * (n // 2), "x": range(n), "target": [0, 1] * (n // 2)})
    synth = real.sample(frac=1, random_state=3).reset_index(drop=True)
    return real, synth


def _files(real: pd.DataFrame, synth: pd.DataFrame, real_name: str = "real.csv", synth_name: str = "synth.csv") -> dict:
    return {
        "real": (real_name, io.BytesIO(_csv_bytes(real)), "text/csv"),
        "synthetic": (synth_name, io.BytesIO(_csv_bytes(synth)), "text/csv"),
    }


def test_health(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["api_key_configured"] is True and body["runs_dir_writable"] is True
    assert "claude-fake-1" in body["allowed_models"]


def test_evaluate_then_fetch_run(client: TestClient, tmp_path: Path) -> None:
    real, synth = _frames()
    r = client.post("/evaluate", files=_files(real, synth), data={"target_column": "target"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok" and body["model_served"] == "claude-fake-1"
    assert body["tools_called"] == ["js_divergence_categorical", "cohen_d_numerical", "tstr_xgboost"]
    assert body["verdict"]["overall"] == "excellent" and body["flagged_columns"] == []
    assert body["usage"]["input_tokens"] == 4200 and body["cost"]["priced"] is False
    assert body["report"].startswith("**Overall verdict: excellent.**")

    run_id = body["run_id"]
    assert (tmp_path / "runs" / run_id / "meta.json").exists()

    r = client.get(f"/runs/{run_id}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["meta"]["run_id"] == run_id and detail["trace"] is None
    files = {f["role"]: f for f in detail["meta"]["inputs"]["files"]}
    assert files["real"]["rows"] == 40 and files["real"]["path"].endswith("real.csv")

    r = client.get(f"/runs/{run_id}?include_trace=true")
    assert len(r.json()["trace"]) == 2 and len(r.json()["trace"][0]["tool_calls"]) == 3

    r = client.get("/runs")
    assert r.status_code == 200 and [x["run_id"] for x in r.json()] == [run_id]
    assert r.json()[0]["verdict_overall"] == "excellent"


def test_upload_name_does_not_reach_the_model(client: TestClient) -> None:
    real, synth = _frames()
    r = client.post("/evaluate", files=_files(real, synth, "real_data.csv", "shuffled_income_leak.csv"), data={"target_column": "target"})
    assert r.status_code == 200
    prompt = r.json()["report"]  # the report is fixed; check the prompt through the logged meta instead
    detail = client.get(f"/runs/{r.json()['run_id']}").json()
    assert "shuffled" not in detail["meta"]["prompt"] and "leak" not in detail["meta"]["prompt"]
    assert prompt


@pytest.mark.parametrize(
    "mutate, status, fragment",
    [
        (lambda r, s: (r, s.drop(columns=["x"])), 400, "column sets differ"),
        (lambda r, s: (r, s), 400, "target column 'nope'"),
        (lambda r, s: (r.head(5), s.head(5)), 400, "at least 10 rows"),
        (lambda r, s: (r.assign(target=0), s), 400, "fewer than two classes"),
    ],
)
def test_validation_errors(client: TestClient, mutate, status: int, fragment: str) -> None:  # type: ignore[no-untyped-def]
    real, synth = mutate(*_frames())
    target = "nope" if "nope" in fragment else "target"
    r = client.post("/evaluate", files=_files(real, synth), data={"target_column": target})
    assert r.status_code == status, r.text
    assert fragment in r.json()["detail"]


def test_non_csv_and_garbage_rejected(client: TestClient) -> None:
    real, synth = _frames()
    files = {"real": ("real.txt", io.BytesIO(b"hello"), "text/plain"), "synthetic": ("s.csv", io.BytesIO(_csv_bytes(synth)), "text/csv")}
    r = client.post("/evaluate", files=files, data={"target_column": "target"})
    assert r.status_code == 400 and "expected a .csv" in r.json()["detail"]
    files = {"real": ("real.csv", io.BytesIO(b"\x00\x01\x02"), "text/csv"), "synthetic": ("s.csv", io.BytesIO(_csv_bytes(synth)), "text/csv")}
    r = client.post("/evaluate", files=files, data={"target_column": "target"})
    assert r.status_code == 400


def test_upload_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 200)
    real, synth = _frames()
    r = client.post("/evaluate", files=_files(real, synth), data={"target_column": "target"})
    assert r.status_code == 413


def test_model_allowlist(client: TestClient) -> None:
    real, synth = _frames()
    r = client.post("/evaluate", files=_files(real, synth), data={"target_column": "target", "model": "claude-made-up"})
    assert r.status_code == 400 and "not allowed" in r.json()["detail"]


def test_unknown_run_and_traversal(client: TestClient) -> None:
    assert client.get("/runs/20260101_000000_abcdef").status_code == 404
    assert client.get("/runs/..%2F..%2Fetc").status_code == 404
    assert client.get("/runs/not-a-run-id").status_code == 404


def test_busy_returns_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    monkeypatch.setattr(api, "_slots", threading.BoundedSemaphore(1))
    api._slots.acquire()
    try:
        real, synth = _frames()
        r = client.post("/evaluate", files=_files(real, synth), data={"target_column": "target"})
        assert r.status_code == 503
    finally:
        api._slots.release()


def test_upstream_failure_is_502(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic
    import httpx2

    class Failing:
        messages = property(lambda self: self)

        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.APIConnectionError(request=req)

    monkeypatch.setattr(agent, "get_client", lambda: Failing())
    real, synth = _frames()
    r = client.post("/evaluate", files=_files(real, synth), data={"target_column": "target"})
    assert r.status_code == 502 and "upstream" in r.json()["detail"]


def test_assess_privacy_flag_changes_the_prompt(client: TestClient) -> None:
    real, synth = _frames()
    r = client.post("/evaluate", files=_files(real, synth), data={"target_column": "target", "assess_privacy": "true"})
    assert r.status_code == 200, r.text
    detail = client.get(f"/runs/{r.json()['run_id']}").json()
    assert "privacy" in detail["meta"]["prompt"] and "disclosure risk" in detail["meta"]["prompt"]
    r2 = client.post("/evaluate", files=_files(real, synth), data={"target_column": "target"})
    assert "privacy" not in client.get(f"/runs/{r2.json()['run_id']}").json()["meta"]["prompt"]
