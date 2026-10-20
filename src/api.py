"""HTTP service around the agent.

    POST /evaluate      two CSV uploads + target column -> run id, verdict, report
    GET  /runs          recent runs (from the run log folders)
    GET  /runs/{id}     one run: meta, usage, report, optionally the trace
    GET  /health        liveness plus the configuration that matters

Run with ``uvicorn api:app`` from ``src/`` (or ``make api``). Configuration is
read from the environment only; the API key is never written anywhere.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Annotated, Any

import anthropic
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from agent import DEFAULT_MODEL, AgentRun, run_agent
from runlog import DEFAULT_RUNS_DIR, run_to_dict

# ----------------------------------------------------------------------------- settings
RUNS_DIR = Path(os.environ.get("RUNS_DIR", str(DEFAULT_RUNS_DIR)))
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024)
MAX_ROWS = int(os.environ.get("MAX_ROWS", "200000"))
MIN_ROWS = 10
ALLOWED_MODELS: tuple[str, ...] = tuple(
    m.strip()
    for m in os.environ.get("ALLOWED_MODELS", "claude-sonnet-4-6,claude-sonnet-5-5,claude-haiku-5-5").split(",")
    if m.strip()
)
API_DEFAULT_MODEL = os.environ.get("API_DEFAULT_MODEL", DEFAULT_MODEL)
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_EVALUATIONS", "2"))
MAX_TURNS = int(os.environ.get("API_MAX_TURNS", "10"))

RUN_ID_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{6}$")
PROMPT_TEMPLATE = (
    "Evaluate the synthetic dataset at {synth} against the real dataset at {real}. "
    "The target column for utility evaluation is '{target}'."
)

def _version() -> str:
    """APP_VERSION if set (the container sets it at build time), else the installed package's version."""
    if os.environ.get("APP_VERSION"):
        return os.environ["APP_VERSION"]
    try:
        return pkg_version("tabular-synth-eval-agent")
    except PackageNotFoundError:  # running from a checkout without an install
        return "0.0.0+local"


VERSION = _version()

_slots = threading.BoundedSemaphore(MAX_CONCURRENT)


# ----------------------------------------------------------------------------- schemas
class Verdict(BaseModel):
    categorical: str | None = None
    numerical: str | None = None
    utility: str | None = None
    overall: str | None = None


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class CostOut(BaseModel):
    priced: bool
    total_usd: float
    pricing_date: str
    note: str = ""


class EvaluateResponse(BaseModel):
    run_id: str
    status: str = Field(description="'ok' or 'error'")
    model_requested: str
    model_served: str
    stop_reason: str
    latency_s: float
    tools_called: list[str]
    verdict: Verdict | None = Field(default=None, description="parsed from the report's trailing JSON block")
    flagged_columns: list[str] = Field(default_factory=list)
    usage: UsageOut
    cost: CostOut
    report: str


class RunSummary(BaseModel):
    run_id: str
    started_at: str
    model_served: str
    status: str
    latency_s: float | None = None
    estimated_cost_usd: float | None = None
    verdict_overall: str | None = None


class RunDetail(BaseModel):
    meta: dict[str, Any]
    usage: dict[str, Any]
    report: str
    trace: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    default_model: str
    allowed_models: list[str]
    api_key_configured: bool
    runs_dir: str
    runs_dir_writable: bool
    max_upload_mb: float
    max_rows: int


class ErrorResponse(BaseModel):
    detail: str


# ----------------------------------------------------------------------------- helpers
def parse_verdict(report: str) -> tuple[Verdict | None, list[str]]:
    """The last ``{"verdict": ...}`` object in the report, or ``(None, [])``."""
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    for m in re.finditer(r"\{\s*\"verdict\"", report):
        try:
            obj, _ = decoder.raw_decode(report, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found = obj
    if not found:
        return None, []
    v = found.get("verdict") or {}
    flagged = [str(c) for c in (found.get("flagged_columns") or [])]
    return Verdict(**{k: v.get(k) for k in ("categorical", "numerical", "utility", "overall")}), flagged


def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = upload.file.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, f"{upload.filename or 'upload'} exceeds the {limit // (1024 * 1024)} MB upload limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _save_csv(upload: UploadFile, dest: Path, role: str) -> pd.DataFrame:
    name = upload.filename or ""
    if not name.lower().endswith(".csv") and (upload.content_type or "") not in ("text/csv", "application/csv"):
        raise HTTPException(400, f"{role}: expected a .csv upload, got {name!r} ({upload.content_type})")
    data = _read_limited(upload, MAX_UPLOAD_BYTES)
    if not data.strip():
        raise HTTPException(400, f"{role}: the upload is empty")
    dest.write_bytes(data)
    try:
        frame = pd.read_csv(dest)
    except Exception as e:  # noqa: BLE001 - any parse failure is the client's problem
        raise HTTPException(400, f"{role}: could not parse CSV ({type(e).__name__}: {e})") from e
    if frame.shape[0] < MIN_ROWS:
        raise HTTPException(400, f"{role}: at least {MIN_ROWS} rows are required, got {frame.shape[0]}")
    if frame.shape[0] > MAX_ROWS:
        raise HTTPException(413, f"{role}: at most {MAX_ROWS} rows are accepted, got {frame.shape[0]}")
    if frame.shape[1] < 2:
        raise HTTPException(400, f"{role}: at least two columns are required")
    return frame


def _validate_pair(real: pd.DataFrame, synth: pd.DataFrame, target: str) -> None:
    if set(real.columns) != set(synth.columns):
        only_real = sorted(set(real.columns) - set(synth.columns))
        only_synth = sorted(set(synth.columns) - set(real.columns))
        raise HTTPException(400, f"column sets differ: only in real {only_real}, only in synthetic {only_synth}")
    if target not in real.columns:
        raise HTTPException(400, f"target column {target!r} is not in the data; columns are {list(real.columns)}")
    if real[target].nunique(dropna=True) < 2:
        raise HTTPException(400, f"target column {target!r} has fewer than two classes in the real data")
    if synth[target].nunique(dropna=True) < 2:
        raise HTTPException(400, f"target column {target!r} has fewer than two classes in the synthetic data")


def _iter_run_dirs(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for d in sorted(root.iterdir(), reverse=True):
        if d.is_dir() and RUN_ID_RE.match(d.name) and (d / "meta.json").exists():
            yield d


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _summary(folder: Path) -> RunSummary:
    meta, usage = _load_json(folder / "meta.json"), _load_json(folder / "usage.json")
    report = (folder / "report.md").read_text(encoding="utf-8") if (folder / "report.md").exists() else ""
    verdict, _ = parse_verdict(report)
    cost = usage.get("cost", {})
    return RunSummary(
        run_id=meta.get("run_id", folder.name),
        started_at=meta.get("started_at", ""),
        model_served=meta.get("model_served") or meta.get("model_requested", ""),
        status=meta.get("status", ""),
        latency_s=meta.get("latency_s"),
        estimated_cost_usd=cost.get("total_usd") if cost.get("priced") else None,
        verdict_overall=verdict.overall if verdict else None,
    )


def _evaluate_response(run: AgentRun) -> EvaluateResponse:
    d = run_to_dict(run)
    verdict, flagged = parse_verdict(run.text)
    return EvaluateResponse(
        run_id=run.run_id,
        status=d["status"],
        model_requested=run.model_requested,
        model_served=run.model_served,
        stop_reason=run.stop_reason,
        latency_s=run.latency_s,
        tools_called=[tc.name for tc in run.tool_calls],
        verdict=verdict,
        flagged_columns=flagged,
        usage=UsageOut(**d["usage"]),
        cost=CostOut(**{k: d["cost"][k] for k in ("priced", "total_usd", "pricing_date", "note")}),
        report=run.text,
    )


# ----------------------------------------------------------------------------- app
app = FastAPI(
    title="tabular-synth-eval-agent",
    version=VERSION,
    description=(
        "Evaluates a synthetic tabular dataset against a real one with a Claude agent that calls "
        "three typed tools (JS distance, Cohen's d, TSTR/TRTR via XGBoost). Public demo deployment; "
        "every run is logged under the runs directory."
    ),
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        writable = os.access(RUNS_DIR, os.W_OK)
    except OSError:
        writable = False
    return HealthResponse(
        status="ok",
        version=VERSION,
        default_model=API_DEFAULT_MODEL,
        allowed_models=list(ALLOWED_MODELS),
        api_key_configured=bool(os.environ.get("ANTHROPIC_API_KEY")),
        runs_dir=str(RUNS_DIR),
        runs_dir_writable=writable,
        max_upload_mb=MAX_UPLOAD_BYTES / (1024 * 1024),
        max_rows=MAX_ROWS,
    )


@app.post(
    "/evaluate",
    response_model=EvaluateResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def evaluate(
    real: Annotated[UploadFile, File(description="the real dataset, CSV")],
    synthetic: Annotated[UploadFile, File(description="the synthetic dataset, CSV, same columns")],
    target_column: Annotated[str, Form(min_length=1, max_length=200, description="classification target present in both files")],
    model: Annotated[str | None, Form(description="one of the allowed models (see /health); default is the service's default")] = None,
) -> EvaluateResponse:
    """Run the agent on one real/synthetic pair. Synchronous: returns when the report is written."""
    chosen = model or API_DEFAULT_MODEL
    if chosen not in ALLOWED_MODELS:
        raise HTTPException(400, f"model {chosen!r} is not allowed; choose one of {list(ALLOWED_MODELS)}")

    if not _slots.acquire(blocking=False):
        raise HTTPException(503, f"busy: {MAX_CONCURRENT} evaluation(s) already running, retry shortly")
    workdir = Path(tempfile.mkdtemp(prefix="eval_api_"))
    try:
        # neutral file names: the model reads paths, and an upload's name must not leak a hint
        real_path, synth_path = workdir / "real.csv", workdir / "synth.csv"
        real_frame = _save_csv(real, real_path, "real")
        synth_frame = _save_csv(synthetic, synth_path, "synthetic")
        _validate_pair(real_frame, synth_frame, target_column)

        prompt = PROMPT_TEMPLATE.format(real=real_path, synth=synth_path, target=target_column)
        try:
            run = run_agent(prompt, max_turns=MAX_TURNS, model=chosen, log_dir=RUNS_DIR)
        except anthropic.AuthenticationError as e:
            raise HTTPException(502, "the service's API key was rejected upstream") from e
        except anthropic.RateLimitError as e:
            raise HTTPException(503, "upstream rate limit; retry shortly") from e
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            raise HTTPException(502, f"upstream API error: {type(e).__name__}") from e
        return _evaluate_response(run)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        _slots.release()


@app.get("/runs", response_model=list[RunSummary])
def list_runs(limit: int = Query(50, ge=1, le=500)) -> list[RunSummary]:
    """Most recent runs first, read from the run log folders."""
    out: list[RunSummary] = []
    for folder in _iter_run_dirs(RUNS_DIR):
        out.append(_summary(folder))
        if len(out) >= limit:
            break
    return out


@app.get("/runs/{run_id}", response_model=RunDetail, responses={404: {"model": ErrorResponse}})
def get_run(run_id: str, include_trace: bool = Query(False)) -> RunDetail:
    """One run's meta, usage and report; add ``?include_trace=true`` for the step-by-step trace."""
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(404, "no such run")
    folder = RUNS_DIR / run_id
    if not (folder / "meta.json").exists():
        raise HTTPException(404, "no such run")
    report = (folder / "report.md").read_text(encoding="utf-8") if (folder / "report.md").exists() else ""
    trace = _load_json_list(folder / "trace.json") if include_trace else None
    return RunDetail(meta=_load_json(folder / "meta.json"), usage=_load_json(folder / "usage.json"), report=report, trace=trace)


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    return data if isinstance(data, list) else []
