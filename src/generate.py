"""LLM tabular generation baseline: ``generate_with_llm``.

Given a column schema and per-column marginal statistics (never any real row), ask
Claude for N rows as strict JSON, validate every row against the schema, and save the
valid ones as CSV. It exists as a *baseline* to compare with CTGAN and TVAE on the same
metrics; it is not registered in the evaluation agent's tool list, because an evaluator
must not generate the data it grades.

What the model is given: column names, types, the category levels with their
frequencies, numeric quantiles, mean and standard deviation, and the target's class
balance. What it is not given: rows, pairwise statistics, or the real file. Any
relationship between columns in the output is therefore the model's prior, not the data's.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from pricing import estimate_cost

DEFAULT_GENERATOR_MODEL = "claude-sonnet-5-5"
MAX_ENUM_LEVELS = 60
INTEGER_AS_CATEGORY_MAX_LEVELS = 10


class ColumnSpec(TypedDict, total=False):
    name: str
    kind: str  # "categorical" | "integer" | "float"
    levels: list[str]
    frequencies: dict[str, float]
    min: float
    max: float
    mean: float
    std: float
    quantiles: dict[str, float]
    integer_levels: list[int]


class GenerationResult(TypedDict):
    output_csv_path: str
    rows_requested: int
    rows_written: int
    rows_rejected: int
    rejection_reasons: dict[str, int]
    batches: int
    model_requested: str
    model_served: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None
    latency_s: float
    columns: list[str]
    schema_summary: list[ColumnSpec]


# ----------------------------------------------------------------------------- schema
def summarise_schema(frame: pd.DataFrame, target_column: str | None = None, exclude: list[str] | None = None) -> list[ColumnSpec]:
    """Column kinds and marginal statistics, the only thing the generator is shown."""
    specs: list[ColumnSpec] = []
    drop = set(exclude or [])
    for col in frame.columns:
        if col in drop:
            continue
        s = frame[col].dropna()
        is_text = col in frame.select_dtypes(include=["object", "category", "string"]).columns
        is_int = pd.api.types.is_integer_dtype(s)
        n_levels = int(s.nunique())
        if is_text or (is_int and n_levels <= INTEGER_AS_CATEGORY_MAX_LEVELS) or (col == target_column and n_levels <= INTEGER_AS_CATEGORY_MAX_LEVELS):
            counts = s.astype(str).value_counts(normalize=True)
            top = counts.head(MAX_ENUM_LEVELS)
            spec: ColumnSpec = {
                "name": str(col),
                "kind": "categorical",
                "levels": [str(v) for v in top.index],
                "frequencies": {str(k): round(float(v), 4) for k, v in top.items()},
            }
            if is_int and not is_text:
                spec["integer_levels"] = sorted(int(v) for v in s.unique())
            specs.append(spec)
        else:
            probs = [0.05, 0.25, 0.5, 0.75, 0.95]
            q = s.quantile(probs)
            specs.append({
                "name": str(col),
                "kind": "integer" if is_int else "float",
                "min": float(s.min()), "max": float(s.max()),
                "mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4),
                "quantiles": {f"q{int(pr * 100):02d}": round(float(v), 4) for pr, v in zip(probs, q.to_numpy(), strict=True)},
            })
    return specs


def json_schema_for(specs: list[ColumnSpec]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for sp in specs:
        if sp["kind"] == "categorical":
            if "integer_levels" in sp:
                props[sp["name"]] = {"type": "integer", "enum": sp["integer_levels"]}
            else:
                props[sp["name"]] = {"type": "string", "enum": sp["levels"]}
        else:
            props[sp["name"]] = {"type": "integer" if sp["kind"] == "integer" else "number"}
    row = {"type": "object", "properties": props, "required": [sp["name"] for sp in specs], "additionalProperties": False}
    return {
        "type": "object",
        # the API accepts no array-size constraints beyond 0/1 items; the batch size is
        # asked for in the prompt and enforced by validation and trimming
        "properties": {"rows": {"type": "array", "items": row}},
        "required": ["rows"],
        "additionalProperties": False,
    }


def validate_row(row: dict[str, Any], specs: list[ColumnSpec], slack: float = 0.05) -> str | None:
    """Return a rejection reason, or None when the row is acceptable."""
    for sp in specs:
        name = sp["name"]
        if name not in row:
            return f"missing:{name}"
        v = row[name]
        if sp["kind"] == "categorical":
            allowed = sp.get("integer_levels")
            if allowed is not None:
                if not isinstance(v, int) or isinstance(v, bool) or v not in allowed:
                    return f"bad_level:{name}"
            elif str(v) not in sp["levels"]:
                return f"bad_level:{name}"
        else:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
                return f"not_numeric:{name}"
            if sp["kind"] == "integer" and float(v) != int(v):
                return f"not_integer:{name}"
            span = max(sp["max"] - sp["min"], abs(sp["max"]), 1.0)
            if v < sp["min"] - slack * span or v > sp["max"] + slack * span:
                return f"out_of_range:{name}"
    return None


# ----------------------------------------------------------------------------- generation
SYSTEM = """You generate synthetic rows for a tabular dataset. You are given only the column
schema and marginal statistics of the real data, never real rows. Produce rows that are
individually plausible, follow the given category frequencies, numeric ranges and quantiles
as closely as a sample of this size can, and keep sensible relationships between columns.
Vary the rows: no duplicates, no obviously repeated templates. Return only the JSON."""


def _user_prompt(specs: list[ColumnSpec], batch_size: int, batch_index: int, n_batches: int, target_column: str | None) -> str:
    stats = json.dumps(specs, indent=1)
    target = f"The classification target is '{target_column}'; respect its class balance.\n" if target_column else ""
    return (
        f"Generate batch {batch_index + 1} of {n_batches}: exactly {batch_size} rows.\n{target}"
        f"Column schema and marginal statistics of the real data:\n{stats}\n"
        "Numeric columns marked integer must be integers. Stay within the min..max of each column."
    )


def generate_with_llm(
    real_csv_path: str,
    output_csv_path: str,
    n_rows: int,
    target_column: str | None = None,
    *,
    model: str = DEFAULT_GENERATOR_MODEL,
    batch_size: int = 25,
    exclude_columns: list[str] | None = None,
    max_attempts_factor: int = 3,
    client: Any = None,
) -> GenerationResult:
    """Generate ``n_rows`` synthetic rows from the schema and marginals of ``real_csv_path``.

    The real file is read only to compute the schema summary; no row is sent to the model.
    Invalid rows (unknown category, out of range, wrong type) are dropped and more batches
    are requested, up to ``max_attempts_factor`` times the minimum number of batches.
    """
    if client is None:
        from agent import get_client

        client = get_client()
    frame = pd.read_csv(real_csv_path)
    specs = summarise_schema(frame, target_column=target_column, exclude=exclude_columns)
    columns = [sp["name"] for sp in specs]
    schema = json_schema_for(specs)
    n_batches = math.ceil(n_rows / batch_size)
    max_batches = n_batches * max_attempts_factor

    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    in_tok = out_tok = 0
    served = ""
    batches = 0
    t0 = time.perf_counter()
    while len(rows) < n_rows and batches < max_batches:
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": _user_prompt(specs, batch_size, batches, n_batches, target_column)}],
        )
        batches += 1
        in_tok += response.usage.input_tokens
        out_tok += response.usage.output_tokens
        served = response.model
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            reasons["unparseable_batch"] = reasons.get("unparseable_batch", 0) + 1
            continue
        for row in payload.get("rows", [])[:batch_size]:
            why = validate_row(row, specs)
            if why is None:
                rows.append({c: row[c] for c in columns})
            else:
                reasons[why] = reasons.get(why, 0) + 1
    rows = rows[:n_rows]
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(out, index=False)
    cost = estimate_cost(served or model, input_tokens=in_tok, output_tokens=out_tok)
    return {
        "output_csv_path": str(out),
        "rows_requested": n_rows,
        "rows_written": len(rows),
        "rows_rejected": sum(reasons.values()),
        "rejection_reasons": reasons,
        "batches": batches,
        "model_requested": model,
        "model_served": served,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "estimated_cost_usd": cost.total_usd if cost.priced else None,
        "latency_s": round(time.perf_counter() - t0, 2),
        "columns": columns,
        "schema_summary": specs,
    }
