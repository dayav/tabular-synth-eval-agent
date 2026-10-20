"""Agent orchestration: dispatches user prompts to Claude with tool use.

``run_agent`` returns an :class:`AgentRun` carrying the final report, the ordered
tool trace, token usage, the stop reason and the model that actually served the
request, so that the evaluation harness in ``evals/`` can grade a run without
re-parsing the conversation.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam
from dotenv import load_dotenv

from report import save_report
from runlog import DEFAULT_RUNS_DIR, write_run
from tools import (
    FidelityResult,
    UtilityResult,
    cohen_d_numerical,
    js_divergence_categorical,
    tstr_xgboost,
)

load_dotenv()

DEFAULT_MODEL: str = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS: int = 16000

JS_THRESHOLD: float = 0.1
COHEN_D_THRESHOLD: float = 0.2
UTILITY_GAP_THRESHOLD: float = 0.05

_client: Anthropic | None = None


def get_client() -> Anthropic:
    """Return the shared API client, creating it on first use.

    Creating the client lazily means importing this module needs no API key,
    which keeps the offline regression suite and the checkers importable.
    """
    global _client
    if _client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Put it in the environment or in a .env file."
            )
        _client = Anthropic()
    return _client


SYSTEM_PROMPT: str = f"""You are an expert in evaluating synthetic tabular data.

You have three tools available:
- js_divergence_categorical: fidelity of categorical columns (Jensen-Shannon distance)
- cohen_d_numerical: fidelity of numerical columns (Cohen's d)
- tstr_xgboost: predictive utility (train-on-synthetic vs train-on-real)

When asked to evaluate a synthetic dataset, call each of the three tools exactly once,
with the real and synthetic paths you were given and, for tstr_xgboost, the target column
you were given. Then write a short plain-English report.

Thresholds (the only thresholds you may use):
- A categorical column is flagged when its JS distance is above {JS_THRESHOLD}.
- A numerical column is flagged when its Cohen's d is above {COHEN_D_THRESHOLD}.
- Utility is poor when the utility gap (TRTR accuracy minus TSTR accuracy) is above
  {UTILITY_GAP_THRESHOLD}. A gap at or below 0 is good: the synthetic data lost no utility.

Verdict rules (apply them mechanically):
- categorical fidelity is "poor" if any categorical column is flagged, otherwise "good".
- numerical fidelity is "poor" if any numerical column is flagged, otherwise "good".
- utility is "poor" if the gap is above its threshold, otherwise "good".
- overall is "excellent" when all three axes are good, "poor" when utility is poor, and
  "acceptable" when utility is good but at least one fidelity axis is poor.

Rules for the report:
- You are a reporter, not a calculator. Quote numbers exactly as the tools returned them
  (rounding is fine). Never compute a new figure of your own: no means, counts, ratios,
  differences, percentages or factors that are not in a tool result.
- Do not introduce any threshold other than the ones above.
- Name a column as a problem only if its own tool result crosses its threshold. If utility
  is poor but no column crosses a fidelity threshold, say that the cause is not visible in
  the univariate fidelity metrics; do not guess which column is responsible.
- The target column appears in the Cohen's d output because it is numeric. It is a label,
  not a feature: never flag it.
- State the overall verdict plainly in the first lines. Name every flagged column with its
  value and the threshold it crossed. Explain each metric in one sentence a non-specialist
  can follow. Mention the caveats that matter (a negative gap, a small test set, a utility
  problem whose cause is not visible). Be honest: if the data is poor on any axis, say so.

End the report with exactly one fenced JSON block of this form, and nothing after it:

```json
{{"verdict": {{"categorical": "good|poor", "numerical": "good|poor", "utility": "good|poor",
             "overall": "excellent|acceptable|poor"}},
 "flagged_columns": ["column", "..."]}}
```
"""

TOOLS: list[ToolParam] = [
    {
        "name": "js_divergence_categorical",
        "description": (
            "Compute the Jensen-Shannon distance between the real and synthetic "
            "datasets for each categorical column. The distance ranges from 0 "
            "(identical distributions) to 1 (completely different). Returns a "
            "per-column dict, the mean across columns, and the column count. Smaller "
            "values mean the synthetic categorical distributions are closer to real."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "real_csv_path": {
                    "type": "string",
                    "description": "Filesystem path to the real dataset CSV.",
                },
                "synth_csv_path": {
                    "type": "string",
                    "description": "Filesystem path to the synthetic dataset CSV.",
                },
            },
            "required": ["real_csv_path", "synth_csv_path"],
        },
    },
    {
        "name": "cohen_d_numerical",
        "description": (
            "Compute absolute Cohen's d effect size between the real and synthetic "
            "datasets for each numerical column. Returns a per-column dict, the mean "
            "across columns, and the column count. Smaller values mean the "
            "synthetic numerical distributions are closer to real."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "real_csv_path": {
                    "type": "string",
                    "description": "Filesystem path to the real dataset CSV.",
                },
                "synth_csv_path": {
                    "type": "string",
                    "description": "Filesystem path to the synthetic dataset CSV.",
                },
            },
            "required": ["real_csv_path", "synth_csv_path"],
        },
    },
    {
        "name": "tstr_xgboost",
        "description": (
            "Compute a utility evaluation of the synthetic dataset using XGBoost. "
            "Trains one XGBoost classifier on the synthetic data and one on the real "
            "data, then evaluates both on a held-out portion of the real data. "
            "Returns: tstr_accuracy (model trained on synthetic), trtr_accuracy "
            "(model trained on real, as baseline), utility_gap (= trtr_accuracy - "
            "tstr_accuracy; positive means synthetic loses utility, near zero means "
            "synthetic is as useful as real), target_column, n_test_samples, n_classes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "real_csv_path": {
                    "type": "string",
                    "description": "Filesystem path to the real dataset CSV.",
                },
                "synth_csv_path": {
                    "type": "string",
                    "description": "Filesystem path to the synthetic dataset CSV.",
                },
                "target_column": {
                    "type": "string",
                    "description": "Name of the column to predict (classification target).",
                },
                "test_csv_path": {
                    "type": "string",
                    "description": (
                        "Optional path to a held-out test CSV. "
                        "If omitted, the real dataset is split 80/20 internally."
                    ),
                },
            },
            "required": ["real_csv_path", "synth_csv_path", "target_column"],
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., FidelityResult | UtilityResult]] = {
    "js_divergence_categorical": js_divergence_categorical,
    "cohen_d_numerical": cohen_d_numerical,
    "tstr_xgboost": tstr_xgboost,
}


@dataclass
class ToolCall:
    """One tool invocation made by the model, with what came back."""

    name: str
    input: dict[str, Any]
    result: Mapping[str, Any]
    is_error: bool = False
    latency_s: float = 0.0
    turn: int = 0


@dataclass
class ApiStep:
    """One API call of the loop: how long it took, what it cost, how it ended."""

    turn: int
    latency_s: float
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    stop_reason: str
    request_id: str | None
    n_tool_calls: int = 0


@dataclass
class Usage:
    """Token usage summed over every API call of one run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, step: ApiStep) -> None:
        self.input_tokens += step.input_tokens
        self.output_tokens += step.output_tokens
        self.cache_read_input_tokens += step.cache_read_input_tokens
        self.cache_creation_input_tokens += step.cache_creation_input_tokens


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


@dataclass
class AgentRun:
    """Everything the harness and the run log need to describe one agent run."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: list[ApiStep] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""
    model_requested: str = DEFAULT_MODEL
    model_served: str = ""
    system_prompt: str = SYSTEM_PROMPT
    user_prompt: str = ""
    messages: list[Any] = field(default_factory=list)
    run_id: str = field(default_factory=new_run_id)
    started_at: str = ""
    finished_at: str = ""
    latency_s: float = 0.0
    error: str | None = None
    log_dir: Path | None = None

    def __str__(self) -> str:
        return self.text


def run_tool(tool_name: str, tool_input: dict[str, Any]) -> tuple[Mapping[str, Any], bool]:
    """Dispatch a tool call by name. Returns ``(result, is_error)``."""
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_name}"}, True
    try:
        return TOOL_REGISTRY[tool_name](**tool_input), False
    except Exception as e:  # noqa: BLE001 - the error text is what the model needs to see
        return {"error": f"{type(e).__name__}: {e}"}, True


def run_agent(
    user_prompt: str,
    max_turns: int = 10,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    log_dir: str | Path | None = DEFAULT_RUNS_DIR,
) -> AgentRun:
    """Run the agent loop until Claude stops calling tools.

    Returns an :class:`AgentRun`. ``str(run)`` (or ``run.text``) is the report.
    ``stop_reason`` is the API's stop reason of the last response, or
    ``"max_turns"`` if the loop was exhausted.

    Unless ``log_dir`` is ``None``, the run is written to ``<log_dir>/<run_id>/``
    (``runs/`` by default) with its inputs, tool trace, per-step latency, token
    usage, estimated cost and report, even when the run fails part-way.
    """
    client = get_client()
    messages: list[MessageParam] = [{"role": "user", "content": user_prompt}]
    run = AgentRun(text="", model_requested=model, user_prompt=user_prompt)
    run.started_at = datetime.now(UTC).isoformat(timespec="seconds")
    t_run = time.perf_counter()

    try:
        for turn in range(max_turns):
            t_api = time.perf_counter()
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                tools=TOOLS,
                messages=messages,
                system=SYSTEM_PROMPT,
            )
            step = ApiStep(
                turn=turn,
                latency_s=round(time.perf_counter() - t_api, 3),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
                cache_creation_input_tokens=response.usage.cache_creation_input_tokens or 0,
                stop_reason=str(response.stop_reason),
                request_id=getattr(response, "_request_id", None),
            )
            run.steps.append(step)
            run.usage.add(step)
            run.model_served = response.model
            run.stop_reason = str(response.stop_reason)

            # Append Claude's response to the conversation
            messages.append({"role": "assistant", "content": response.content})

            # If Claude is done calling tools, return the text
            if response.stop_reason != "tool_use":
                run.text = "\n\n".join(
                    block.text for block in response.content if block.type == "text"
                )
                break

            # Otherwise, run each tool and append the results
            tool_results: list[ToolResultBlockParam] = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_input = dict(block.input) if isinstance(block.input, Mapping) else {}
                    t_tool = time.perf_counter()
                    result, is_error = run_tool(block.name, tool_input)
                    run.tool_calls.append(
                        ToolCall(
                            name=block.name, input=tool_input, result=result, is_error=is_error,
                            latency_s=round(time.perf_counter() - t_tool, 3), turn=turn,
                        )
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                            "is_error": is_error,
                        }
                    )
            step.n_tool_calls = len(tool_results)
            messages.append({"role": "user", "content": tool_results})
        else:
            run.stop_reason = "max_turns"
    except Exception as e:
        run.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        run.messages = messages
        run.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
        run.latency_s = round(time.perf_counter() - t_run, 3)
        if log_dir is not None:
            run.log_dir = write_run(run, log_dir)
    return run


if __name__ == "__main__":
    user_prompt = (
        "Evaluate the synthetic dataset at examples/synth_sample.csv "
        "against the real dataset at examples/real_sample.csv. "
        "The target column for utility evaluation is 'target'."
    )
    final_run = run_agent(user_prompt)
    print(final_run.text)

    path = save_report(final_run)
    print(f"\nReport saved to: {path}")
    print(f"Run logged to:   {final_run.log_dir}")
