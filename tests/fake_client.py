"""A scripted stand-in for the Anthropic client, shaped like the SDK's responses.

Used by the run-logging and API tests so the real agent loop runs end to end
without a key or a network call.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def response(blocks: list[Any], stop_reason: str, in_tok: int, out_tok: int, cache_read: int = 0, model: str = "claude-fake-1") -> Any:
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok,
                              cache_read_input_tokens=cache_read, cache_creation_input_tokens=0),
        _request_id="req_test",
    )


def tool_use(name: str, inp: dict[str, Any], id_: str) -> Any:
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=id_)


def text(t: str) -> Any:
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


REPORT = (
    "**Overall verdict: excellent.** Everything is fine.\n\n```json\n"
    '{"verdict": {"categorical": "good", "numerical": "good", "utility": "good", "overall": "excellent"}, "flagged_columns": []}'
    "\n```"
)


class EchoClient:
    """Calls the three tools on whatever paths the prompt names, then writes a fixed report.

    The prompt is parsed from the first user message, so this works for any
    caller that follows the agent's prompt template.
    """

    def __init__(self, model: str = "claude-fake-1", report: str = REPORT) -> None:
        self.model = model
        self.report = report
        self.messages = self
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        import re

        self.calls.append(kwargs)
        if len(self.calls) == 1:
            prompt = kwargs["messages"][0]["content"]
            m = re.search(r"dataset at (\S+) against the real dataset at (\S+)\. .*'([^']+)'", prompt)
            assert m, prompt
            synth, real, target = m.group(1), m.group(2), m.group(3)
            paths = {"real_csv_path": real, "synth_csv_path": synth}
            return response(
                [tool_use("js_divergence_categorical", dict(paths), "t1"),
                 tool_use("cohen_d_numerical", dict(paths), "t2"),
                 tool_use("tstr_xgboost", {**paths, "target_column": target}, "t3")],
                "tool_use", 1800, 300, model=self.model,
            )
        return response([text(self.report)], "end_turn", 2400, 700, model=self.model)
