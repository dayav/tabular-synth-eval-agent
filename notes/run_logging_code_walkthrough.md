# Run logging, file by file: a code walkthrough

Companion to [run_logging.md](run_logging.md), which explains *why* a run is recorded the way it is. This one explains *how the code does it*, in the order the data flows, with a pointer into the source for every claim, so that you can open the files next to someone and talk them through phase 2.

Conventions: `file:line` links point into the repository. Each section opens with the one idea it implements and where the theory lives, then shows the code.

## 0. The map

| What | Where | Theory in the note |
|---|---|---|
| The loop measures itself | [../src/agent.py](../src/agent.py) | sections 2, 4, 6 |
| Writing the folder and recovering the inputs | [../src/runlog.py](../src/runlog.py) | sections 2, 3 |
| Estimating cost from a dated price list | [../src/pricing.py](../src/pricing.py) | section 5 |
| One row per run | [../runs/summarize.py](../runs/summarize.py) | section 7 |
| The eval harness shares the record | [../evals/run_eval.py](../evals/run_eval.py) | section 8 |
| Offline tests with a fake API | [../tests/test_runlog.py](../tests/test_runlog.py) | section 6 |
| Figures | [figures/make_runlog_figures.py](figures/make_runlog_figures.py) | every figure |
| What is committed and what is not | [../.gitignore](../.gitignore), [../runs/README.md](../runs/README.md) | section 9 |

One sentence to hold on to: **the loop records facts as they happen; everything else is derived from those facts and could be recomputed.**

## 1. The loop measures itself

**Idea.** Instrument the two places where time and money are spent: around each API call and around each tool call. Keep the record on the `AgentRun` the loop already returns. (Note, sections 2 and 4.)

### 1.1 The step record

[agent.py:218-231](../src/agent.py#L218-L231):

```python
@dataclass
class ApiStep:
    turn: int
    latency_s: float
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    stop_reason: str
    request_id: str | None
    n_tool_calls: int = 0
```

One of these per API call. `Usage.add(step)` ([line 233](../src/agent.py#L233)) sums the four token fields, so the totals are always consistent with the steps. `ToolCall` gained `latency_s` and `turn`, the turn being what lets `trace.json` nest tool calls under the step that asked for them.

`AgentRun` ([line 253](../src/agent.py#L253)) gained `steps`, `run_id` (a timestamp plus six hex characters, [line 248](../src/agent.py#L248)), `started_at`, `finished_at`, `latency_s`, `error` and `log_dir`.

### 1.2 The timing points

Read [run_agent](../src/agent.py#L287-L381) with the learner and find the four clocks:

- [line 309](../src/agent.py#L309): `t_run` starts before the first API call.
- [line 313](../src/agent.py#L313): `t_api` starts just before `client.messages.create` and is read into the `ApiStep` at [line 321](../src/agent.py#L321), together with `response.usage.*`, `response.stop_reason` and `response._request_id`. The cache fields can be `None` on some responses, hence `or 0`.
- [line 351](../src/agent.py#L351): `t_tool` wraps `run_tool` for each tool call; the `ToolCall` is built with its latency and the current `turn` ([line 356](../src/agent.py#L356)).
- [line 367](../src/agent.py#L367): after the tools, the step learns how many calls it triggered.

Why measure tools separately from API calls rather than subtracting: because the two overlap with nothing and add up to the wall time, so three numbers (wall, API, tool) check each other. The example run has 18.834 = 18.738 + 0.096.

### 1.3 Three control-flow details worth teaching

- **`for ... else`** ([line 369](../src/agent.py#L369)). The `else` clause of a `for` loop runs only if the loop was not broken out of. The loop `break`s when the model stops asking for tools; if it instead runs out of turns, the `else` sets `stop_reason = "max_turns"`. Many people have never seen this construct; it is the cleanest way to say "the budget ran out".
- **`except ... raise`** ([line 371](../src/agent.py#L371)). The exception text is stored on `run.error` and the exception is re-raised unchanged. The caller still sees the failure; the log still gets written.
- **`finally`** ([line 374](../src/agent.py#L374)). Finishing time, wall latency and the folder write happen whether the loop returned, broke or raised. This is the whole of section 6 of the note in four lines.

### 1.4 Opting out

`log_dir` defaults to `runs/` ([line 293](../src/agent.py#L293)). Passing `None` skips the folder entirely; the `AgentRun` is still complete, which is how the eval harness uses it.

## 2. Writing the folder

**Idea.** Serialise the record once, derive four files from it, and recover the inputs from the tool calls rather than the prompt. (Note, sections 2 and 3.)

[runlog.py](../src/runlog.py), top to bottom:

- `describe_csv` ([lines 42-60](../src/runlog.py#L42-L60)): size, SHA-256 (streamed in 1 MiB chunks, [line 34](../src/runlog.py#L34)), rows, columns and column names. It never raises: a missing or unreadable file becomes an `error` field. Metadata must not fail a run that otherwise succeeded.
- `input_metadata` ([lines 63-74](../src/runlog.py#L63-L74)): walks the tool calls and collects every `real_csv_path`, `synth_csv_path`, `test_csv_path` and the first `target_column`. Explain why this is better than parsing the prompt: it is what the agent actually opened.
- `git_revision` ([line 77](../src/runlog.py#L77)): `git rev-parse --short HEAD` with a timeout; `None` outside a repository.
- `run_to_dict` ([lines 87-113](../src/runlog.py#L87-L113)): the shared serialiser. It uses `dataclasses.asdict` on steps, tool calls and usage, calls `estimate_cost` on the served model, and deliberately leaves out `messages`, which holds raw SDK objects.
- `write_run` ([lines 116-148](../src/runlog.py#L116-L148)): builds `meta` (identity, timings, prompt, inputs, tools called, git revision, system-prompt hash), groups tool calls by `turn` into their step to make `trace`, computes the API and tool latency sums for `usage`, and writes the four files. The folder name is the run id.

A point for the learner: `write_run` does not know about the eval harness, and `run_eval.py` does not know about folders. Both call `run_to_dict`. That is the seam.

## 3. Estimating cost

**Idea.** Tokens times a dated list price, with "unknown" preferred to "guessed". (Note, section 5.)

[pricing.py](../src/pricing.py):

- `PRICING_DATE` and `DEFAULT_PRICING` ([lines 21-46](../src/pricing.py#L21-L46)): USD per million tokens, input and output, with explicit cache-read prices where published and defaults otherwise (0.1× input for reads, 1.25× for writes). The Haiku 5.5 long-prompt tier is noted in a comment and not modelled.
- `load_pricing` ([line 49](../src/pricing.py#L49)): the default table or the JSON file named by `PRICING_JSON`.
- `estimate_cost` ([lines 74-112](../src/pricing.py#L74-L112)): returns a `CostEstimate` with the four components and the total, rounded to six decimals, plus `priced`, `pricing_date` and a `note`. A model missing from the table yields `priced=False`, zero cost and a note that says so.

Ask the learner why the estimate names its date. Because a cost figure without a date is unfalsifiable; with the date, anyone can re-price the run from the token counts, which are the real record.

## 4. The summary table

**Idea.** The smallest useful consumer of the folder format, with no dependency on the agent code. (Note, section 7.)

[summarize.py](../runs/summarize.py):

- `overall_verdict` ([lines 20-30](../runs/summarize.py#L20-L30)) finds the last `{"verdict": ...}` object in the report with `json.JSONDecoder.raw_decode`, the same trick the eval harness uses, and returns its `overall` value. It is re-implemented here rather than imported so the script stays standalone.
- `load_run` ([lines 33-62](../runs/summarize.py#L33-L62)) reads the three JSON files and the report and flattens them into one row: model served, status, synthetic file name and row count, tool and turn counts, wall, API and tool latency, tokens, cache reads, cost (only if priced) and the verdict.
- `table` ([lines 71-95](../runs/summarize.py#L71-L95)) renders Markdown and, if any run is priced, a footer with the total and the price-list dates involved.
- `main` ([line 98](../runs/summarize.py#L98)) takes `--runs DIR` (default: the script's own folder) and `--json`.

## 5. The eval harness shares the record

[run_eval.py:79](../evals/run_eval.py#L79) calls `run_agent(..., log_dir=None)`: the transcript it writes is the log for that run. [Lines 121-124](../evals/run_eval.py#L121-L124) add `steps` and `cost` to the transcript through `run_to_dict`, and the summary gains `estimated_cost_usd` ([line 193](../evals/run_eval.py#L193)). Nothing else in the harness changed; the checkers never look at timing.

## 6. Testing without the API

**Idea.** Drive the real loop with a fake client, so the instrumentation, the failure path and the folder layout are tested on every commit for free. (Note, section 6.)

[test_runlog.py](../tests/test_runlog.py):

- The fake ([lines 27-65](../tests/test_runlog.py#L27-L65)) is a handful of `SimpleNamespace` objects shaped like SDK responses: `content` blocks with a `.type`, a `usage` with the four token fields, `model`, `stop_reason`, `_request_id`. `FakeMessages.create` pops a scripted response per call and raises when the script runs out, which is how the failure test triggers an error on the second call.
- `monkeypatch.setattr(agent, "get_client", lambda: client)` swaps the client in; the loop code under test is the real one.
- `test_run_agent_records_steps_tools_usage_and_writes_folder` ([line 89](../tests/test_runlog.py#L89)) scripts three API calls (two tool calls, then one, then the report) against real CSVs in `tmp_path`, so the tools genuinely run. It then asserts the step and tool records, the token sums including a cache read, the four files, the recovered input metadata with row counts and differing digests, the trace nesting, and that a fake model is unpriced.
- `test_failed_run_is_logged_then_reraised` ([line 136](../tests/test_runlog.py#L136)): the first response asks for a tool on a file that does not exist (the tool errors, `is_error` is recorded), the second call raises; the test checks the exception propagates *and* the folder exists with `status: "error"`.
- `test_log_dir_none_writes_nothing`, `test_max_turns_is_recorded`, the two pricing tests, and `test_run_to_dict_and_summarizer`, which writes a hand-built run and reads it back through the summariser.

Pedagogical point: none of these tests needs a key or spends money, and together they cover every branch of the `try / except / finally`.

## 7. What is committed

[.gitignore:12-15](../.gitignore#L12-L15):

```
runs/*
!runs/README.md
!runs/summarize.py
!runs/example/
```

A detail that trips people up: `runs/` would ignore the directory itself and git could not re-include anything under it; `runs/*` ignores the *entries*, which can then be negated one by one. `runs/example/` is a real run on the bundled sample, moved there by hand; `make example` produces a fresh one to move over it.

## 8. If you have ten minutes to explain the whole thing

1. Open [agent.py:321](../src/agent.py#L321). "Every API call becomes one of these. Latency, tokens, request id."
2. Open [agent.py:374](../src/agent.py#L374). "`finally`: the folder is written even when the run fails."
3. Open [runlog.py:63](../src/runlog.py#L63). "The inputs come from the tool calls, hashed, not copied."
4. Open [pricing.py:74](../src/pricing.py#L74). "Tokens times a dated list price. Unknown model, unpriced."
5. Open [runs/example/trace.json](../runs/example/trace.json). "Three tool calls under one API call: that is parallel tool use, and the second call is the expensive one."
6. Run `make runs`. "One row per run. Percentiles come when there are rows to compute them from."
7. Open [test_runlog.py:50](../tests/test_runlog.py#L50). "A fake client, so all of this is tested for free."

## 9. Exercises

1. Run the agent twice on the same files and `diff` the two `meta.json` files. Which fields differ, which do not, and which *should* never differ?
2. Add a `p50 / p95` latency line to `summarize.py` that appears only when there are at least ten runs. Decide what to print below ten and why.
3. Point `PRICING_JSON` at a table that prices `claude-sonnet-4-6` at zero and watch the example's cost. Then argue whether a zero price should count as "priced".
4. Add a redaction hook to `input_metadata` that replaces column names with their index when an environment variable is set, and write the test first.
5. Make the first API call in the fake client return `stop_reason: "max_tokens"` and check what the folder says. Is that a model failure or a plumbing failure, and where in the eval harness is that distinction made?

## 10. Gotchas met while building

- `response._request_id` has an underscore and is still the documented public way to read the request id.
- `cache_read_input_tokens` and `cache_creation_input_tokens` can be `None`, not `0`, on some responses.
- The folder must be written in `finally`, but `finished_at` and `latency_s` must be set *before* `write_run` inside that same block, or the error folder reports zero latency.
- A cost total computed from floats prints as `0.028943999999999998`; round the components and the total at the source, not in the table.
- `runs/` in `.gitignore` cannot be partially negated; `runs/*` can.
- The eval runner should not log folders: it would write 54 folders per eval and duplicate its transcripts. Share the serialiser, not the destination.
