# Changelog

## Phase 2 — run logging (2026-10-19)

### Added
- `src/runlog.py`: every `run_agent` call writes `runs/<timestamp>_<id>/` with
  `meta.json` (model requested and served, prompt, input files with size, sha256,
  rows and columns, target column, status, timings, git revision), `trace.json`
  (one entry per API step with latency, tokens, request id and the tool calls it
  triggered, each with its own latency), `usage.json` (token totals, API versus tool
  latency, estimated cost with its pricing date) and `report.md`. A failed run is
  logged too, with its error.
- `src/pricing.py`: list-price table dated 2026-10-06 and `estimate_cost`; an
  unknown model is reported as unpriced, never guessed. `PRICING_JSON` overrides.
- `runs/summarize.py` (`make runs`): one row per run folder, standalone.
- `runs/example/`: one real run kept as a reference; the rest of `runs/` is ignored.
- `tests/test_runlog.py`: offline tests with a fake API client covering the loop
  instrumentation, the folder layout, cost estimation and the summarizer.
- `notes/run_logging.md` (design, figures, references) and
  `notes/run_logging_code_walkthrough.md` (file-by-file); figures from
  `notes/figures/make_runlog_figures.py`, drawn from `runs/example`.

### Changed
- `AgentRun` gains `steps` (per-API-call latency, tokens, cache tokens, request id,
  stop reason), `run_id`, `started_at`, `finished_at`, `latency_s`, `error` and
  `log_dir`; `ToolCall` gains `latency_s` and `turn`; `Usage` tracks cache tokens.
- `run_agent(..., log_dir="runs")`; pass `log_dir=None` to opt out. The eval runner
  opts out and instead records `steps` and `cost` in each transcript; each eval
  summary carries an estimated agent cost.


## Phase 1 — agent eval harness and regression suite (2026-10-18)

Rebuilt from the 2026-10-15 design note after the original implementation was lost
unpushed. Design unchanged; data, numbers and figures regenerated.

### Added
- `evals/`: six golden cases generated from a known process (`generate_cases.py`),
  case loader with neutral-path copies (`cases.py`), three mechanical checkers
  (tool sequence, numeric faithfulness, verdict) and an LLM judge with a hashed
  rubric (`checks/`), a parallel runner that writes transcripts, plumbing errors and
  a summary per model (`run_eval.py`, `--rejudge` to re-score saved transcripts), and
  a cross-run comparison table (`compare.py`).
- `regression/`: ten deterministic pytest files, each named after a failure and
  documenting its provenance; an oracle report that must pass every checker; three
  record-and-replay fixtures (one recorded live on 2026-10-18, two reconstructed from
  the lost build's note); a template for adding a new failure in under five minutes.
- `tests/test_tools.py`: known-answer unit tests for the three tools.
- `notes/eval_harness.md`: the design note, rewritten against this code, with ten
  generated figures (`notes/figures/make_figures.py`) and a reference per concept.
- `Makefile` targets: `test`, `unit`, `regression`, `eval`, `eval-nojudge`, `compare`,
  `cases`, `fixtures`, `figures`, `typecheck`, `lint`.
- `.gitignore`, pytest and ruff configuration in `pyproject.toml`.

### Changed
- `run_agent` returns an `AgentRun` (report text, tool trace with results and error
  flags, token usage, stop reason, requested and served model, prompts, raw messages)
  and takes `model` and `max_tokens` keyword arguments; `str(run)` is the report.
- `save_report` accepts a string or an `AgentRun`, so the README's two-function API
  still works.
- The API client is created on first use; importing `agent` needs no key.
- `max_tokens` 2048 → 16000. Tool errors are returned with `is_error` set.
- System prompt states the thresholds and verdict rules, forbids new figures and new
  thresholds, forbids blaming a column without evidence, and asks for a trailing JSON
  verdict block. The JS tool description states the metric's 0–1 range.

### Fixed
- `js_divergence_categorical` reindexed on the real file's levels only, so a level
  present only in the synthetic file was dropped and the distance could be exactly
  0.0 for a visibly different column. It now uses the union of levels with a zero
  count for a missing one.

### Measured
See `notes/eval_harness.md`, section 10, for the three-model run of 2026-10-18.
