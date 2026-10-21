# Changelog

## Phase 6 — LLM generation baseline (2026-10-21)

### Added
- `src/generate.py`: `generate_with_llm(real_csv_path, output_csv_path, n_rows, target_column, model, batch_size)`.
  Summarises the real file into a column schema with marginal statistics only (category
  levels and frequencies, numeric min/max/mean/std/quantiles, low-cardinality integers
  treated as categorical), asks Claude for rows as strict JSON through structured output
  (`output_config.format`), validates every row (known level, numeric type, integer-ness,
  range with 5% slack), drops the rest, requests more batches up to a budget, writes a CSV
  and returns counts, rejection reasons, tokens, estimated cost and latency. No real row is
  ever sent to the model. It is not registered in the evaluation agent's tools: an
  evaluator must not generate what it grades.
- `scripts/fetch_generator_outputs.py`: downloads the thesis datasets (adult, cardio,
  credit; train and test) and the CTGAN/TVAE baseline outputs from the public Hugging Face
  dataset repo, converting the HDF5 tables to CSV under `data/generators/` (git-ignored).
- `scripts/compare_generators.py`: for each dataset draws one real reference sample and
  one held-out test sample, cuts each generator's output to the same number of rows
  (CTGAN and TVAE from the pipeline, the LLM generated from the reference sample's schema
  and cached under `results/generators/`), scores all three with the four tools against
  the same files, asks the evaluation agent for its verdict, and writes
  `results/generators/comparison.{json,md}` plus the README table between markers.
- `tests/test_generate.py` (schema summary, strict JSON schema, row validation, batching
  with a scripted client, attempt budget) and `tests/test_compare_generators.py`.
- `make fetch-generators`, `make compare-generators`; `tables` in the dev extra.
- `notes/llm_generation_baseline.md` (design, measured figures, references) and
  `notes/llm_generation_baseline_code_walkthrough.md` (file-by-file); figures drawn from
  `results/generators/comparison.json` by `notes/figures/make_generation_figures.py`.

### Gotchas
- Structured output accepts no `minItems`/`maxItems` beyond 0/1 on arrays; the batch size
  is requested in the prompt and enforced by trimming and validation.

### Measured (2026-10-21, 500 rows per generator, adult / cardio / credit)
- Generation: 500 valid rows per dataset in 20 to 22 batches of 25; 1 row rejected in
  1,500 (a credit payment amount out of range); about 0.90 / 0.52 / 1.62 USD at list
  prices, 4 to 12 minutes per dataset.
- Table in README and `results/generators/comparison.md`. In one line: on the same
  footing the marginals-only LLM baseline lands in the same band as CTGAN and TVAE on
  fidelity and utility (utility gaps within ±0.04 for every generator), flags fewer or
  comparable columns than CTGAN, more than TVAE on the numeric-heavy datasets, has zero
  exact matches by construction, and identifiability of 0.35 to 0.39. The agent's
  verdicts and flagged columns agreed with the mechanical flags on all nine outputs.
- Three public datasets, one draw each, a few hundred rows: a baseline, not a claim
  about LLMs and tabular data.


## Phase 5 — retrieval over metric documentation (2026-10-21)

### Added
- `docs/metrics/`: one page per metric (`js_distance`, `cohens_d`, `utility_gap`,
  `identifiability`) plus `verdict_rules`, each with YAML front matter holding the
  machine-readable threshold and five fixed sections: Definition, Threshold,
  Interpretation, Caveats, Citation (every page cites arXiv:2602.06390).
- `src/guidance.py`: parses the pages, exposes `thresholds()`, and builds a small local
  index over the sections (TF-IDF, word 1–2 grams plus character 3–5 grams, cosine) with
  `lookup(query, top_k, metric, section)`. Lexical on purpose: five pages, exact technical
  vocabulary, builds in milliseconds, runs offline.
- Agent: fifth tool `lookup_metric_guidance`. The system prompt no longer contains any
  threshold; it instructs the model to look up every threshold and reference level and
  name the source section. `agent.py` and `evals/checks/verdict.py` both read their
  threshold constants from the docs' front matter, so there is one source of truth.
- `evals/checks/tool_sequence.py`: `lookup_metric_guidance` is a repeatable tool (at least
  once, any number of times); all golden cases now expect it.
- Regression: `test_thresholds_come_from_docs.py` pins that docs, agent and checker agree,
  that the prompt carries no threshold number, that a report quoting a threshold without a
  lookup result fails numeric faithfulness and passes with it, that at least one lookup is
  required, and that each metric's threshold query lands on its Threshold section. The
  oracle report now makes one lookup per metric; pre-phase-5 fixtures are replayed with the
  guidance results appended (`replay_results`), since they were recorded when the prompt
  carried the thresholds.
- `tests/test_guidance.py`: page structure, thresholds, top hits for eight queries,
  filters, and an index built from another folder.

### Changed
- The allowed set of the numeric-faithfulness checker now includes retrieved guidance,
  which is the definition of a grounded threshold. Numbers in the docs are therefore
  chosen with care (the JS page says "roughly 0.29", never a bare "0.3").

### Measured (2026-10-21, all eight cases, one run each, Opus judge)
- claude-haiku-5-5 and claude-sonnet-5-5: 8/8 on all three checks. Every run made one
  lookup per metric (3 to 5 calls), quoted the documented thresholds, and produced the
  expected verdict; no ungrounded number in 294 quoted figures. Clarity 4.38 and 4.83.


## Phase 4 — privacy tool (2026-10-21)

### Added
- `identifiability_heom` in `src/tools.py`, ported from the thesis pipeline
  (`privacy_sampling/heom.py`, `get_epsilon_heom_any`): HEOM encoding (numerics min-max
  scaled on the real data, categoricals one-hot scaled by 1/√2 so a mismatch counts 1),
  then nearest neighbours. Returns `identifiability` (fraction of real records whose
  nearest synthetic record is closer than their nearest other real record, on a
  size-matched synthetic subsample; honest sample ≈ 0.5, copy = 1.0), the thesis
  `epsilon_any` (fraction of synthetic rows inside some real record's k-NN ball),
  `exact_matches` / `exact_match_fraction`, median distances, the column split and the
  rows dropped for missing values. `k` and `exclude_columns` are parameters.
- Agent: fourth tool in `TOOLS` and `TOOL_REGISTRY`; the system prompt calls it only when
  the request asks about privacy, states the thresholds (identifiability > 0.7 or exact
  matches > 5% is poor; `epsilon_any` is never thresholded), adds a `privacy` key to the
  verdict block when assessed, and makes overall poor when utility or privacy is poor.
- Golden cases `identical_privacy` (copy, must be poor overall) and `clean_privacy`
  (honest control, must stay excellent); cases carry `ask_privacy` and the prompt gains a
  privacy sentence. `derive_expected` takes the privacy result; `check_verdict` fails a
  privacy axis that was not asked for and a missing one that was.
- Regression: oracle report with a privacy section; tests for the privacy axis matching
  the request and for the privacy tool being unexpected when not asked.
- Unit tests for the tool: exact copy maximal, honest sample near 0.5, far sample zero,
  HEOM distance units, `k`, `exclude_columns`, missing values, size-matched subsample.
- API: `assess_privacy` form field on `POST /evaluate`; `Verdict.privacy`.
- `notes/privacy_tool.md` (design, calibration figures, references) and
  `notes/privacy_tool_code_walkthrough.md` (file-by-file); figures computed from the golden
  data by `notes/figures/make_privacy_figures.py`.

### Measured (design calibration, 2026-10-21, 2000-row golden data)
- identifiability: copy 1.00, honest re-draw 0.49, copy + 0.2 sd jitter 0.83,
  copy + 0.1 sd jitter 0.96, 50% rows copied 0.66 (caught by exact matches 0.50).
- epsilon_any on the honest re-draw is 0.67: the ANY rule's honest level is high, which
  is why the agent is told not to threshold it.
- Live check (2026-10-21, `identical_privacy`, `clean_privacy`, `clean_synthetic`, two runs
  each, Opus judge): claude-haiku-5-5 and claude-sonnet-5-5 both 12/12 on all three
  checks. Both called the privacy tool exactly when asked and never otherwise, called the
  copy poor overall, and kept the honest control excellent. Clarity 4.50 and 4.67.


## Phase 3 — API, container, CI and deployment path (2026-10-20)

### Added
- `src/api.py`: FastAPI service. `POST /evaluate` (two CSV uploads + target column,
  optional model from an allowlist) validates the inputs (CSV only, size and row limits,
  identical column sets, target with two or more classes), saves them under neutral names,
  runs the agent with run logging on, and returns run id, parsed verdict, flagged columns,
  usage, estimated cost and report. `GET /runs`, `GET /runs/{id}` (`?include_trace=true`)
  read the run folders; `GET /health` reports version, models and whether a key is set.
  Concurrency is capped (503 when busy); upstream API errors map to 502/503.
- `Dockerfile` (multi-stage, `python:3.12-slim` + `libgomp1`, non-root, `HEALTHCHECK`),
  `.dockerignore`, `requirements.lock` (exact versions, `make lock` to refresh).
- `.github/workflows/ci.yml`: `test` (ruff, mypy, unit, regression, API tests) on every PR
  and push; `evals` on `main` with Haiku and one run per case, or on manual trigger with a
  chosen model; `docker` builds, pushes to GHCR and smoke-tests the image; `deploy` rolls
  an Azure Container App via OIDC when `vars.AZURE_DEPLOY` is `true`.
- `docs/deploy.md`: local container run, what CI does, the exact `az` commands for the
  Container Apps demo, optional Azure Files persistence for run logs, OIDC setup, the
  secrets and variables GitHub needs, tear-down, and the honest status.
- `tests/test_api.py` (real app, real loop, fake model) and `tests/fake_client.py`
  (shared scripted client, now also used by the run-logging tests).
- `make api`, `make docker-build`, `make docker-run`, `make lock`.
- `notes/service_and_deployment.md` (design, figures, references) and
  `notes/service_and_deployment_code_walkthrough.md` (file-by-file); diagrams from
  `notes/figures/make_service_figures.py`.

### Changed
- `pyproject.toml`: version 0.3.0; `fastapi`, `uvicorn`, `python-multipart`, `pydantic`
  are runtime dependencies; `httpx` joins the dev extra for the test client.

### Not done
- The Azure resources are not created (no `az` login in the build environment). The
  README says so and claims no URL.


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
