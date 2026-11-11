# Synthetic Data Evaluation Agent

An LLM agent that evaluates synthetic tabular data against the real data it imitates, and an evaluation harness that grades the agent itself.

You give the agent a real CSV, a synthetic CSV and a target column. Claude, through the Anthropic SDK's native tool use, chooses among five typed Python tools, looks up every threshold in the metric documentation, and writes a report with a verdict on fidelity, utility and, on request, privacy. The harness then runs the agent on eight real/synthetic pairs whose right answer is known by construction, and checks its tool calls, its numbers and its verdict mechanically.

The metrics and the method come from my master's thesis on privacy-preserving synthetic tabular data ([paper](https://arxiv.org/abs/2602.06390) · [thesis pipeline](https://github.com/dayav/pipeline_synthetic_tabular_data)).

> **Status, 2026-11-11.** Built and on `main`: the agent, the eval harness, the regression suite, run logging, the HTTP service, the container image (published to GHCR by CI) and the generator comparison. Not done: the Azure demo deployment is documented and wired into CI but has never been run, so nothing is deployed and there is no traffic beyond my own runs.

## Results

### The agent, graded

Each trial is one agent run on one golden case. It is checked mechanically on three points:

- **Right tools.** The agent called the metric tools the case needs, once each, on the files it was given, and the privacy tool only when the request asked about privacy.
- **Numbers traceable.** Every number in the report can be traced to a tool result (the retrieved guidance included), a tool description or the prompt. A figure the model computed itself fails the trial.
- **Right verdict.** The verdict per axis, the overall verdict and the flagged columns match the answer key.

A second model (`claude-opus-5-5`) scores clarity from 1 to 5 against a fixed rubric. It never decides a pass.

<!-- EVAL TABLE START -->
7 sessions, 82 scored trials, last session 2026-10-21. A trial is one agent run on one golden case. 0 trials were lost to API errors, truncation or refusals.

| session (UTC) | agent under test | model | cases × runs | right tools | numbers traceable to a tool result | trials with no invented number | right verdict | all three checks | clarity (judge, 1–5) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-18 | 3 tools, thresholds in the prompt | claude-haiku-5-5 | 6 × 3 | 100% (18/18) | 100% (275/275) | 100% (18/18) | 100% (18/18) | 100% (18/18) | 4.64 † |
| 2026-10-18 | 3 tools, thresholds in the prompt | claude-sonnet-4-6 | 6 × 3 | 100% (18/18) | 99.8% (404/405) | 94.4% (17/18) | 100% (18/18) | 94.4% (17/18) | 4.21 |
| 2026-10-18 | 3 tools, thresholds in the prompt | claude-sonnet-5-5 | 6 × 3 | 100% (18/18) | 100% (299/299) | 100% (18/18) | 100% (18/18) | 100% (18/18) | 4.74 † |
| 2026-10-21 | 4 tools (privacy added), thresholds in the prompt | claude-haiku-5-5 | 3 × 2 | 100% (6/6) | 100% (106/106) | 100% (6/6) | 100% (6/6) | 100% (6/6) | 4.50 |
| 2026-10-21 | 4 tools (privacy added), thresholds in the prompt | claude-sonnet-5-5 | 3 × 2 | 100% (6/6) | 100% (123/123) | 100% (6/6) | 100% (6/6) | 100% (6/6) | 4.67 |
| 2026-10-21 | 5 tools, thresholds retrieved from the docs | claude-haiku-5-5 | 8 × 1 | 100% (8/8) | 100% (151/151) | 100% (8/8) | 100% (8/8) | 100% (8/8) | 4.38 |
| 2026-10-21 | 5 tools, thresholds retrieved from the docs | claude-sonnet-5-5 | 8 × 1 | 100% (8/8) | 100% (143/143) | 100% (8/8) | 100% (8/8) | 100% (8/8) | 4.83 |
| **all sessions** | | | **82 trials** | 100% (82/82) | 99.9% (1501/1502) | 98.8% (81/82) | 100% (82/82) | 98.8% (81/82) | |

Trials by agent version: 3 tools, thresholds in the prompt: 54; 4 tools (privacy added), thresholds in the prompt: 12; 5 tools, thresholds retrieved from the docs: 16.

Noise floor on a pass rate (1/√n): about ±24 points at 18 trials, about ±35 points at 8 trials, about ±41 points at 6 trials.

† Scored under the first judge prompt, which did not show the judge the agent's instructions; not comparable with the undaggered rows.
<!-- EVAL TABLE END -->

How to read the table:

- **One trial failed.** `claude-sonnet-4-6` on the `age_x10` case quoted a Cohen's d of 4.909 correctly, then added that it was "more than 24 times the flagging threshold", a ratio it computed itself. The verdict was right, so a check on the final answer alone would have passed it. That transcript is now a regression fixture.
- **The table does not rank the models on correctness.** Every difference between rows is inside the noise floor. What it supports is narrower: the harness measures tool selection, numeric faithfulness and verdict correctness across model versions and agent versions, and it catches a subtle failure.
- **Most trials measured an earlier agent.** The session with three runs per case (2026-10-18) ran the first agent: three tools, thresholds written in the prompt, six cases. The agent as it is today, with five tools, thresholds retrieved from the documentation and eight cases, has been run once per case on two models.
- **Apart from that one trial, every model passes every case every time.** The suite therefore works as a regression check on the agent. It is not hard enough to separate one model from another.
- **Clarity supports a ranking at most.** The judge has not been calibrated against human ratings.

Results per golden case and the source folders are in [results/agent_evals.md](results/agent_evals.md). What this evaluation still lacks, with the evidence for each gap, is in [notes/agent_evals_gap_analysis.md](notes/agent_evals_gap_analysis.md).

### Three generators, one evaluator

The same four metric tools and the same agent scored three generators on three public datasets: CTGAN and TVAE outputs from the thesis pipeline, and an LLM baseline that saw only the column schema and marginal statistics. How the baseline works is [further down](#llm-generation-baseline).

<!-- GENERATOR TABLE START -->
Measured 2026-10-21: 3 datasets, 500 rows per generator, same real reference sample and the same held-out test rows for every generator. CTGAN and TVAE outputs are the thesis pipeline baselines (trained on the full training sets); the LLM (claude-sonnet-5-5) saw only the schema and marginal statistics of the reference sample. Agent verdicts by claude-sonnet-5-5.

| dataset | generator | JS mean / max (flagged) | Cohen's d mean / max (flagged) | TSTR / TRTR | utility gap | identifiability | exact matches | agent overall |
|---|---|---|---|---|---|---|---|---|
| adult | CTGAN | 0.124 / 0.265 (6/9) | 0.086 / 0.197 (0/6) | 0.825 / 0.835 | +0.010 | 0.36 | 0.0% | acceptable |
| adult | TVAE | 0.093 / 0.175 (5/9) | 0.077 / 0.166 (0/6) | 0.799 / 0.835 | +0.035 | 0.39 | 0.0% | acceptable |
| adult | LLM | 0.103 / 0.237 (4/9) | 0.026 / 0.077 (0/6) | 0.816 / 0.835 | +0.019 | 0.39 | 0.0% | acceptable |
| cardio | CTGAN | - / - (0/0) | 0.094 / 0.343 (2/11) | 0.647 / 0.684 | +0.037 | 0.51 | 0.0% | acceptable |
| cardio | TVAE | - / - (0/0) | 0.056 / 0.122 (0/11) | 0.664 / 0.684 | +0.020 | 0.47 | 0.0% | excellent |
| cardio | LLM | - / - (0/0) | 0.113 / 0.212 (1/11) | 0.705 / 0.684 | -0.020 | 0.35 | 0.0% | acceptable |
| credit | CTGAN | - / - (0/0) | 0.210 / 0.533 (9/23) | 0.791 / 0.798 | +0.007 | 0.36 | 0.0% | acceptable |
| credit | TVAE | - / - (0/0) | 0.077 / 0.186 (0/23) | 0.801 / 0.798 | -0.003 | 0.43 | 0.0% | excellent |
| credit | LLM | - / - (0/0) | 0.128 / 0.259 (6/23) | 0.788 / 0.798 | +0.010 | 0.38 | 0.0% | acceptable |

Read with care: three public datasets, a few hundred rows per generator, one draw each; this is a baseline, not a claim about LLMs and tabular data. Thresholds: JS above 0.1 and Cohen's d above 0.2 per column are flagged; utility gap above 0.05 is poor; identifiability above 0.7 or exact matches above 5% is poor. The LLM cannot copy rows it never saw, so its identifiability is a measure of how closely marginals alone place points near real records.
<!-- GENERATOR TABLE END -->

## What is in the repository

| piece | where | check it yourself |
|---|---|---|
| Agent loop and five tools | [src/agent.py](src/agent.py), [src/tools.py](src/tools.py), [src/guidance.py](src/guidance.py) | `python src/agent.py` (needs an API key) |
| Metric pages the agent retrieves its thresholds from | [docs/metrics/](docs/metrics/) | |
| Eval harness: eight golden cases, three checks, an LLM judge | [evals/](evals/) | `make eval` (needs an API key) |
| Regression suite: 55 tests, no model call, a few seconds | [regression/](regression/) | `make regression` |
| Unit tests: 58, on the tools, the API, the run log, retrieval and generation | [tests/](tests/) | `make unit` |
| Run logging: trace, latency, tokens, estimated cost per run | [src/runlog.py](src/runlog.py), [runs/](runs/) | `make runs` |
| HTTP service | [src/api.py](src/api.py) | `make api` |
| Container image and CI | [Dockerfile](Dockerfile), [.github/workflows/ci.yml](.github/workflows/ci.yml) | `docker pull ghcr.io/dayav/tabular-synth-eval-agent:latest` |
| LLM generation baseline and the generator comparison | [src/generate.py](src/generate.py), [scripts/](scripts/) | `make compare-generators` (needs an API key) |
| Result tables | [results/](results/) | `make results` |
| Design notes with figures and references | [notes/](notes/) | |

Test counts are as of 2026-11-11.

## How the agent works

The agent has five tools. Three always run, one runs on request, and one supplies the thresholds.

| tool | what it returns | when the agent calls it |
|---|---|---|
| `js_divergence_categorical` | Jensen-Shannon distance between the real and synthetic distribution of each categorical column. `0` is identical, `1` is maximally different. | every evaluation |
| `cohen_d_numerical` | Cohen's *d* for each numerical column: the difference in means in units of standard deviation. | every evaluation |
| `tstr_xgboost` | Accuracy of an XGBoost classifier trained on synthetic data and tested on held-out real data (TSTR), the same trained on real data (TRTR), and the utility gap `TRTR − TSTR`. | every evaluation |
| `identifiability_heom` | Nearest-neighbour identifiability in HEOM space: the fraction of real records whose nearest synthetic record is closer than their nearest other real record (about `0.5` for an honest sample of the same size, `1.0` for a copy). Also the thesis filter statistic εANY and the fraction of exact matches. | only when the request asks about privacy or disclosure risk |
| `lookup_metric_guidance` | Sections of the metric pages, found through a small local index (TF-IDF over word and character n-grams, cosine similarity, no neural embeddings, no network). | for every metric it computed, before writing the report |

**Thresholds come from the documentation, not from the prompt.** Every metric has a page under [docs/metrics/](docs/metrics/) with a definition, the threshold, interpretation, caveats and a citation. The agent's prompt carries no threshold: it tells the agent to take every threshold and reference level from the retrieved guidance and to name the source. The eval harness reads its answer keys from the same pages, and a regression test fails if the pages, the agent and the checker disagree, or if a report quotes a threshold the agent never looked up. This is retrieval over metric documentation, not over customer data.

**The verdict is mechanical on purpose, so that it can be checked.** Each axis (categorical fidelity, numerical fidelity, utility and, when assessed, privacy) is good or poor against its documented threshold. Overall is poor when utility or privacy is poor, excellent when every assessed axis is good, and acceptable otherwise. The rules are in [docs/metrics/verdict_rules.md](docs/metrics/verdict_rules.md). The privacy tool's design and calibration are in [notes/privacy_tool.md](notes/privacy_tool.md).

## How the agent is evaluated

Eight real/synthetic pairs are built so that the right answer is known by construction: an exact copy, a clean re-draw, a shuffled column, a leaked label, a column scaled ×10, a missing category level, and the copy and the re-draw again with the prompt asking about privacy. `make eval` runs the agent on each pair N times, applies the three checks described under [Results](#the-agent-graded), and asks the judge for a clarity score.

`make regression` is separate and needs no API key. Its 55 tests replay recorded failures and known edge cases through the same checkers in a few seconds, so a failure that was fixed once stays fixed. Three agent transcripts are replayed: one was recorded live, and two were rebuilt from the recorded failure text after the original transcripts were lost. Each fixture states its provenance.

```bash
make test                                   # unit tests and the regression suite, no API key needed
make eval MODEL=claude-sonnet-5-5 RUNS=3    # live eval, judge defaults to claude-opus-5-5
make eval-nojudge MODEL=claude-haiku-5-5    # mechanical checks only
make results                                # rebuild results/agent_evals.md and the table above from the saved summaries
```

The design is explained from first principles, with figures and references, in [notes/eval_harness.md](notes/eval_harness.md). [notes/agent_evals_explained.md](notes/agent_evals_explained.md) walks through the same harness against Anthropic's article on agent evals.

## Installation

```bash
pip install -r requirements.txt          # the agent and the API
pip install -e ".[dev]"                  # plus pytest, ruff, mypy, matplotlib, httpx for the harness, figures and API tests
```

The agent calls the Anthropic API, so it needs an API key. Copy the example file and fill in your key:

```bash
cp .env.example .env
```

Then edit `.env` and set `ANTHROPIC_API_KEY` to your [Anthropic API key](https://console.anthropic.com/).

## Usage

The simplest way to run an end-to-end evaluation on the bundled sample data:

```bash
python src/agent.py
```

This evaluates `examples/synth_sample.csv` against `examples/real_sample.csv`, prints the assessment, and writes it to `report.md`.

To run it on your own data, use the two-function API directly:

```python
from agent import run_agent
from report import save_report

prompt = (
    "Evaluate the synthetic dataset at path/to/synth.csv "
    "against the real dataset at path/to/real.csv. "
    "The target column for utility evaluation is 'target'."
)

# run_agent returns an AgentRun: the report plus the tool trace, token usage,
# stop reason and the model that served the request. str(run) is the report.
run = run_agent(prompt)                      # run_agent(prompt, model="claude-haiku-5-5") to pick a model
print(run.text)
print([call.name for call in run.tool_calls], run.usage, run.latency_s, run.log_dir)

# save_report writes the report to a timestamped Markdown file (report.md by default)
path = save_report(run)
print(f"Report saved to: {path}")
```

## Example output

Run on the bundled sample data, the agent produces something like this:

> ### Categorical Fidelity (Jensen-Shannon Distance)
>
> | Column | JS Distance | Rating |
> |---|---|---|
> | `gender` | 0.174 | ⚠️ Moderate |
> | `education` | 0.000 | ✅ Perfect |
> | `city` | 0.044 | ✅ Good |
> | **Mean** | **0.073** | ✅ Acceptable |
>
> The overall mean JS distance of 0.073 is just under the 0.1 threshold... `gender` stands out with a score of 0.174 and warrants attention. `education` is perfectly replicated.
>
> *[...continues with numerical fidelity and utility evaluation. Full report at [report.md](report.md).]*

## Run logging

Every run writes a folder under `runs/` with what went in, what happened, and what it cost:

```
runs/20261019_001530_3f9a1c/
  meta.json    model requested and served, prompt, input files (bytes, sha256, rows, columns), status, timings
  trace.json   one entry per API step: latency, tokens, request id, and the tool calls it triggered with their latency
  usage.json   token totals, API vs tool latency, estimated cost and the pricing date it rests on
  report.md    the report
```

```bash
make runs                 # one row per run: model, latency, tokens, estimated cost, verdict
```

The design, with figures and references, is in [notes/run_logging.md](notes/run_logging.md); the code walkthrough is [notes/run_logging_code_walkthrough.md](notes/run_logging_code_walkthrough.md). Pass `log_dir=None` to `run_agent` to skip logging. Cost is an estimate from list prices in `src/pricing.py`, dated in every `usage.json`; a model missing from the table is reported as unpriced rather than guessed. One real run is committed under [runs/example/](runs/example/).

## HTTP service and container

The agent is also served as a small FastAPI app ([src/api.py](src/api.py)):

| route | what |
|---|---|
| `POST /evaluate` | multipart upload of a real and a synthetic CSV plus `target_column` (optional `model`, optional `assess_privacy=true`); runs the agent and returns the run id, parsed verdict, flagged columns, usage, estimated cost and the report |
| `GET /runs` | recent runs from the run log |
| `GET /runs/{id}` | one run's meta, usage and report; `?include_trace=true` adds the step-by-step trace |
| `GET /health` | liveness plus the configuration that matters (default model, allowed models, whether a key is configured, upload limits) |

Uploads are validated before the model is called: CSV only, size and row limits, identical column sets, target present with at least two classes. Files are saved under neutral names so an upload's file name cannot hint the model. Concurrent evaluations are capped (default 2; a 503 says so). The API key is read from the environment only.

```bash
make api                                                # dev server on :8000, interactive docs at /docs
curl -s -F real=@examples/real_sample.csv -F synthetic=@examples/synth_sample.csv \
     -F target_column=target localhost:8000/evaluate | python -m json.tool

make docker-build && make docker-run                    # same service in the container, runs/ mounted
```

The multi-stage [Dockerfile](Dockerfile) installs the pinned [requirements.lock](requirements.lock) in a builder stage and ships a slim, non-root runtime with a health check.

[CI](.github/workflows/ci.yml) runs lint, type checks (mypy strict on `src`), the unit tests and the regression suite on every pull request and every push. On `main` it also builds the image, publishes it to GHCR and smoke-tests it. Two more jobs are wired and do not run today: a live eval (Haiku, one run per case), skipped because the repository has no `ANTHROPIC_API_KEY` secret, and the Azure roll-out, off until `AZURE_DEPLOY` is set.

**Deployment status:** the image builds, is published and serves locally. The Azure Container Apps demo is documented in [docs/deploy.md](docs/deploy.md) and has never been run. When it is live, the URL goes here and nowhere else, labelled "public demo deployment". The design is in [notes/service_and_deployment.md](notes/service_and_deployment.md).

## LLM generation baseline

A `generate_with_llm` function ([src/generate.py](src/generate.py)) asks Claude for rows as strict JSON from a dataset's column schema and marginal statistics only, validates every row and writes a CSV. No real row is sent to the model. It is a baseline next to the thesis pipeline's CTGAN and TVAE outputs, scored by the same four tools and judged by the same agent. It is deliberately not one of the evaluation agent's tools, because an evaluator must not generate what it grades. The measured table is under [Results](#three-generators-one-evaluator); the design and figures are in [notes/llm_generation_baseline.md](notes/llm_generation_baseline.md).

```bash
python scripts/fetch_generator_outputs.py        # thesis datasets + CTGAN/TVAE outputs into data/generators/ (git-ignored)
python scripts/compare_generators.py             # same metrics, same agent; rewrites the table and results/generators/
```

## Known gaps

- **The current agent has few trials.** One run per case on two models. The three-run session predates the privacy tool and the retrieved thresholds.
- **The default model has not been run on the current agent.** `claude-sonnet-4-6` is the default of the agent, the service and the Makefile, and its only session is the first one.
- **The suite is saturated.** Every model passes nearly every trial, so there is no capability eval yet.
- **The judge is uncalibrated**, and two saved clarity scores come from an older judge prompt (marked † in the table).
- **The golden cases come from one constructed process.** On real generator outputs the agent agreed with the mechanical flags nine times out of nine, but that is one run per output, outside the suite.
- **Nothing is deployed**, so there is no production monitoring and no user feedback.

The full list, 31 gaps with evidence and a fix order, is [notes/agent_evals_gap_analysis.md](notes/agent_evals_gap_analysis.md).

Out of scope: multivariate fidelity (correlation matrices, joint distributions), framework-based orchestration (for example LangGraph), a web UI.

## Project structure

```
src/
  agent.py       # Agent loop: sends the prompt to Claude, dispatches tool calls, returns an AgentRun
  tools.py       # The four metric tools (JS distance, Cohen's d, TSTR/XGBoost, HEOM identifiability)
  guidance.py    # Local index over docs/metrics; the lookup_metric_guidance tool and the thresholds
  generate.py    # generate_with_llm: schema + marginals -> strict JSON rows -> CSV (baseline, not an agent tool)
  report.py      # Writes the agent's assessment to a Markdown report
  api.py         # FastAPI service: POST /evaluate, GET /runs, GET /runs/{id}, GET /health
  runlog.py      # Per-run folder writer
  pricing.py     # Dated list prices behind the cost estimates
evals/           # Golden cases, checkers, LLM judge, eval runner, saved session summaries
regression/      # Deterministic pytest suite replaying recorded failures
tests/           # Known-answer unit tests
results/         # agent_evals.md (every eval session) and generators/ (the generator comparison)
runs/            # Per-run logs; one example run committed
scripts/         # fetch_generator_outputs.py, compare_generators.py
docs/metrics/    # One page per metric: definition, threshold, interpretation, caveats, citation
docs/deploy.md   # Container and Azure Container Apps setup
notes/           # Design notes and code walkthroughs, with generated figures
examples/        # Sample real and synthetic CSVs
Dockerfile, requirements.lock, .github/workflows/ci.yml, Makefile, CHANGELOG.md
```

## License

See [LICENSE](LICENSE).
