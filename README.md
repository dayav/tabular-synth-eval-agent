# Synthetic Data Evaluation Agent

An LLM agent that evaluates the quality of synthetic tabular data against a real reference dataset. Given two CSV files, the agent autonomously selects and runs fidelity and utility evaluations, then writes a plain-English assessment and saves it as a Markdown report.

It is built on the evaluation methodology from my master's thesis on privacy-preserving synthetic tabular data ([paper](https://arxiv.org/abs/2602.06390) · [thesis pipeline](https://github.com/dayav/pipeline_synthetic_tabular_data)) and is powered by Claude through the Anthropic SDK's native tool use.

> **Status:** MVP agent plus an evaluation harness and regression suite for the agent itself (v2 phase 1). Not deployed; no traffic beyond the author's own runs.

## How it works

You give the agent the path to a real dataset and the path to a synthetic version of it. The agent then calls three evaluation tools and synthesizes the results into a readable summary. It currently covers two dimensions of quality:

### Univariate fidelity — does each column *look* like the real thing?

- **Categorical columns — Jensen–Shannon distance.** Measures how closely each synthetic categorical column's distribution matches the real one. `0` means identical, larger values mean more divergence (typically `< 0.1` is good).
- **Numerical columns — Cohen's *d*.** A standardized effect size for the difference in means between the real and synthetic values of each numerical column. Smaller is better (`< 0.2` is a small effect, indicating good fidelity).

### Utility — is the synthetic data *useful* for a downstream model?

- **Train on Synthetic, Test on Real (TSTR).** An XGBoost classifier is trained on the synthetic data and evaluated on held-out real data. Its accuracy is compared against a **Train on Real, Test on Real (TRTR)** baseline. The resulting **utility gap** (`TRTR − TSTR`) quantifies how much predictive value, if any, is lost by training on synthetic data instead of real data.

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
| `POST /evaluate` | multipart upload of a real and a synthetic CSV plus `target_column` (and optional `model`); runs the agent and returns the run id, parsed verdict, flagged columns, usage, estimated cost and the report |
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

The multi-stage [Dockerfile](Dockerfile) installs the pinned [requirements.lock](requirements.lock) in a builder stage and ships a slim, non-root runtime with a health check. [CI](.github/workflows/ci.yml) runs lint, types, unit tests, the regression suite and the API tests on every PR; on `main` it also runs a cheap live eval (Haiku, one run per case), builds and publishes the image to GHCR, and, if the Azure variables are set, rolls the public demo deployment. The exact Azure setup is in [docs/deploy.md](docs/deploy.md); the design, with figures and references, is in [notes/service_and_deployment.md](notes/service_and_deployment.md) and the code walkthrough in [notes/service_and_deployment_code_walkthrough.md](notes/service_and_deployment_code_walkthrough.md).

**Deployment status:** the image builds and serves locally; the Azure Container Apps demo is documented and wired into CI but **not yet live**. When it is, the URL goes here and nowhere else, labelled "public demo deployment".

## Evaluating the agent

An agent that grades data but is never itself graded is not much of a tool, so the repository ships an evaluation harness for the agent and a regression suite that runs on every commit. The design is explained from first principles, with figures and references, in [notes/eval_harness.md](notes/eval_harness.md).

In short: six real/synthetic pairs are built so that the right answer is known by construction (an exact copy, a clean re-draw, a shuffled column, a leaked label, a column scaled ×10, a missing category level). `make eval` runs the agent on each of them N times and checks three things mechanically: did it call the right tools, is every number in its report traceable to a tool result or the prompt, and is its verdict right. A second model scores clarity against a fixed rubric. `make regression` replays recorded failures through the checkers in a few seconds, without calling the model.

```bash
make test                                   # unit tests for the tools + regression suite, no API key needed
make eval MODEL=claude-sonnet-5-5 RUNS=3    # live eval, judge defaults to claude-opus-5-5
make eval-nojudge MODEL=claude-haiku-5-5    # mechanical checks only
make compare                                # one row per saved results folder
```

Measured on 2026-10-18, six cases × three runs per model, judge `claude-opus-5-5`, no run lost to API errors or truncation:

| model | tool-selection | numeric faithfulness | runs with no invented number | verdict | clarity (1–5, judge) |
|---|---|---|---|---|---|
| claude-sonnet-4-6 | 100% (18/18) | 99.8% (404/405) | 94% (17/18) | 100% (18/18) | 3.96 |
| claude-sonnet-5-5 | 100% (18/18) | 100% (299/299) | 100% (18/18) | 100% (18/18) | 4.74 |
| claude-haiku-5-5 | 100% (18/18) | 100% (275/275) | 100% (18/18) | 100% (18/18) | 4.64 |

The one failure: Sonnet 4.6 wrote that a Cohen's d of 4.909 was "more than 24 times the flagging threshold", a ratio it computed itself; that transcript is now a regression fixture. At eighteen runs the noise floor on a pass rate is about ±24 points, so the faithfulness difference is one observation, not a result; the clarity gap is the only difference that clears the floor.

Read these as measured on six synthetic cases with thresholds given in the prompt; they support "the harness measures tool selection, numeric faithfulness and verdict correctness across model versions and catches subtle failures", not fine-grained quality claims. The judge is uncalibrated against human ratings. Details, the failures found, and the noise floor are in the note.

## Project structure

```
src/
  agent.py       # Agent loop: sends the prompt to Claude, dispatches tool calls, returns an AgentRun
  tools.py       # The three evaluation tools (JS distance, Cohen's d, TSTR/XGBoost)
  report.py      # Writes the agent's assessment to a Markdown report
  api.py         # FastAPI service: POST /evaluate, GET /runs, GET /runs/{id}, GET /health
  runlog.py      # Per-run folder writer; pricing.py holds the dated list prices
evals/           # Golden cases, checkers, LLM judge, eval runner (make eval)
runs/            # Per-run logs: trace, latency, tokens, cost (make runs); one example committed
regression/      # Deterministic pytest suite replaying recorded failures (make regression)
tests/           # Known-answer unit tests for the tools
examples/        # Sample real and synthetic CSVs
notes/           # Scope, the eval-harness and run-logging notes, generated figures
docs/deploy.md   # Container and Azure Container Apps setup
Dockerfile, requirements.lock, .github/workflows/ci.yml
CHANGELOG.md
Makefile
```

## Roadmap

v2, in order (phases 1 to 3 are done, bar the live Azure step):

1. ~~Agent eval harness and regression suite~~
2. ~~Run logging: per-run folder with trace, latency, tokens, cost~~
3. ~~FastAPI service, Dockerfile, CI with the regression suite as a gate~~; the Azure demo deployment is documented and wired, not yet live
4. Privacy tool (HEOM-based identifiability, from the thesis pipeline)
5. Retrieval over metric documentation so thresholds come from the docs
6. An LLM tabular-generation baseline evaluated by the same agent

Out of scope: multivariate fidelity (correlation matrices, joint distributions), framework-based orchestration (e.g. LangGraph), a web UI.

## License

See [LICENSE](LICENSE).
