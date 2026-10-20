# The service and the pipeline, file by file: a code walkthrough

Companion to [service_and_deployment.md](service_and_deployment.md), which explains *why* the service, container and pipeline are shaped as they are. This one explains *how the code does it*, in request order and then build order, with a pointer into the source for every claim, so you can open the files next to someone and talk them through phase 3.

Conventions: `file:line` links point into the repository. Each section opens with the one idea it implements and where the theory lives, then shows the code.

## 0. The map

| What | Where | Theory in the note |
|---|---|---|
| Settings from the environment | [../src/api.py:35-66](../src/api.py#L35-L66) | sections 4, 7 |
| Request and response schemas | [../src/api.py:70-136](../src/api.py#L70-L136) | section 1 |
| Upload reading, parsing, pair validation | [../src/api.py:158-203](../src/api.py#L158-L203) | section 2 |
| The evaluate endpoint | [../src/api.py:285-323](../src/api.py#L285-L323) | section 2 |
| Health and the run routes | [../src/api.py:265-352](../src/api.py#L265-L352) | section 3 |
| The container | [../Dockerfile](../Dockerfile), [../.dockerignore](../.dockerignore), [../requirements.lock](../requirements.lock) | section 4 |
| The pipeline | [../.github/workflows/ci.yml](../.github/workflows/ci.yml) | section 5 |
| The deployment guide | [../docs/deploy.md](../docs/deploy.md) | section 6 |
| Tests with a fake model | [../tests/test_api.py](../tests/test_api.py), [../tests/fake_client.py](../tests/fake_client.py) | section 8 |
| Figures | [figures/make_service_figures.py](figures/make_service_figures.py) | every figure |

One sentence to hold on to: **nothing reaches the model until every cheap check has passed, and nothing secret is ever written down.**

## 1. Settings: everything configurable comes from the environment

[api.py:35-46](../src/api.py#L35-L46). Each limit is read once at import with a default: the runs directory, the upload size in MB, the maximum and minimum row counts, the model allowlist (a comma-separated string), the default model, the concurrency cap and the turn budget. The run id pattern at [line 48](../src/api.py#L48) is the same shape `new_run_id` produces in `agent.py`; the prompt template at [line 49](../src/api.py#L49) is the one the eval harness uses, so the service and the harness ask the agent the same question.

`_version` ([line 54](../src/api.py#L54)) prefers `APP_VERSION`, which the Dockerfile sets at build time, and falls back to the installed package's metadata. The semaphore at [line 66](../src/api.py#L66) is module-level: one per process, which is why the container runs one worker.

Teaching point: tests override these by monkeypatching the module attributes (section 8), not by setting environment variables, because the values were read at import.

## 2. Schemas: the contract is code

[api.py:70-136](../src/api.py#L70-L136). Six Pydantic models. `EvaluateResponse` is the one to read with a learner: run id and status, the model requested and served, stop reason, latency, the tools called, the parsed `Verdict` and flagged columns, `UsageOut`, `CostOut` and the report. Everything the run folder knows that a caller could act on, and nothing that only the operator should see (no prompt, no file paths).

`parse_verdict` ([lines 140-155](../src/api.py#L140-L155)) is the same `raw_decode` scan as `runs/summarize.py`, kept local so `src/` does not depend on `evals/`.

## 3. The upload path, in the order it runs

**Idea.** Cheap checks first; each failure is a specific status and a sentence a human can act on. (Note, section 2 and Figure 4.)

- `_read_limited` ([lines 158-169](../src/api.py#L158-L169)) reads the upload in 1 MiB chunks and raises 413 the moment the running total passes the limit ([line 166](../src/api.py#L166)). Ask the learner why not `len(await upload.read())`: because that reads the whole thing first.
- `_save_csv` ([lines 172-190](../src/api.py#L172-L190)) checks the extension or content type ([line 174](../src/api.py#L174)), rejects an empty body, writes the bytes to the destination, parses with pandas and turns any parse failure into a 400, then enforces the row bounds ([lines 184-186](../src/api.py#L184-L186)) and a minimum of two columns.
- `_validate_pair` ([lines 193-203](../src/api.py#L193-L203)) compares the column sets and reports exactly which columns are missing on which side ([lines 194-196](../src/api.py#L194-L196)), checks the target exists, and that it has at least two classes in both files ([lines 200-202](../src/api.py#L200-L202)). The last check is what keeps XGBoost from failing later with a message nobody could act on.

## 4. The evaluate endpoint

[api.py:285-323](../src/api.py#L285-L323). Read it as a `try / finally` around the whole request:

1. Resolve the model and check the allowlist (400).
2. [Line 301](../src/api.py#L301): `_slots.acquire(blocking=False)`; a `False` is a 503 with the cap in the message. Non-blocking is the choice to explain: a queue would hide overload; a 503 reports it.
3. [Line 303](../src/api.py#L303): a fresh `mkdtemp` per request; [line 306](../src/api.py#L306): the uploads become `real.csv` and `synth.csv` inside it. This is the path-leak lesson from the eval harness applied at the boundary.
4. Save, parse, validate ([line 309](../src/api.py#L309)).
5. [Line 313](../src/api.py#L313): `run_agent(prompt, max_turns=MAX_TURNS, model=chosen, log_dir=RUNS_DIR)`. Logging stays on, so the folder is written even if the next lines raise.
6. [Lines 314-319](../src/api.py#L314-L319): three `except` clauses map upstream failures to 502 (rejected key, API or connection error) and 503 (rate limit). The SDK's typed exception classes make this a chain from most to least specific rather than a string match.
7. [Lines 322-323](../src/api.py#L322-L323), `finally`: remove the temporary directory, release the slot. Both run on every exit path, including a 400 raised in step 4.

The endpoint is a plain `def`, not `async def`. FastAPI runs synchronous endpoints in a thread pool, so one slow evaluation does not block the health check, and the agent code stays synchronous.

## 5. Health and the run routes

- `health` ([lines 265-282](../src/api.py#L265-L282)) tries to create the runs directory and reports whether it is writable, whether a key is present (as a boolean, never the value), the models, the limits and the version. This is the endpoint the Dockerfile's `HEALTHCHECK`, the CI smoke test and the deploy job all poll.
- `list_runs` ([lines 326-334](../src/api.py#L326-L334)) walks `_iter_run_dirs`, which admits only directories whose name matches the run id pattern and that contain a `meta.json` ([line 210](../src/api.py#L210)); `runs/example/` is therefore not listed, by design.
- `get_run` ([lines 337-347](../src/api.py#L337-L347)) matches the id against the pattern *before* joining it to a path ([line 340](../src/api.py#L340)). `..%2F..%2Fetc` is a 404, not a file read. `?include_trace=true` loads `trace.json`.

## 6. The container

[Dockerfile](../Dockerfile), two stages:

- **builder** ([lines 4-9](../Dockerfile#L4-L9)): `python:3.12-slim`, a virtualenv at `/opt/venv`, `pip install -r requirements.lock`. Nothing from this stage ships except the venv.
- **runtime** ([lines 11-32](../Dockerfile#L11-L32)): the same base plus `libgomp1` ([line 14](../Dockerfile#L14)), which the XGBoost wheel links against and which the slim image lacks (the first thing that fails if you forget it is the first tool call, inside the container, with a message about `libgomp.so.1`). An `app` user with uid 1000 ([line 15](../Dockerfile#L15)); the venv copied from the builder ([line 17](../Dockerfile#L17)); only `src/` copied ([line 18](../Dockerfile#L18)); `PYTHONPATH=/app/src` ([line 20](../Dockerfile#L20)) because the modules import each other by top-level name; `RUNS_DIR=/data/runs` owned by `app`; `USER app` ([line 27](../Dockerfile#L27)); a `HEALTHCHECK` that polls `/health` with a 20 s start period ([line 29](../Dockerfile#L29)); and `CMD uvicorn api:app` with one worker ([line 32](../Dockerfile#L32)).

[.dockerignore](../.dockerignore) keeps `.env`, tests, notes, docs, results and runs out of the build context; `README.md` and `.env.example` are allowed through. [requirements.lock](../requirements.lock) is the resolved set from the development environment, refreshed with `make lock`; `pyproject.toml` still declares the intent with lower bounds.

Teaching point: run `docker image history` on the built image with the learner and show that the key is in no layer and `pip` is in no runtime layer.

## 7. The pipeline

[ci.yml](../.github/workflows/ci.yml). Triggers at the top: every pull request, every push to `main`, and a manual `workflow_dispatch` with three inputs (model, runs, judge).

- `test` ([line 28](../.github/workflows/ci.yml#L28)): Python 3.12 to match the image, install from the lock plus the dev extra, then ruff, mypy, `pytest` and a timed run of the regression suite. No key, no secrets.
- `evals` ([lines 44-74](../.github/workflows/ci.yml#L44-L74)): `needs: test`; runs on `main` or by hand ([line 46](../.github/workflows/ci.yml#L46)). The three `if: env.ANTHROPIC_API_KEY` guards ([lines 58, 66, 69](../.github/workflows/ci.yml#L58)) make a missing secret a warning, not a failure. Defaults for the automatic run are the cheapest model and one run per case; the manual inputs override them. The comparison table is appended to `$GITHUB_STEP_SUMMARY`; the results folder is uploaded as an artifact.
- `docker` ([lines 76-110](../.github/workflows/ci.yml#L76-L110)): `main` only; `packages: write` permission; buildx with GitHub Actions layer cache; tags `:sha` and `:latest`; `APP_VERSION` passed as a build argument; then the pushed image is run and `/health` polled before the job passes. The job outputs the SHA-tagged image name for the next job.
- `deploy` ([lines 112-139](../.github/workflows/ci.yml#L112-L139)): gated on `vars.AZURE_DEPLOY == 'true'` ([line 114](../.github/workflows/ci.yml#L114)); `id-token: write` for OIDC; `azure/login` with client, tenant and subscription ids from the `demo` environment's secrets; `az containerapp secret set` for the key and `az containerapp update` with `ANTHROPIC_API_KEY=secretref:anthropic-api-key` ([line 134](../.github/workflows/ci.yml#L134)); then poll the public `/health` and print the URL into the summary.

Ask the learner where money is spent in this file. Only in `evals`, only on `main` or by hand, and only with the cheapest model unless someone presses the button.

## 8. Testing without a key

[fake_client.py](../tests/fake_client.py): `response`, `tool_use` and `text` build `SimpleNamespace` objects shaped like SDK responses; `FakeClient` replays a script; `EchoClient` ([lines 56-85](../tests/fake_client.py#L56-L85)) is the one the API tests use. It reads the prompt from the first message, extracts the two paths and the target with a regex, asks for all three tools on those exact paths, and on the second call returns a fixed report with a verdict block. The real tools run on the uploaded files, so the test exercises the agent loop, the tool code and the run logger, with only the model faked.

[test_api.py](../tests/test_api.py): the `client` fixture ([lines 24-30](../tests/test_api.py#L24-L30)) points `RUNS_DIR` at `tmp_path`, narrows the allowlist to include the fake model, and swaps `agent.get_client`. Then, one test per layer of Figure 4:

| test | layer |
|---|---|
| `test_health` | health reports the configuration, including that a key is set, without the value |
| `test_evaluate_then_fetch_run` | the happy path end to end, the folder on disk, `GET /runs/{id}` with and without the trace, `GET /runs` |
| `test_upload_name_does_not_reach_the_model` | an upload named `shuffled_income_leak.csv` produces a prompt with neither word in it |
| `test_validation_errors` | four 400s with their messages: column sets, missing target, too few rows, one class |
| `test_non_csv_and_garbage_rejected` | extension check and unparseable bytes |
| `test_upload_limit` | 413 with the limit lowered to 200 bytes |
| `test_model_allowlist` | 400 for a model not on the list |
| `test_unknown_run_and_traversal` | 404 for an unknown id, a traversal attempt, and a malformed id |
| `test_busy_returns_503` | the semaphore held by the test, the request told to retry |
| `test_upstream_failure_is_502` | a client that raises the SDK's connection error becomes a 502 |

## 9. The deployment guide

[docs/deploy.md](../docs/deploy.md) is written to be executed top to bottom: local container run (section 1), what each CI job does and which secret it needs (2), the `az` commands for the group, environment and app with its secret and limits (3.1 to 3.2), the optional Azure Files mount for persistent run folders (3.3), the federated credential and the exact secrets and variables to set in GitHub (3.4), tear-down (4) and the honest status (5). Walk a learner through 3.2 and ask what each flag protects: `--min-replicas 0` protects the bill, `secretref` protects the key, `API_DEFAULT_MODEL=claude-haiku-5-5` and `MAX_CONCURRENT_EVALUATIONS=1` protect against a stranger finding the URL.

## 10. If you have ten minutes to explain the whole thing

1. Open [api.py:290](../src/api.py#L290). "One request: allowlist, slot, limit, parse, validate, neutral names, run, map errors, clean up."
2. Open [api.py:340](../src/api.py#L340). "The run id is matched before it becomes a path."
3. Open [Dockerfile:11](../Dockerfile#L11). "Second stage: slim base, one system library, non-root, venv and `src/` only, health check."
4. Open [ci.yml:28](../.github/workflows/ci.yml#L28). "Tests gate everything. Money only in `evals`, only on main or by hand."
5. Open [ci.yml:114](../.github/workflows/ci.yml#L114). "Deploy is a variable and three OIDC secrets; without them the image is still published."
6. Open [docs/deploy.md](../docs/deploy.md) section 3.2. "These are the commands; they have not been run yet, and the README says so."
7. Open [test_api.py:87](../tests/test_api.py#L87). "The one test that connects this phase to phase 1: a file name cannot hint the model."

## 11. Exercises

1. Lower `MAX_ROWS` to 50 with an environment variable, restart `make api`, and upload a 2000-row golden case. Which status comes back, and which line produced it?
2. Hold the semaphore from a Python shell while sending two requests from another terminal. Explain why the second gets a 503 and not a timeout.
3. Add `GET /runs/{id}/report` that returns `text/markdown`. Decide whether it should also match the run id pattern first, and why.
4. Make the `evals` job run on pull requests too, then compute what it would cost per PR at the current list prices. Decide whether to keep the change.
5. Run the guide's section 3 against a sandbox subscription, then delete the resource group. Time both.

## 12. Gotchas met while building

- Ruff's B008 flags `File(...)` and `Form(...)` as defaults; the `Annotated[UploadFile, File(...)]` form is both lint-clean and the current FastAPI idiom.
- The installed package's version lags `pyproject.toml` until `pip install -e .` is re-run, and a container never installs the package at all; hence `APP_VERSION` from the environment.
- The slim Python image lacks `libgomp1`; XGBoost imports fine and fails on first use.
- `.gitignore` and `.dockerignore` have the same syntax but different jobs; the second must exclude `.env` and the results folders explicitly.
- `pip freeze` includes `httpcore` from the test client; harmless in the lock, but a lock built from a dev environment is wider than the runtime needs.
- A `HEALTHCHECK` without `--start-period` marks a container unhealthy while uvicorn is still importing XGBoost.
