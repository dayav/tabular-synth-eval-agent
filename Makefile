# Phase 1 targets. MODEL/RUNS/JUDGE override the eval defaults.
MODEL ?= claude-sonnet-4-6
RUNS  ?= 3
JUDGE ?= claude-opus-5-5
PY    ?= python
IMAGE ?= ghcr.io/dayav/tabular-synth-eval-agent:local
PORT  ?= 8000

.PHONY: fetch-generators compare-generators test regression unit eval eval-nojudge compare cases fixtures figures typecheck lint runs example api docker-build docker-run lock

test: unit regression

unit:
	$(PY) -m pytest tests -q

regression:
	$(PY) -m pytest regression -q

eval:
	$(PY) -m evals.run_eval --model $(MODEL) --runs $(RUNS) --judge $(JUDGE)

eval-nojudge:
	$(PY) -m evals.run_eval --model $(MODEL) --runs $(RUNS) --no-judge

compare:
	$(PY) -m evals.compare

runs:
	$(PY) runs/summarize.py

example:  # one logged run on the bundled sample data; move the folder to runs/example to refresh the committed one
	$(PY) src/agent.py

cases:
	$(PY) -m evals.generate_cases

fixtures:
	$(PY) regression/fixtures/make_fixtures.py

figures:
	$(PY) notes/figures/make_figures.py
	$(PY) notes/figures/make_runlog_figures.py
	$(PY) notes/figures/make_service_figures.py
	$(PY) notes/figures/make_privacy_figures.py
	$(PY) notes/figures/make_generation_figures.py

typecheck:
	$(PY) -m mypy

lint:
	$(PY) -m ruff check src evals regression tests notes

api:  # development server with auto-reload; reads ANTHROPIC_API_KEY from the environment or .env
	cd src && $(PY) -m uvicorn api:app --reload --port $(PORT)

docker-build:
	docker build -t $(IMAGE) .

docker-run:  # serves on $(PORT); run folders land in ./runs via the volume mount
	docker run --rm -p $(PORT):8000 --env-file .env -v "$(CURDIR)/runs:/data/runs" $(IMAGE)

lock:  # refresh requirements.lock from the current environment (direct deps + their resolved set)
	$(PY) -m pip freeze | grep -iE "^(anthropic|pandas|scikit-learn|scipy|xgboost|numpy|python-dotenv|fastapi|uvicorn|python-multipart|pydantic|pydantic_core|starlette|httpx2|anyio|joblib|threadpoolctl|annotated-types|typing_extensions|h11|click|sniffio|distro|jiter|docstring-parser|python-dateutil|pytz|tzdata|six|idna|certifi|uvloop|httptools|websockets|watchfiles|PyYAML|annotated-doc)==" | sort -f > requirements.lock

fetch-generators:  # thesis datasets + CTGAN/TVAE outputs into data/generators/ (needs the 'tables' package)
	$(PY) scripts/fetch_generator_outputs.py

compare-generators:  # scores CTGAN, TVAE and the LLM baseline with the four tools and the agent; updates the README table
	$(PY) scripts/compare_generators.py
