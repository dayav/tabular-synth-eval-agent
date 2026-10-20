# Phase 1 targets. MODEL/RUNS/JUDGE override the eval defaults.
MODEL ?= claude-sonnet-4-6
RUNS  ?= 3
JUDGE ?= claude-opus-5-5
PY    ?= python

.PHONY: test regression unit eval eval-nojudge compare cases fixtures figures typecheck lint runs example

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

typecheck:
	$(PY) -m mypy

lint:
	$(PY) -m ruff check src evals regression tests notes
