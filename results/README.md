# Results

Everything measured in this repository, in two tables. Both are written by a script from saved output, so neither is edited by hand.

| table | what it measures | built from | rebuild |
|---|---|---|---|
| [agent_evals.md](agent_evals.md) | The agent itself: tool selection, numeric faithfulness, verdict correctness and judged clarity, for every eval session and every golden case | [../evals/results/](../evals/results/) summaries and [../evals/cases/](../evals/cases/) answer keys | `make results` (no API key) |
| [generators/comparison.md](generators/comparison.md) | CTGAN, TVAE and an LLM baseline on Adult, Cardio and Credit, scored by the agent's four metric tools and judged by the agent | [generators/comparison.json](generators/comparison.json) and the generated samples next to it | `make compare-generators` (API key, and `make fetch-generators` first) |

`make results` only tabulates sessions that already exist. A new session comes from `make eval`, which calls the model.

How to read each table, and what neither of them shows, is in the [README](../README.md#results).
