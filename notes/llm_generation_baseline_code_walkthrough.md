# The generation baseline, file by file: a code walkthrough

Companion to [llm_generation_baseline.md](llm_generation_baseline.md), which explains *why* the comparison is set up as it is and what it showed. This one explains *how the code does it*, from the schema summary to the README splice, with a pointer into the source for every claim, so you can open the files next to someone and talk them through phase 6.

Conventions: `file:line` links point into the repository. Each section opens with the one idea it implements and where the theory lives, then shows the code.

## 0. The map

| What | Where | Theory in the note |
|---|---|---|
| Schema summary: what the model is shown | [../src/generate.py:63-96](../src/generate.py#L63-L96) | section 2 |
| The strict JSON schema | [../src/generate.py:99-117](../src/generate.py#L99-L117) | section 2 |
| Row validation | [../src/generate.py:120-143](../src/generate.py#L120-L143) | section 2 |
| The generation loop | [../src/generate.py:163-242](../src/generate.py#L163-L242) | sections 2, 6 |
| Fetching the datasets and generator outputs | [../scripts/fetch_generator_outputs.py](../scripts/fetch_generator_outputs.py) | section 3 |
| Scoring on equal footing and the table | [../scripts/compare_generators.py](../scripts/compare_generators.py) | sections 3, 4 |
| Tests with a scripted client | [../tests/test_generate.py](../tests/test_generate.py), [../tests/test_compare_generators.py](../tests/test_compare_generators.py) | section 2 |
| Figures | [figures/make_generation_figures.py](figures/make_generation_figures.py) | every figure |

One sentence to hold on to: **the model is shown a summary and returns candidates; deterministic code decides what the summary says and which candidates count.**

## 1. The schema summary

**Idea.** Reduce the real file to the least the model needs, and nothing it could copy. (Note, section 2.)

[generate.py:63-96](../src/generate.py#L63-L96), `summarise_schema(frame, target_column, exclude)`:

- [line 71](../src/generate.py#L71): text columns are found with the same `select_dtypes` call the tools use (pandas 3's `str` dtype included).
- [line 74](../src/generate.py#L74): a column is treated as categorical if it is text, *or* an integer column with at most ten distinct values (`INTEGER_AS_CATEGORY_MAX_LEVELS`), *or* the target with at most ten values. For categoricals the spec carries `levels` and `frequencies` (top 60 levels), and for integer-coded ones also `integer_levels`, so the JSON schema can enumerate the admissible integers.
- Otherwise the spec carries `kind` (`integer` or `float`), `min`, `max`, `mean`, `std` and five quantiles.

Ask the learner what is *not* in a spec: no row, no pair of columns, no correlation. That absence is the experiment.

## 2. The JSON schema and the validator

**Idea.** Constrain what can be parsed; validate what the constraint cannot express. (Note, section 2.)

[generate.py:99-117](../src/generate.py#L99-L117), `json_schema_for(specs)`: one object per row with every column `required` and `additionalProperties: false` ([line 109](../src/generate.py#L109)); categoricals get an `enum` (strings, or integers for integer-coded columns); numerics get `integer` or `number`. The outer object is `{"rows": [...]}`. There is no `minItems` or `maxItems`: the API accepts no array-size constraint beyond zero or one item, so the batch size is asked for in the prompt and enforced by trimming ([line 216](../src/generate.py#L216)).

[generate.py:120-143](../src/generate.py#L120-L143), `validate_row(row, specs, slack=0.05)` returns a reason string or `None`: `missing:<col>`, `bad_level:<col>`, `not_numeric:<col>`, `not_integer:<col>`, `out_of_range:<col>` ([line 140](../src/generate.py#L140), range widened by 5% of the span). Booleans are rejected as numbers on purpose (`True` is an `int` in Python). The reasons are counted in the result so a reviewer sees *why* rows were dropped, not only how many.

## 3. The generation loop

[generate.py:163-242](../src/generate.py#L163-L242), `generate_with_llm(real_csv_path, output_csv_path, n_rows, target_column, model, batch_size, exclude_columns, max_attempts_factor, client)`:

1. Read the real file, summarise it, build the schema. The file is not touched again.
2. Loop while fewer than `n_rows` valid rows and fewer than `max_batches` calls ([line 190](../src/generate.py#L190): the minimum number of batches times the attempt factor, so a model that keeps returning invalid rows cannot spend without bound).
3. Each call ([line 203](../src/generate.py#L203)) uses `output_config` with `effort: "low"` and the JSON-schema format; the system prompt says the model is given marginals only and must vary rows; the user prompt carries the batch index and the full spec list.
4. Parse the text as JSON; an unparseable batch is counted and skipped. Validate each row of the first `batch_size` returned; keep the valid ones in the spec's column order.
5. Trim to `n_rows` ([line 222](../src/generate.py#L222)), write the CSV, price the tokens with the phase 2 estimator ([line 226](../src/generate.py#L226)), and return the `GenerationResult`: counts, reasons, batches, models requested and served, tokens, estimated cost, latency, columns and the schema summary.

`client` is injectable, which is what the offline tests use; the default is the agent's lazy client. The tool is not added to `TOOLS` in `agent.py`, and the note says why.

## 4. Fetching the inputs

[fetch_generator_outputs.py](../scripts/fetch_generator_outputs.py): the public Hugging Face dataset repo of the thesis pipeline holds the train and test CSVs under `data/` and the baseline outputs as HDF5 under `baseline/<ds>_<gen>_baseline/`. `download` ([line 34](../scripts/fetch_generator_outputs.py#L34)) streams a file into `data/generators/<ds>/` unless it exists; `h5_to_csv` ([line 45](../scripts/fetch_generator_outputs.py#L45)) reads the `synthetic_data` key with pandas and PyTables and writes a CSV, so the comparison itself needs no HDF5 reader. The folder is git-ignored; the three datasets total about 35 MB.

## 5. The comparison on equal footing

[compare_generators.py](../scripts/compare_generators.py), `compare_dataset` ([line 87](../scripts/compare_generators.py#L87)):

- [line 92](../scripts/compare_generators.py#L92): the reference sample, `rows` real training rows, seeded; [line 93](../scripts/compare_generators.py#L93): the held-out sample, 2,000 test rows, seeded. Both are written once per dataset and reused for every generator.
- For `ctgan` and `tvae`, the full output is read and cut to `rows` with the same seed ([line 112](../scripts/compare_generators.py#L112)). For `llm`, the output is generated from the *reference sample* and cached under `results/generators/llm_<ds>_<model>.csv` with a JSON of the generation stats ([line 101](../scripts/compare_generators.py#L101)); `--regenerate` refreshes it.
- [line 113](../scripts/compare_generators.py#L113): every synthetic frame is reordered to the reference's columns, and the credit `ID` column is excluded everywhere (`EXCLUDE`).
- `evaluate_pair` ([line 55](../scripts/compare_generators.py#L55)) runs the four tools. The utility tool is called with `test_csv_path` ([line 58](../scripts/compare_generators.py#L58)), so TRTR is the same number for all three generators of a dataset. The summary excludes the target from the Cohen's d statistics and counts flagged columns with the documented thresholds.
- `run_agent_verdict` ([line 72](../scripts/compare_generators.py#L72)) prompts the real agent with the privacy sentence, with `log_dir=None` ([line 81](../scripts/compare_generators.py#L81)) so a comparison does not write nine run folders, and parses the verdict block with the API's parser.

`markdown_table` ([line 128](../scripts/compare_generators.py#L128)) renders the README table with the footing sentence above it and the honesty sentence below; `splice_readme` ([line 158](../scripts/compare_generators.py#L158)) replaces the text between the two HTML-comment markers and refuses to run if they are missing. `main` writes `comparison.json` and `comparison.md` to `results/generators/`.

Teaching point: ask where in this file the LLM could have been advantaged or disadvantaged by accident. The answers are the reference sample (it is what the LLM was summarised from, and what every generator is compared against), the row count (equal), and the test set (equal and external). The only asymmetry left is the intended one.

## 6. The tests

[test_generate.py](../tests/test_generate.py), with a `ScriptedClient` ([line 62](../tests/test_generate.py#L62)) whose first batch contains one invalid row:

| test | property |
|---|---|
| `test_schema_summary_kinds_and_stats` | text and low-cardinality integers become categorical with levels and frequencies; numerics get range and quantiles; `exclude` works |
| `test_json_schema_is_strict` | every column required, no extra properties, enums for levels and integer levels, no array-size constraint |
| `test_validate_row_reasons` | each rejection reason on a concrete bad row |
| `test_generate_with_llm_batches_validates_and_writes` | 12 rows requested, one rejected, three batches, tokens summed, unpriced fake model, CSV in column order |
| `test_generate_gives_up_after_attempt_budget` | a client that only returns bad rows stops at the budget with zero rows written |

[test_compare_generators.py](../tests/test_compare_generators.py): `evaluate_pair` on the golden files against a fresh held-out draw (the copy scores identifiability 1.0, the honest re-draw stays in band), and the table renderer plus the README splice on a temporary README.

## 7. The figures

[figures/make_generation_figures.py](figures/make_generation_figures.py) reads `results/generators/comparison.json`: a pipeline diagram, the what-each-saw diagram, and three grouped bar charts (utility gap, fraction of columns flagged, identifiability) with the thresholds and the 0.5 reference drawn in. Rerun it after `make compare-generators` and the figures follow the numbers.

## 8. If you have ten minutes to explain the whole thing

1. Open [generate.py:74](../src/generate.py#L74). "This is everything the model is told. No rows."
2. Open [generate.py:109](../src/generate.py#L109) and [140](../src/generate.py#L140). "The schema makes the parse safe; the validator makes the values safe. Both are needed."
3. Open [compare_generators.py:92](../scripts/compare_generators.py#L92). "One reference sample, one test set, 500 rows each: the footing."
4. Open [compare_generators.py:58](../scripts/compare_generators.py#L58). "Explicit test file, so TRTR is identical across the three generators."
5. Open the README table. "Same band on utility and fidelity, zero exact matches, and the honesty sentence underneath."
6. Open the note's section 5. "These datasets are famous; the model may remember them. The design prevents showing rows, not remembering them, and the measurement says it did not reproduce them."
7. Open `agent.py` and show the tool list. "The generator is not here. An evaluator must not generate what it grades."

## 9. Exercises

1. Add a ten-bin histogram per numeric column to the spec and regenerate credit. Does the number of flagged payment-status columns fall?
2. Run the comparison with `--rows 200` and `--rows 1000` on cardio. Which numbers move, and which should not?
3. Replace XGBoost in `tstr_xgboost` with logistic regression for one run and compare the gaps. Which generator benefits?
4. Generate 500 rows from a dataset the model cannot have seen (the golden real data from phase 1 is one) and compare identifiability with the adult result.
5. Add a `--repeats 3` option that regenerates the LLM output three times and reports mean and spread per cell.

## 10. Gotchas met while building

- Structured output rejects `minItems` and `maxItems` other than 0 or 1 on arrays; ask for the count in the prompt and trim.
- `True` is an `int` in Python; a validator that checks `isinstance(v, int)` accepts booleans unless it excludes them.
- The pipeline's HDF5 files hold the table under a single key with pandas categoricals; converting once to CSV spares every later reader a PyTables dependency.
- `python -I` ignores user-site packages; for reading untrusted data files, run from a directory that holds only data and keep the interpreter normal.
- One row in 1,500 failed validation; without the count nobody would have known whether it was one or three hundred.
