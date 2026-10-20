# The eval harness, file by file: a code walkthrough

This is the companion to [eval_harness.md](eval_harness.md). That note explains *why* the harness is designed the way it is and gives the theory and the references. This one explains *how the code does it*, in the order the data flows, with a pointer into the source for every claim. It is written so that you can sit next to someone, open the files it names, and talk them through commit `12f3d91` ("Phase 1: agent eval harness and regression suite").

Conventions: `file:line` links point into the repository. Each section opens with the one idea it implements and the place in the note where the theory lives, then shows the code.

## 0. The map

| What | Where | Theory in the note |
|---|---|---|
| The agent returns a gradeable record of its run | [../src/agent.py](../src/agent.py) | section 1, section 9 |
| Known-answer inputs | [../evals/generate_cases.py](../evals/generate_cases.py), [../evals/cases/](../evals/cases/) | sections 2 to 4 |
| Loading a case, hiding its name from the model | [../evals/cases.py](../evals/cases.py) | section 6, item 1 |
| Shared number parsing | [../evals/checks/common.py](../evals/checks/common.py) | section 5.2 |
| Check 1: tool calls | [../evals/checks/tool_sequence.py](../evals/checks/tool_sequence.py) | section 5.1 |
| Check 2: every number is quoted, not computed | [../evals/checks/numeric_faithfulness.py](../evals/checks/numeric_faithfulness.py) | section 5.2 |
| Check 3: the verdict, and the answer key's own rules | [../evals/checks/verdict.py](../evals/checks/verdict.py) | sections 4, 5.3 |
| A second model grades clarity | [../evals/checks/judge.py](../evals/checks/judge.py) | section 7 |
| Running everything N times and summarising | [../evals/run_eval.py](../evals/run_eval.py), [../evals/compare.py](../evals/compare.py) | section 6 |
| Cheap deterministic tests that never call the model | [../regression/](../regression/) | section 8 |
| Known-answer tests for the tools | [../tests/test_tools.py](../tests/test_tools.py) | section 1 |
| Figures | [figures/make_figures.py](figures/make_figures.py) | every figure |

One sentence to hold on to while reading: **the model is the only non-deterministic component, so everything around it is built to be exact, and the model's output is tested by properties, repeatedly.**

## 1. The agent: make a run gradeable

**Idea.** You cannot grade what you cannot see. Before this commit `run_agent` returned a string. Now it returns everything the checkers need. (Note, section 1 and section 9.)

### 1.1 The record of a run

[agent.py:200-237](../src/agent.py#L200-L237) defines three dataclasses:

```python
@dataclass
class ToolCall:            # one tool invocation: what the model asked, what came back
    name: str
    input: dict[str, Any]
    result: Mapping[str, Any]
    is_error: bool = False

@dataclass
class Usage:               # tokens summed over every API call of the run
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass
class AgentRun:            # the whole run
    text: str                                   # the report
    tool_calls: list[ToolCall]                  # ordered trace
    usage: Usage
    stop_reason: str                            # "end_turn", "max_tokens", "refusal", "max_turns"
    model_requested: str
    model_served: str                           # what the API actually answered with
    system_prompt: str
    user_prompt: str
    messages: list[Any]                         # raw conversation, for debugging
```

How to explain each field: `tool_calls` feeds check 1; `tool_calls[*].result` plus the two prompts feed check 2's allowed set; `text` feeds checks 2 and 3; `stop_reason` lets the runner separate "the model was cut off" from "the model was wrong"; `model_served` makes a silent model substitution visible in the results; `usage` is read from the API, never estimated.

`AgentRun.__str__` returns the text, and [report.py:14-24](../src/report.py#L14-L24) accepts either a string or anything with a `.text`, so the README's `save_report(run_agent(prompt))` still works. This is the "do not break the existing API" constraint from the spec.

### 1.2 The loop

[agent.py:250-311](../src/agent.py#L250-L311). Walk through it top to bottom:

- [line 267](../src/agent.py#L267): `for _turn in range(max_turns)`. One iteration is one API call. The loop ends when the model stops asking for tools or when the turn budget runs out.
- [lines 275-277](../src/agent.py#L275-L277): after every response, add its tokens to `run.usage`, record `response.model` and `response.stop_reason`. Doing this on *every* turn, not only the last, is why the token totals are right.
- [line 283](../src/agent.py#L283): `if response.stop_reason != "tool_use"` the model is done. Collect the text blocks and return.
- [line 295](../src/agent.py#L295): otherwise run each requested tool through `run_tool`, which returns `(result, is_error)`. An exception inside a tool becomes `{"error": "KeyError: ..."}` with `is_error=True`; it is appended to the trace *and* sent back to the model with `"is_error": is_error` on the tool result block ([line 304](../src/agent.py#L304)). The model sees that its tool failed; check 1 sees it too.
- [line 309](../src/agent.py#L309): if the loop exhausts, `stop_reason = "max_turns"`. The runner classes that as a plumbing failure, not a model failure.

### 1.3 Lazy client

[agent.py:42-54](../src/agent.py#L42-L54). The old module created `Anthropic()` at import time, which needs a key. Now `get_client()` creates it on first use. Why it matters: the regression suite imports `agent` (for `SYSTEM_PROMPT`, `TOOLS`, `ToolCall`) and must run in CI without a key.

### 1.4 The system prompt is part of the test contract

[agent.py:58-104](../src/agent.py#L58-L104). The thresholds are module constants ([lines 35-37](../src/agent.py#L35-L37)) interpolated into the prompt, so the prompt and the answer-key rules cannot drift apart. Four things the prompt does that the checkers rely on:

1. states the verdict rules mechanically (check 3 compares against exactly these);
2. "You are a reporter, not a calculator" and "never compute a new figure" (check 2 enforces it);
3. "Name a column as a problem only if its own tool result crosses its threshold" (check 3's exact-flag rule; the shuffle case);
4. asks for one fenced JSON block at the end (check 3 parses it).

A junior will ask "isn't it cheating to tell the model the rules?" The honest answer is in the note, section 10: yes, it makes the verdict task easier, and the note says so. The harness measures whether the model *follows* rules and *quotes* numbers faithfully. Those are the properties a tool-using agent must have before anything else.

### 1.5 The one tool fix

[tools.py:64-80](../src/tools.py#L64-L80). The JS tool used to reindex both frequency tables on the real file's categories. A category present only in the synthetic file was silently dropped, and because `scipy.spatial.distance.jensenshannon` renormalises its inputs, the two tables could come out identical and the distance exactly 0.0. Now the categories are the union of both files and a missing level gets a zero count, not a pseudo-count. `regression/test_synth_only_level_not_dropped.py` pins it.

## 2. Golden cases: the answer is known before the model runs

**Idea.** Build inputs by applying one known corruption to a known dataset, so that the correct verdict follows from the construction. (Note, section 2; theory: metamorphic testing, references 3 and 4.)

### 2.1 The generative process

[generate_cases.py:43-63](../evals/generate_cases.py#L43-L63), `draw_population`. Three categoricals from fixed proportions, three numerics from normals, and the label at [line 51](../evals/generate_cases.py#L51):

```python
logit = 1.2 * _z(income.astype(float)) + 0.6 * _z(score) + rng.normal(0, 1, size=n)
target = (logit > 0).astype(int)
```

Two design choices to point out:

- `income` is independent of `age`, `education`, `city`. If it were not, shuffling it would leave a proxy behind and the shuffle case would not drop utility.
- The noise term keeps the task learnable but not perfect: TRTR lands around 0.77, which leaves room for both a drop (shuffle, leak) and a rise (identical).

Everything is seeded (`SEED = 42`, [line 34](../evals/generate_cases.py#L34)), so `make cases` is byte-for-byte reproducible.

### 2.2 The six constructions

[generate_cases.py:66-168](../evals/generate_cases.py#L66-L168), `build_cases`. One line each:

| case | the line | what it preserves / breaks |
|---|---|---|
| `clean_synthetic` | a second `draw_population(rng)` | the control |
| `income_shuffled` | [line 107](../evals/generate_cases.py#L107) `rng.permutation(income)` | keeps the marginal, breaks the joint |
| `score_leaks_target` | [line 123](../evals/generate_cases.py#L123) `np.where(target == 1, 80.0, 60.0)` | keeps mean 70 and sd 10, makes classes separable |
| `age_x10` | [line 139](../evals/generate_cases.py#L139) `age * 10` | loud numeric shift, no label signal |
| `phd_missing` | [line 154](../evals/generate_cases.py#L154) `phd -> master` | one level disappears |
| `identical` | `real.copy()` in `main` | everything; exposes the held-out overlap |

### 2.3 The answer key is written by hand, on purpose

Each case's `expected.json` is built in the same function from the *construction*, not from running the tools. Look at the `expected(...)` helper calls: the verdict and `flagged_columns` are literals. This matters pedagogically: if the key were derived from tool output, a bug in a tool would silently become "expected". Instead `regression/test_expected_matches_tool_output.py` checks hand-written key against derived key, and a disagreement is a signal to investigate, not something to auto-fix.

## 3. Loading cases and hiding their names

**Idea.** The model reads file paths. A path like `evals/cases/age_x10/synth.csv` is an answer key. (Note, section 6, item 1; the Clever Hans references 30 and 31.)

[cases.py:28-49](../evals/cases.py#L28-L49) is a frozen `Case` dataclass: the two paths, the parsed `expected.json`, and `prompt()` which fills `PROMPT_TEMPLATE` ([lines 22-25](../evals/cases.py#L22-L25)). `load_cases` ([line 61](../evals/cases.py#L61)) discovers every directory under `evals/cases/` that holds an `expected.json`.

`neutral_copy` ([lines 85-99](../evals/cases.py#L85-L99)) is the interesting part:

```python
suffix = "".join(random.choices(string.ascii_lowercase, k=6))   # letters only
directory = base / f"eval_{suffix}"
shutil.copyfile(case.real_path, real)                           # real.csv
shutil.copyfile(case.synth_path, synth)                         # synth.csv
```

Two details to explain: letters only, because a digit in the path would be picked up by check 2 as a "number in the prompt" and widen the allowed set; and the copy is cleaned up by the runner after scoring. `regression/test_case_names_do_not_leak_into_prompts.py` asserts that no case name, no corruption word and no digit appears in a prompt built from a neutral copy.

## 4. Shared plumbing: finding numbers in text

**Idea.** Both check 2 and the allowed-set builder need "every number in this text, with its precision". (Note, section 5.2.)

[common.py:13-17](../evals/checks/common.py#L13-L17):

```python
NUMBER_RE = re.compile(
    r"(?<![\w.])[-−]?\d[\d,]*(?:\.\d+)?"
    r"(?:%|x|×|\s?(?:percentage\s+points?|percent|pp)\b)?(?!\w)"
)
```

Read it aloud with the learner:

- `(?<![\w.])` look-behind: the digit must not follow a word character or a dot. This keeps `age_x10`, `v2`, `sha1` and the `5` inside `0.5` from being matched on their own.
- `[-−]?` optional ASCII or Unicode minus.
- `\d[\d,]*(?:\.\d+)?` digits with optional thousands separators and decimals.
- the optional suffix: `%`, `x`, `×`, or the words `percent`, `percentage points`, `pp`. Suffixes are captured so that "a factor of 10x" is still a number and "21 percentage points" is read as a percentage.
- `(?!\w)` no word character after, so `2nd` is not a number.

`parse_number_token` ([lines 55-71](../evals/checks/common.py#L55-L71)) turns a match into `(value, decimals)`: `"79%"` becomes `(0.79, 2)`, `"1,234.5"` becomes `(1234.5, 1)`. The *decimals* count is what the grounding rule keys on.

`iter_numbers` ([line 32](../evals/checks/common.py#L32)) walks a tool result recursively (dicts, lists, scalars, strings) and yields every number. `CheckResult` ([line 20](../evals/checks/common.py#L20)) is the common return type: a name, `passed`, and a `details` dict that ends up in the transcript.

## 5. Check 1: were the right tools called, on the right files?

**Idea.** The tool trace is structured data, so it can be checked exactly. Grade the outcome, not the order. (Note, section 5.1; references 26 and 27.)

[tool_sequence.py:23-58](../evals/checks/tool_sequence.py#L23-L58). The function receives the trace and the case's facts and builds a `problems` list:

- [line 31](../evals/checks/tool_sequence.py#L31): `Counter(tc.name ...)`; each expected tool must appear exactly once, and no unexpected tool may appear.
- [lines 45-47](../evals/checks/tool_sequence.py#L45-L47): both path arguments of every call must resolve to the case's files (`_same_file` compares resolved paths, so `./x.csv` and `x.csv` agree).
- [line 49](../evals/checks/tool_sequence.py#L49): `tstr_xgboost` must have received the prompt's target column.
- [line 51](../evals/checks/tool_sequence.py#L51): any `is_error` fails the run.

Why order is not asserted: the three tools do not depend on each other. Asserting an order would fail correct runs and teach nothing.

## 6. Check 2: every number must be quoted, not computed

**Idea.** The agent's only legitimate sources of numbers are the tool results and the text it was shown. A number outside that set is invented, even if the arithmetic was right. (Note, section 5.2 and Figure 6; faithfulness references 28 and 29.)

Three steps in [numeric_faithfulness.py](../evals/checks/numeric_faithfulness.py):

**Build the allowed set**, [lines 35-43](../evals/checks/numeric_faithfulness.py#L35-L43): every number in every tool result (via `iter_numbers`) plus every number in the system prompt, the user prompt and the tool descriptions. This is where 0.1, 0.2, 0.05, 80, 20, 0 and 1 come from.

**Clean the report**, [lines 46-54](../evals/checks/numeric_faithfulness.py#L46-L54): remove the verdict block, fenced code, Markdown table rules (`|---|`), horizontal rules, numbered headings (`## 2. Numerical`) and list markers (`1.`). Those digits are furniture, not figures.

**Ground each number**, [lines 67-79](../evals/checks/numeric_faithfulness.py#L67-L79):

```python
def is_grounded(value, decimals, allowed):
    v = abs(value)
    for a in allowed:
        a = abs(a)
        if math.isclose(a, v, abs_tol=1e-9):                       # exact
            return True
        if decimals >= 2 and (round_half_up(a, decimals) == v       # rounds to it
                              or round(a, decimals) == v
                              or truncate(a, decimals) == v):      # or truncates to it
            return True
    return False
```

The rule to explain carefully: **two or more decimals may be a rounding or a truncation of an allowed value; fewer than two must match exactly.** Ask the learner why. The answer: at one decimal almost everything collides. The real education distance 0.294 would "ground" an invented "0.3 threshold" if rounding were allowed there. `regression/test_no_invented_threshold.py` has exactly that scenario.

Sign is ignored (`abs`) because "a gap of 0.23 in favour of synthetic" is a faithful quote of −0.230.

Three live examples to use when teaching, all real (note, section 10): "more than 24 times the threshold" fails (4.909 / 0.2 is the model's arithmetic); "1" in "0 = identical, 1 = different" *used* to fail until the JS range was added to the tool description, which is the right fix because now the model was shown it; "21 percentage points" used to fail until percentage words were parsed like `%`. The lesson: when the checker flags something, read it; the checker can be wrong, and the fix should make the allowed set more honest, not the rule looser.

The result ([lines 82-103](../evals/checks/numeric_faithfulness.py#L82-L103)) carries `numbers_total`, `numbers_grounded`, and the list of `ungrounded` tokens, so the summary can report both "fraction of numbers grounded" and "fraction of runs with no invented number".

## 7. Check 3: the verdict block and the rules behind the answer key

**Idea.** Parsing "excellent" out of prose is brittle, so ask for a JSON block and compare it mechanically. And keep the answer key honest by deriving it from the same rules. (Note, sections 4 and 5.3.)

### 7.1 Finding the block

[verdict.py:28-67](../evals/checks/verdict.py#L28-L67). `_candidates` finds every `{"verdict": ...}` object: fenced blocks first (regex `FENCED_BLOCK`), then unfenced ones by `json.JSONDecoder().raw_decode` from each `{"verdict"` occurrence ([line 45](../evals/checks/verdict.py#L45)). `parse_verdict_block` takes the last one ([line 58](../evals/checks/verdict.py#L58)), because a model that corrects itself puts the correction last. `strip_verdict_block` ([line 61](../evals/checks/verdict.py#L61)) returns the prose without that block, which both check 2 and the "named in prose" rule use.

A bug worth telling as a story: `raw_decode` returns an *absolute end index*, not a length. The first version added it to the start offset and swallowed everything after the block. A regression test caught it within minutes because the appended sentence in `test_no_invented_threshold` vanished. That is what the oracle-and-null tests are for.

### 7.2 The rules, in code

[verdict.py:78-109](../evals/checks/verdict.py#L78-L109), `derive_expected`:

```python
cat_flags = columns with JS  > 0.1, excluding the target
num_flags = columns with d   > 0.2, excluding the target
utility   = "poor" if gap > 0.05 else "good"          # gap <= 0 is good
overall   = "poor" if utility poor
            else "excellent" if both fidelity axes good
            else "acceptable"
```

This is the system prompt's rule set written as a function. Its job is not to grade the model; it is to grade the *answer key*: `test_expected_matches_tool_output.py` runs the real tools on each case and asserts the hand-written `expected.json` agrees. Change a threshold in one place and the test tells you the key is stale.

### 7.3 The comparison

[verdict.py:116-142](../evals/checks/verdict.py#L116-L142), `check_verdict`: the four axes must match; the flagged set must match exactly, in both directions ([lines 130-133](../evals/checks/verdict.py#L130-L133), "flagged without evidence" and "missed flags"); and every expected column must be named in the prose ([line 136](../evals/checks/verdict.py#L136)), so a model cannot satisfy the block while writing prose about something else. The "flagged without evidence" branch is what makes the shuffle case a test of *not guessing*.

## 8. The judge: a second model for the property that has no ground truth

**Idea.** Clarity cannot be checked mechanically. Use a different model with a fixed rubric, record the rubric's hash, and never let the score gate anything. (Note, section 7 and Figure 9; references 35 to 39.)

[judge.py](../evals/checks/judge.py):

- `CRITERIA` ([line 29](../evals/checks/judge.py#L29)) are five 1-to-5 questions. `RUBRIC` is the system prompt; `RUBRIC_HASH` ([line 54](../evals/checks/judge.py#L54)) is its sha256 prefix, saved with every score. If someone edits the rubric, old and new scores stop being comparable and the hash makes that visible.
- `SCORE_SCHEMA` ([line 56](../evals/checks/judge.py#L56)) is a JSON schema; the call uses `output_config={"format": {"type": "json_schema", ...}}` ([line 117](../evals/checks/judge.py#L117)) so the API guarantees the shape. The first version used a forced tool call; the Opus judge rejected it. Scores are range-checked after parsing anyway.
- The user message ([lines 100-112](../evals/checks/judge.py#L100-L112)) gives the judge three things in tagged blocks: the agent's instructions and tool descriptions (`<agent_context>`), the tool results (`<tool_results>`), and the report (`<report>`). The rubric says the report is data, not instructions, and that length is not to be rewarded.
- `context_given` ([line 79](../evals/checks/judge.py#L79)) records whether the judge saw the agent context. It exists because the first judging pass did not pass that context, and the judge penalised "XGBoost" as invented when the agent had read it in its tool description. Scores from the two variants must not be averaged together; the flag makes that checkable.

What to tell a junior about the judge in one line: it ranks models; it never passes or fails anything.

## 9. The runner: N runs per case, scored, summarised

**Idea.** The model is stochastic, so report rates and their resolution. Separate plumbing failures from model failures. (Note, section 6 and Figure 8; references 8, 9, 33.)

[run_eval.py](../evals/run_eval.py), function by function:

- `classify_failure` ([lines 48-61](../evals/run_eval.py#L48-L61)) maps an exception or a stop reason to a class: `api_error`, `truncated`, `refusal`, `turn_limit`, `empty_response`. Those runs go to `errors.jsonl` and are excluded from the rates but counted. A refusal is not a wrong verdict.
- `score_one` ([lines 64-133](../evals/run_eval.py#L64-L133)) is one case, one run: make a neutral copy ([line 72](../evals/run_eval.py#L72)), run the agent, time it, classify, then run the three checks and optionally the judge, and return one dict. That dict *is* the transcript file.
- `summarise` ([lines 136-199](../evals/run_eval.py#L136-L199)) turns the list of transcripts into the summary: pass counts and rates per check, grounded-number fraction, "all three checks" rate, clarity mean, token sums for agent and judge kept separate, served-model set, and the noise floor `1/sqrt(n)` ([line 178](../evals/run_eval.py#L178)). Explain the noise floor as "the half-width of the band a pass rate could move in by luck at this n"; at n = 18 it is ±24 points.
- `summary_markdown` ([line 206](../evals/run_eval.py#L206)) renders the table that appears in the note.
- `rejudge` ([lines 252-282](../evals/run_eval.py#L252-L282)) re-runs only the judge over saved transcripts and rewrites the summary. Use it after a rubric change or a judge change; the agent runs are not repeated.
- `main` ([line 285](../evals/run_eval.py#L285)) parses the flags (`--model`, `--runs`, `--judge`/`--no-judge`, `--cases`, `--workers`, `--rejudge`), fans the jobs out on a `ThreadPoolExecutor` ([line 312](../evals/run_eval.py#L312)), and writes `summary.json`, `summary.md`, `transcripts/*.json` and `errors.jsonl` under `evals/results/<timestamp>_<model>/`.

[compare.py](../evals/compare.py) reads every `summary.json` and prints one row per folder. That is the "one row per model version" the note promises.

## 10. The regression suite: deterministic, free, on every commit

**Idea.** A regression test for an LLM can be deterministic if it replays a recorded transcript through the checkers instead of calling the model. It tests the checker and the pipeline, not the model. (Note, section 8 and Figure 7; references 40 to 42.)

### 10.1 Fixtures and the oracle

[conftest.py](../regression/conftest.py) provides two session-scoped fixtures: `cases` and `tool_outputs`, the real tool results on every case, computed once ([lines 24-40](../regression/conftest.py#L24-L40)). The tools are deterministic (seeded split), so this is exact and takes a few seconds.

[helpers.py:43-82](../regression/helpers.py#L43-L82), `oracle_report`, builds a report straight from tool results: it quotes every value verbatim, applies `derive_expected` ([line 47](../regression/helpers.py#L47)), says the honest thing when utility is poor and nothing is flagged, and ends with the JSON block ([line 80](../regression/helpers.py#L80)). It is the reference answer: it must pass all three checks on every case. If a checker change makes it fail, the checker broke. `fake_tool_calls` builds a correct trace for check 1; `prompts_for` returns the three prompt texts that define the allowed set.

### 10.2 The tests, and the kind of thing each one teaches

| file | teaches |
|---|---|
| `test_checkers_pass_oracle_and_fail_null.py` | the sanity bracket: the oracle passes, an empty report fails, broken traces fail |
| `test_expected_matches_tool_output.py` | the answer key agrees with the rules, and the constructions produce their intended effects (shuffle keeps d, drops TSTR; leak keeps d under 0.2) |
| `test_numeric_faithfulness_rules.py` | each grounding rule in isolation, plus the reconstructed recomputed-mean transcript |
| `test_verdict_block_parsing.py` | fenced, unfenced, last block wins, prose mention, exact flags |
| `test_no_invented_threshold.py` | 0.3 next to a real 0.294 is still invented |
| `test_shuffle_not_blamed.py` | flagging income or score without evidence fails; the honest report passes |
| `test_identical_negative_gap_is_excellent.py` | a gap at or below zero is good |
| `test_synth_only_level_not_dropped.py` | the tool bug, pinned with a four-row CSV |
| `test_case_names_do_not_leak_into_prompts.py` | neutral paths; the reconstructed path-leak transcript |
| `test_no_computed_ratio.py` | the one genuinely recorded live failure ("24 times") |

Point out the docstrings: each says where the failure came from. That provenance line is the most useful thing in the file six months later.

### 10.3 Adding a new failure

`regression/TEMPLATE_test_new_failure.py.txt` is the recipe: copy a failing transcript from `evals/results/<run>/transcripts/` into `regression/fixtures/`, copy the template, fill the docstring, keep the assertion that the checkers catch it. `test_no_computed_ratio.py` is a finished example, built exactly that way from a live run on 2026-10-18. The two older fixtures are reconstructions; `regression/fixtures/README.md` and `make_fixtures.py` say so.

## 11. Unit tests for the tools

[../tests/test_tools.py](../tests/test_tools.py). These are ordinary known-answer tests and a good warm-up for a junior before the harness:

- JS distance is 0 for identical tables and 1 for disjoint ones; a hand-computed two-level case matches the closed form.
- Cohen's d for `[1..5]` vs `[3..7]` is `2 / sqrt(2.5)`; it is symmetric (absolute) and `None` for a constant column.
- TSTR on an identical synthetic scores at least the baseline; a label-shuffled synthetic loses utility; results are deterministic; string labels and an explicit test file work; `n_test_samples` is 20% of the rows.

## 12. Figures

[figures/make_figures.py](figures/make_figures.py) regenerates every figure in the note from the cases and real tool output, with no API key. `tool_outputs()` ([line 103](figures/make_figures.py#L103)) runs the tools once; each `figN_*` function draws one figure; the colour roles are fixed at the top (blue = deterministic code and real data, orange = the model and synthetic data, red only for "must flag" and always with a text marker). It also writes `tool_output_table.csv`, the source of the table in the note's section 2.

## 13. If you have ten minutes to explain the whole thing

1. Open [agent.py:222](../src/agent.py#L222). "The agent now returns this. Everything else exists to grade it."
2. Open [generate_cases.py:107](../evals/generate_cases.py#L107). "One line of corruption, so the right answer is known. Six of these."
3. Open [cases.py:91](../evals/cases.py#L91). "The model reads paths, so we hide the case name."
4. Open [numeric_faithfulness.py:67](../evals/checks/numeric_faithfulness.py#L67). "A number is allowed only if the model could have read it. Two decimals may round; fewer must match exactly."
5. Open [verdict.py:78](../evals/checks/verdict.py#L78). "The prompt's rules as a function. It grades the answer key, not the model."
6. Open [run_eval.py:178](../evals/run_eval.py#L178). "Rates, with their noise floor. At eighteen runs that is ±24 points."
7. Open [helpers.py:43](../regression/helpers.py#L43). "The oracle must pass; recorded failures must fail; none of this calls the model."
8. Open the note's section 10. "Here is what it found, including what it found wrong with itself."

## 14. Exercises for someone learning it

1. Add a seventh case: a column with the same mean and spread as real but a different shape (bimodal). Write its `expected.json` by hand first, then run `make regression` and see whether `test_expected_matches_tool_output` agrees with you.
2. Change `COHEN_D_THRESHOLD` to 0.5 in `agent.py` only. Run `make regression`. Explain every failure, then decide where the threshold should really live.
3. Take any transcript under `evals/results/`, edit one number in its report by a rounding error, save it as a fixture, and write the test with the template. Time yourself; the target is five minutes.
4. Write a report that passes checks 1 to 3 and is nevertheless a bad report. (Hint: quote every number, get the verdict right, and explain nothing.) Then read the judge rubric and explain which criterion catches it.

## 15. Gotchas met while building, worth passing on

- `json.JSONDecoder.raw_decode` returns an absolute end index, not a length.
- Forced `tool_choice` is not supported by every model; structured output via `output_config` is the portable way to get typed JSON.
- A regex that finds numbers must decide what to do with `age_x10`, `v2`, `0.5`, `2nd`, `10x`, `21 percentage points` and `1,234`. Write the test file for the rules before the regex.
- The judge sees only what you send it. If it does not see the agent's instructions, it will call "XGBoost" an invention.
- Tag anything that changes the meaning of a score (`rubric_hash`, `context_given`) so that two result folders cannot be compared by accident.
- When a checker flags a number, read the sentence. Two of the first three flags were the checker's fault, and both fixes made the allowed set more honest rather than the rule looser.
