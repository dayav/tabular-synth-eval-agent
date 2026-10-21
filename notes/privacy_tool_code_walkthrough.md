# The privacy tool, file by file: a code walkthrough

Companion to [privacy_tool.md](privacy_tool.md), which explains *what* disclosure risk means here and *why* the tool reports two statistics. This one explains *how the code does it*, from the encoder to the agent prompt to the tests, with a pointer into the source for every claim, so you can open the files next to someone and talk them through phase 4.

Conventions: `file:line` links point into the repository. Each section opens with the one idea it implements and where the theory lives, then shows the code.

## 0. The map

| What | Where | Theory in the note |
|---|---|---|
| The HEOM encoder | [../src/tools.py:259-288](../src/tools.py#L259-L288) | section 2 |
| The tool and its two statistics | [../src/tools.py:291-383](../src/tools.py#L291-L383) | sections 3, 4 |
| The result type | [../src/tools.py:223-256](../src/tools.py#L223-L256) | section 3 |
| Tool schema, registry, prompt rules | [../src/agent.py](../src/agent.py) | section 5 |
| Privacy axis in the answer key and the checker | [../evals/checks/verdict.py](../evals/checks/verdict.py) | section 5 |
| The two golden cases and the privacy prompt | [../evals/generate_cases.py](../evals/generate_cases.py), [../evals/cases.py](../evals/cases.py) | section 6 |
| Regression: oracle, fake calls, new tests | [../regression/helpers.py](../regression/helpers.py), [../regression/test_checkers_pass_oracle_and_fail_null.py](../regression/test_checkers_pass_oracle_and_fail_null.py) | section 5 |
| Unit tests for the tool | [../tests/test_tools.py:113-184](../tests/test_tools.py#L113-L184) | sections 2 to 4 |
| API flag | [../src/api.py](../src/api.py) | section 5 |
| Figures | [figures/make_privacy_figures.py](figures/make_privacy_figures.py) | every figure |

One sentence to hold on to: **one encoder, one set of real radii, two fractions; the one with a reference level is the rule, the other is evidence.**

## 1. The encoder

**Idea.** Turn mixed-type rows into vectors so that Euclidean distance *is* HEOM distance. (Note, section 2 and Figure 1.)

[tools.py:259-288](../src/tools.py#L259-L288), `_heom_encode(real, synth, num_cols, cat_cols)`:

- [line 270](../src/tools.py#L270): `scale = 1 / sqrt(2)`. With two one-hot entries differing by this amount, the squared distance is ½ + ½ = 1, so a category mismatch contributes exactly 1.
- [line 274](../src/tools.py#L274): `MinMaxScaler().fit(real[num_cols])`. Fitted on the real data only, then applied to both. "Range" means the real range; a synthetic value outside it scales beyond [0, 1], which is correct (it is further away).
- [line 278](../src/tools.py#L278): the levels of each categorical column are the real data's levels, in order of first appearance. A synthetic level not among them maps to nothing, so its block stays zero ([line 284](../src/tools.py#L284) only writes where the mapping hit).
- The blocks are horizontally stacked; the function returns two matrices with the same columns.

Ask the learner what happens to a constant numeric column. `MinMaxScaler` handles a zero range by leaving the column at 0, so it contributes nothing to any distance, which is right.

## 2. The tool

**Idea.** Compute each real record's radius once, then answer two questions with nearest-neighbour queries. (Note, section 3 and Figure 2.)

[tools.py:291-383](../src/tools.py#L291-L383), `identifiability_heom(real_csv_path, synth_csv_path, k=2, exclude_columns=None)`, in order:

1. **Guard and select.** `k < 2` is rejected: with k = 1 every real radius would be zero. The usable columns are those in both files and not excluded. Rows with missing values are dropped ([line 330](../src/tools.py#L330)) and the count reported, because a NaN has no distance.
2. **Split by type** ([line 338](../src/tools.py#L338)) with `select_dtypes(include=["object", "category", "string"])`, the same call the fidelity tools use. A gotcha met while building: pandas 3 stores text in a `str` dtype that an `== object` check misses; `select_dtypes` with `"string"` catches it.
3. **Radii** ([lines 345-347](../src/tools.py#L345-L347)): a k-nearest-neighbours fit on the real vectors queried with the real vectors; column k−1 of the distances is the radius (column 0 is the point itself at distance 0).
4. **Nearest real for every synthetic row** ([lines 349-351](../src/tools.py#L349-L351)), from which `exact` counts rows at distance ≤ 1e-9.
5. **Identifiability** ([lines 353-359](../src/tools.py#L353-L359)): subsample the synthetic rows to the real size with a seeded generator (`default_rng(0)`, so the number is reproducible), fit nearest neighbours on that subsample, query with every real row, and count real rows whose nearest synthetic distance is strictly below their radius. Strict `<` is the published definition; a duplicate real row (radius 0) can therefore never be "identified" by distance, which exact matches cover.
6. **εANY** ([lines 361-367](../src/tools.py#L361-L367)): the thesis loop. Radius-neighbours from every synthetic row to the real set at the *largest* radius, then the ANY rule: unsafe if any returned neighbour is nearer than its own radius ([line 366](../src/tools.py#L366)). The `r_max > 0` guard skips the query when all real rows are duplicates.
7. **Return** ([line 370](../src/tools.py#L370) onward) the `PrivacyResult`: both fractions with their counts, exact matches, the two median distances, k, the column split and the dropped-row count.

The result type ([tools.py:223-256](../src/tools.py#L223-L256)) is a `TypedDict` like the other tools', and its docstring is the one place the two statistics are defined side by side with their reference levels. The tool description the model sees in `agent.py` repeats those reference levels, so the numbers 0.5 and 1.0 are in the allowed set of the faithfulness checker when the model quotes them.

## 3. The agent: prompt, schema, registry

**Idea.** The tool is available always, used only when asked, and its two thresholds are prompt constants that the answer key shares. (Note, section 5.)

[agent.py](../src/agent.py):

- Constants [lines 45-46](../src/agent.py#L45-L46): `IDENTIFIABILITY_THRESHOLD = 0.7`, `EXACT_MATCH_THRESHOLD = 0.05`, interpolated into the prompt so the text and the rules cannot drift.
- The prompt ([line 73](../src/agent.py#L73) onward): the fourth tool is listed; [line 77](../src/agent.py#L77) says to call it exactly once, *only* when the request mentions privacy, disclosure risk, re-identification, memorisation or copying; [lines 86-89](../src/agent.py#L86-L89) give the two thresholds, the 0.5 reference, and the instruction never to threshold εANY; [lines 96-99](../src/agent.py#L96-L99) add the privacy axis "only when it was assessed" and the rule that overall is poor when utility or privacy is poor. The JSON block example shows the `privacy` key and says to include it only then.
- Schema [line 212](../src/agent.py#L212): `real_csv_path`, `synth_csv_path`, optional `k` and `exclude_columns`. Registry [line 255](../src/agent.py#L255).

Teaching point: nothing in the loop changed. A new tool is a dictionary entry, a registry entry and prompt text; the trace, the run log and the API pick it up unchanged.

## 4. The answer key and the checker

**Idea.** The harness must enforce "assessed only when asked" from both directions. (Note, sections 5 and 6.)

[verdict.py](../evals/checks/verdict.py):

- `derive_expected` ([line 85](../evals/checks/verdict.py#L85)) takes an optional `privacy_result`. When given, [lines 106-110](../evals/checks/verdict.py#L106-L110) apply the two thresholds and add the axis; the overall rule becomes "poor if utility or privacy is poor". When absent, the output has no privacy key, exactly like a report on a non-privacy request.
- `check_verdict` ([line 139](../evals/checks/verdict.py#L139)): for each axis in the full list, if the expected key does not have it and the report does, that is a problem ("assessed although it was not asked for"); if the expected key has it, it must match.

[cases.py:26](../evals/cases.py#L26) holds the privacy sentence, and [lines 45-54](../evals/cases.py#L45-L54) append it to the prompt when `expected.json` has `ask_privacy: true`. The same sentence is used by the API ([api.py:53](../src/api.py#L53)), so the harness and the service ask the same question.

[generate_cases.py:195-224](../evals/generate_cases.py#L195-L224) builds the two cases from the existing copy and re-draw frames, with `tools` extended by the fourth name and the verdicts written by hand: poor/poor for the copy, good/excellent for the control. The notes field in each says why.

## 5. Regression and unit tests

- [conftest.py](../regression/conftest.py) computes the privacy output only for cases whose expected tool list includes the fourth tool, so the six original cases are unaffected.
- [helpers.py:38](../regression/helpers.py#L38): `fake_tool_calls` adds the fourth call when a privacy output exists; [line 86](../regression/helpers.py#L86): the oracle report adds a privacy section that quotes identifiability, exact matches and εANY verbatim, with the sentence that εANY is supporting evidence only.
- [test_checkers_pass_oracle_and_fail_null.py:64](../regression/test_checkers_pass_oracle_and_fail_null.py#L64): a privacy call on a non-privacy case fails the tool-sequence check. [Line 68](../regression/test_checkers_pass_oracle_and_fail_null.py#L68), `test_privacy_axis_must_match_the_request`: a privacy verdict on `clean_synthetic` fails; a missing one on `identical_privacy` fails; the oracle passes both.
- `test_expected_matches_tool_output.py` now covers eight cases and asserts the calibration facts the note relies on: the copy at 1.0 / 100% exact, the honest control between 0.35 and 0.65 with no exact matches, and εANY above 0.5 on the honest control (the reason it is not thresholded).

[tests/test_tools.py:113-184](../tests/test_tools.py#L113-L184), one test per property:

| test | property |
|---|---|
| `test_identifiability_exact_copy_is_maximal` | copy: identifiability 1.0, εANY 1.0, exact matches 100%, median distance 0, column split |
| `test_identifiability_honest_sample_sits_near_one_half` | two independent draws of 1500 rows: identifiability in (0.4, 0.6), no exact matches |
| `test_identifiability_far_away_sample_is_zero` | a sample fifty standard deviations away: both fractions 0 |
| `test_heom_distance_units` | a category mismatch is 1, half the range is 0.5, read off the median distance |
| `test_identifiability_k_and_excluded_columns_and_missing` | `exclude_columns` removes an id column, NaN rows are dropped and counted, k = 3 widens radii, k = 1 raises |
| `test_identifiability_size_matched_subsample` | 900 synthetic rows against 300 real: 300 used for identifiability |

## 6. The API

[api.py:297](../src/api.py#L297): `assess_privacy` is a form field, default false. [Lines 315-316](../src/api.py#L315-L316) append the privacy sentence to the prompt when it is true, and the `Verdict` schema has an optional `privacy` field. `test_assess_privacy_flag_changes_the_prompt` in `tests/test_api.py` checks the logged prompt with and without the flag.

## 7. The figures

[figures/make_privacy_figures.py](figures/make_privacy_figures.py) calls the real tool on the golden real data: a jitter sweep for Figure 3, a copy-fraction sweep for Figure 4, and a size-ratio sweep for Figure 5 that also recomputes identifiability *without* the subsample by calling `_heom_encode` directly, to show what the subsample prevents. Figures 1 and 2 are diagrams. Running it takes about a minute and no key.

## 8. If you have ten minutes to explain the whole thing

1. Open [tools.py:270](../src/tools.py#L270). "One over root two: that is what makes a category mismatch cost exactly one."
2. Open [tools.py:346](../src/tools.py#L346). "Every real record gets a radius: how close its nearest neighbour is. Everything else compares against that."
3. Open [tools.py:359](../src/tools.py#L359) and [366](../src/tools.py#L366). "Same radii, two questions: how many real records have a synthetic point inside their radius; how many synthetic points are inside any radius."
4. Open Figure 3. "Honest re-draw: 0.49 and 0.67. One of those has a reference level; that one gets the threshold."
5. Open [agent.py:86](../src/agent.py#L86). "The prompt says exactly that, and says to call the tool only when asked."
6. Open [verdict.py:139](../evals/checks/verdict.py#L139). "The checker fails a privacy verdict nobody asked for, and a missing one somebody did."
7. Open `evals/cases/identical_privacy/expected.json`. "Perfect fidelity, perfect utility, poor overall. That is the whole reason for the phase."

## 9. Exercises

1. Set `k=5` and rerun the jitter sweep. Which way does the honest-sample identifiability move, and why does the 0.5 argument no longer hold exactly?
2. Replace the 1/√2 scale with 1 in `_heom_encode` and rerun the unit tests. Which test fails, and what distance does a category mismatch now cost?
3. Add a `duplicates_in_real` count to the result and a test with a real file that has duplicate rows. Decide what identifiability should report for them.
4. Run the tool on the 19-row bundled sample. Explain the number you get for identifiability in terms of how many real rows there are.
5. Write a third privacy golden case: the honest re-draw with 10% of rows replaced by real rows. Write `expected.json` by hand first, then check it against the derived verdict.

## 10. Gotchas met while building

- pandas 3 text columns are dtype `str`, not `object`; `select_dtypes(include=["string"])` is the portable check.
- εANY on an honest sample is about 0.67, not near zero. If a threshold on it had been copied from the thesis filter, the honest control would have failed its own golden case.
- More synthetic rows lower the nearest-synthetic distance for free; without the size-matched subsample an honest sample reads 0.66 at twice the real size and 0.78 at four times, over the threshold. A first version of Figure 5 built the larger sets by resampling the same rows and showed no effect, which was the experiment being wrong, not the claim.
- Strict `<` in both rules means duplicate real rows are never "inside" anything; exact matches are the fallback.
- The faithfulness checker needs 0.5 and 1.0 in the allowed set when the model explains the reference levels; putting them in the tool description, as with the JS range in phase 1, is the honest fix.
