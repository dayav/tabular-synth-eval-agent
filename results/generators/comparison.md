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
