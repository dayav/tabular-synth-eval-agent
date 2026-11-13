# Eval summary: claude-sonnet-5-5

Run 2026-11-11T03:05:36+00:00 on 8 cases x 3 runs; 24 scored, 0 plumbing failures . Served: claude-sonnet-5-5. Judge: claude-opus-5-5 (rubric 6bb2ff45a47d).

| model | tool-selection | numeric faithfulness | runs with no invented number | verdict | all three checks | clarity (1-5) | agent tokens in / out |
|---|---|---|---|---|---|---|---|
| claude-sonnet-5-5 | 100% (24/24) | 100% (436/436) | 100% (24/24) | 100% (24/24) | 100% | 4.79 | 259k / 37k |

Noise floor on a pass rate at n=24: about +/-20 points.

## Per case

| case | n | tool-selection | no invented number | verdict | ungrounded numbers |
|---|---|---|---|---|---|
| age_x10 | 3 | 3/3 | 3/3 | 3/3 | - |
| clean_privacy | 3 | 3/3 | 3/3 | 3/3 | - |
| clean_synthetic | 3 | 3/3 | 3/3 | 3/3 | - |
| identical | 3 | 3/3 | 3/3 | 3/3 | - |
| identical_privacy | 3 | 3/3 | 3/3 | 3/3 | - |
| income_shuffled | 3 | 3/3 | 3/3 | 3/3 | - |
| phd_missing | 3 | 3/3 | 3/3 | 3/3 | - |
| score_leaks_target | 3 | 3/3 | 3/3 | 3/3 | - |
