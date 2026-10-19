# Eval summary: claude-sonnet-5-5

Run 2026-10-18T23:54:15+00:00 on 6 cases x 3 runs; 18 scored, 0 plumbing failures . Served: claude-sonnet-5-5. Judge: claude-opus-5-5 (rubric 6bb2ff45a47d).

| model | tool-selection | numeric faithfulness | runs with no invented number | verdict | all three checks | clarity (1-5) | agent tokens in / out |
|---|---|---|---|---|---|---|---|
| claude-sonnet-5-5 | 100% (18/18) | 100% (299/299) | 100% (18/18) | 100% (18/18) | 100% | 4.74 | 95k / 19k |

Noise floor on a pass rate at n=18: about +/-24 points.

## Per case

| case | n | tool-selection | no invented number | verdict | ungrounded numbers |
|---|---|---|---|---|---|
| age_x10 | 3 | 3/3 | 3/3 | 3/3 | - |
| clean_synthetic | 3 | 3/3 | 3/3 | 3/3 | - |
| identical | 3 | 3/3 | 3/3 | 3/3 | - |
| income_shuffled | 3 | 3/3 | 3/3 | 3/3 | - |
| phd_missing | 3 | 3/3 | 3/3 | 3/3 | - |
| score_leaks_target | 3 | 3/3 | 3/3 | 3/3 | - |
