# Eval summary: claude-sonnet-5-5

Run 2026-10-21T01:02:50+00:00 on 8 cases x 1 runs; 8 scored, 0 plumbing failures . Served: claude-sonnet-5-5. Judge: claude-opus-5-5 (rubric 6bb2ff45a47d).

| model | tool-selection | numeric faithfulness | runs with no invented number | verdict | all three checks | clarity (1-5) | agent tokens in / out |
|---|---|---|---|---|---|---|---|
| claude-sonnet-5-5 | 100% (8/8) | 100% (143/143) | 100% (8/8) | 100% (8/8) | 100% | 4.83 | 87k / 13k |

Noise floor on a pass rate at n=8: about +/-35 points.

## Per case

| case | n | tool-selection | no invented number | verdict | ungrounded numbers |
|---|---|---|---|---|---|
| age_x10 | 1 | 1/1 | 1/1 | 1/1 | - |
| clean_privacy | 1 | 1/1 | 1/1 | 1/1 | - |
| clean_synthetic | 1 | 1/1 | 1/1 | 1/1 | - |
| identical | 1 | 1/1 | 1/1 | 1/1 | - |
| identical_privacy | 1 | 1/1 | 1/1 | 1/1 | - |
| income_shuffled | 1 | 1/1 | 1/1 | 1/1 | - |
| phd_missing | 1 | 1/1 | 1/1 | 1/1 | - |
| score_leaks_target | 1 | 1/1 | 1/1 | 1/1 | - |
