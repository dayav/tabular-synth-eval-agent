# Eval summary: claude-sonnet-4-6

Run 2026-11-11T03:04:17+00:00 on 8 cases x 3 runs; 24 scored, 0 plumbing failures . Served: claude-sonnet-4-6. Judge: claude-opus-5-5 (rubric 6bb2ff45a47d).

| model | tool-selection | numeric faithfulness | runs with no invented number | verdict | all three checks | clarity (1-5) | agent tokens in / out |
|---|---|---|---|---|---|---|---|
| claude-sonnet-4-6 | 100% (24/24) | 99% (838/843) | 83% (20/24) | 88% (21/24) | 71% | 4.23 | 247k / 45k |

Noise floor on a pass rate at n=24: about +/-20 points.

## Per case

| case | n | tool-selection | no invented number | verdict | ungrounded numbers |
|---|---|---|---|---|---|
| age_x10 | 3 | 3/3 | 2/3 | 2/3 | 10× |
| clean_privacy | 3 | 3/3 | 3/3 | 3/3 | - |
| clean_synthetic | 3 | 3/3 | 2/3 | 3/3 | 0.0418 |
| identical | 3 | 3/3 | 3/3 | 3/3 | - |
| identical_privacy | 3 | 3/3 | 2/3 | 1/3 | 99.75, 76.75 |
| income_shuffled | 3 | 3/3 | 3/3 | 3/3 | - |
| phd_missing | 3 | 3/3 | 2/3 | 3/3 | 0.018 |
| score_leaks_target | 3 | 3/3 | 3/3 | 3/3 | - |
