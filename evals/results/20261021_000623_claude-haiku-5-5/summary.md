# Eval summary: claude-haiku-5-5

Run 2026-10-21T00:06:43+00:00 on 3 cases x 2 runs; 6 scored, 0 plumbing failures . Served: claude-haiku-5-5. Judge: claude-opus-5-5 (rubric 6bb2ff45a47d).

| model | tool-selection | numeric faithfulness | runs with no invented number | verdict | all three checks | clarity (1-5) | agent tokens in / out |
|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | 100% (6/6) | 100% (106/106) | 100% (6/6) | 100% (6/6) | 100% | 4.50 | 44k / 8k |

Noise floor on a pass rate at n=6: about +/-41 points.

## Per case

| case | n | tool-selection | no invented number | verdict | ungrounded numbers |
|---|---|---|---|---|---|
| clean_privacy | 2 | 2/2 | 2/2 | 2/2 | - |
| clean_synthetic | 2 | 2/2 | 2/2 | 2/2 | - |
| identical_privacy | 2 | 2/2 | 2/2 | 2/2 | - |
