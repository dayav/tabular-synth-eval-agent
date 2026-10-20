# Run logs

Every `run_agent` call writes one folder here (pass `log_dir=None` to opt out):

```
runs/<YYYYMMDD_HHMMSS>_<id>/
  meta.json    model requested and served, prompt, input files (path, bytes, sha256,
               rows, columns), target column, status, timings, git revision
  trace.json   one entry per API step: latency, tokens, request id, stop reason,
               and the tool calls that step triggered with their own latency and result
  usage.json   token totals, API vs tool latency, and the estimated cost with the
               pricing date it rests on
  report.md    the report the agent returned
```

`python runs/summarize.py` (or `make runs`) prints one row per folder. The folder is
git-ignored except for `example/`, one real run kept as a reference.

Cost figures are estimates from list prices (`src/pricing.py`, dated in every
`usage.json`). Set `PRICING_JSON=/path/to/table.json` to price against a different
rate card.
