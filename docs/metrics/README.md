# Metric guidance

One page per metric the agent can compute. Each page has YAML front matter with the
machine-readable threshold and a fixed set of sections: Definition, Threshold,
Interpretation, Caveats, Citation. The agent's `lookup_metric_guidance` tool retrieves
sections from these pages, and the agent is instructed to take every threshold from them.
The eval harness reads the same front matter, so the answer key and the agent cannot
disagree about a threshold without a test failing.

| page | tool | threshold |
|---|---|---|
| [js_distance.md](js_distance.md) | `js_divergence_categorical` | flag a column above 0.1 |
| [cohens_d.md](cohens_d.md) | `cohen_d_numerical` | flag a column above 0.2 |
| [utility_gap.md](utility_gap.md) | `tstr_xgboost` | utility poor above 0.05 |
| [identifiability.md](identifiability.md) | `identifiability_heom` | privacy poor above 0.7, or above 0.05 exact matches |
| [verdict_rules.md](verdict_rules.md) | (the agent) | how the axes combine |

Editing a threshold here changes the agent's behaviour and the eval harness's expected
answers together. Regenerate the golden keys (`make cases`) and run `make regression`
after any change.
