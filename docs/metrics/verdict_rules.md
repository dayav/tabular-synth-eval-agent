---
metric: verdict_rules
title: How the axes combine into a verdict
tool: none
axis: overall
---

# Verdict rules

## Definition

The report carries one verdict per axis, good or poor, and one overall verdict, excellent,
acceptable or poor. The axes are categorical fidelity, numerical fidelity, utility and,
only when the request asked about privacy, privacy.

## Threshold

A fidelity axis is poor when any of its columns crosses the documented threshold for its
metric. Utility is poor when the utility gap is above its documented threshold. Privacy,
when assessed, is poor when either documented privacy threshold is crossed. Overall is poor
when utility is poor or privacy is poor; otherwise excellent when every assessed axis is
good, and acceptable when at least one fidelity axis is poor.

## Interpretation

Flagged columns are exactly the columns that cross a fidelity threshold. The label column
is never flagged. A column must not be named as the cause of a utility problem unless its
own tool result crosses a threshold; when utility is poor and no column is flagged, the
cause is not visible in the univariate metrics.

## Caveats

These rules make the verdict mechanical on purpose, so that it can be checked. They do not
replace judgement about what the data is for; a dataset that is acceptable for one
downstream task may be unusable for another.

## Citation

Yavo (2026), arXiv:2602.06390, and this repository's evaluation harness notes.
