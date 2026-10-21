---
metric: js_distance
title: Jensen-Shannon distance
tool: js_divergence_categorical
axis: categorical
threshold: 0.1
direction: flag_above
---

# Jensen-Shannon distance (categorical fidelity)

## Definition

For one categorical column, the Jensen-Shannon distance compares the frequency table of
the real data with the frequency table of the synthetic data. It is the square root of the
Jensen-Shannon divergence, computed with base-2 logarithms, so it is a true metric bounded
between 0 (identical distributions) and 1 (no category in common). Category levels are
taken from the union of both files: a level present in only one file counts as a
difference, not as missing data.

## Threshold

A categorical column is flagged when its Jensen-Shannon distance is above 0.1. The
categorical fidelity axis is poor when any column is flagged.

## Interpretation

Values near 0 mean the synthetic data reproduces the category frequencies. Values around
0.1 correspond to visible but modest shifts in a few levels. Larger values usually mean a
level is missing or badly over-represented; a level removed entirely from a three-level
column gives roughly 0.29 on the golden data. The mean across columns is reported for
convenience but the verdict is per column: one bad column is a bad axis.

## Caveats

The distance looks at one column at a time. It is blind to relationships between columns,
so a synthetic file with every column permuted independently scores a perfect 0 on every
column and is nevertheless useless. The label column is a feature here if it is
categorical; the agent should still report it but not treat it as a fidelity failure on
its own. On very small files (tens of rows) frequency tables are noisy and a distance of
0.1 can arise by chance.

## Citation

Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Transactions on
Information Theory, 37(1), 145-151. Endres, D. M., & Schindelin, J. E. (2003). A new metric
for probability distributions. IEEE Transactions on Information Theory, 49(7), 1858-1860.
Used as the categorical fidelity metric in Yavo (2026), arXiv:2602.06390.
