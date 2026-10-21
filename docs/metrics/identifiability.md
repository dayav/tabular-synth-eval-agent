---
metric: identifiability
title: HEOM nearest-neighbour identifiability
tool: identifiability_heom
axis: privacy
threshold: 0.7
direction: poor_above
exact_match_threshold: 0.05
---

# HEOM nearest-neighbour identifiability (privacy)

## Definition

Both tables are encoded in HEOM space: numeric columns are min-max scaled on the real
data, categorical columns become one-hot blocks scaled so that a category mismatch counts
exactly 1. For every real record, its radius is the HEOM distance to its nearest other
real record (k = 2). Identifiability is the fraction of real records whose nearest
synthetic record is closer than that radius, computed on a random subsample of the
synthetic data the size of the real data. The tool also reports epsilon_any, the fraction
of synthetic rows that fall inside at least one real record's radius, and the fraction of
exact matches, synthetic rows at distance zero from some real row.

## Threshold

Privacy is poor when identifiability is above 0.7, or when the exact match fraction is
above 0.05. epsilon_any has no threshold: it is supporting evidence only.

## Interpretation

An honest synthetic sample of the same size as the real data has identifiability near 0.5,
because its nearest point and the nearest real point are equally likely to be the closer
one. A verbatim copy scores 1.0. A copy with small noise added scores between the two;
on the golden data the 0.7 line is crossed when the noise is below about a third of each
column's standard deviation. epsilon_any for an honest sample is high and data-dependent
(about 0.65 on the golden data), which is why it is reported and never thresholded.
Exact matches catch partial copying that identifiability alone is slow to see.

## Caveats

This is a proximity measure of disclosure risk, not an attribute-inference attack and not
a formal privacy guarantee. The threshold was calibrated on seven-column synthetic data
and should be re-read on real data before anyone relies on it. With fewer synthetic rows
than real rows the honest reference falls below 0.5; the tool reports how many rows it
used. With many one-hot levels nearest-neighbour distances lose contrast.

## Citation

Yoon, J., Drumright, L. N., & van der Schaar, M. (2020). Anonymization through data
synthesis using generative adversarial networks (ADS-GAN). IEEE Journal of Biomedical and
Health Informatics, 24(8), 2378-2388 (the identifiability score). Wilson, D. R., &
Martinez, T. R. (1997). Improved heterogeneous distance functions. Journal of Artificial
Intelligence Research, 6, 1-34 (HEOM). The epsilon_any filter statistic and the HEOM
encoder are from Yavo (2026), arXiv:2602.06390.
