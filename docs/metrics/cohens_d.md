---
metric: cohens_d
title: Cohen's d
tool: cohen_d_numerical
axis: numerical
threshold: 0.2
direction: flag_above
---

# Cohen's d (numerical fidelity)

## Definition

For one numerical column, Cohen's d is the absolute difference between the real mean and
the synthetic mean divided by the pooled standard deviation of the two samples. It is a
standardised effect size: a value of 1 means the means differ by one standard deviation.
It is undefined (reported as null) when the pooled standard deviation is zero.

## Threshold

A numerical column is flagged when its Cohen's d is above 0.2. The numerical fidelity axis
is poor when any column is flagged. The 0.2 value is Cohen's conventional boundary for a
small effect; 0.5 is medium and 0.8 is large, but only 0.2 is used as a flag here.

## Interpretation

Values under 0.1 are indistinguishable from sampling noise at a few thousand rows. A value
of 0.2 means the synthetic column's centre has drifted by a fifth of the spread. A value of
5 (for example a column multiplied by ten) means the two columns barely overlap.

## Caveats

Cohen's d compares means and spreads only. Two columns with the same mean and standard
deviation but different shapes (for example a bimodal synthetic column against a normal
real column, or a column that perfectly separates the classes) score near 0. A good
Cohen's d therefore never proves that a column is right; a bad one proves it is wrong. The
label column, if numeric, appears in the output: it is a label, not a feature, and must not
be flagged.

## Citation

Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
Lawrence Erlbaum Associates. Used as the numerical fidelity metric in Yavo (2026),
arXiv:2602.06390.
