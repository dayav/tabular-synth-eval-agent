---
metric: utility_gap
title: Utility gap (TSTR vs TRTR)
tool: tstr_xgboost
axis: utility
threshold: 0.05
direction: poor_above
---

# Utility gap: train on synthetic, test on real

## Definition

Two XGBoost classifiers are trained on the same task: one on the synthetic data (TSTR,
train on synthetic, test on real) and one on 80% of the real data (TRTR, train on real,
test on real). Both are scored by accuracy on the same held-out 20% of the real data. The
utility gap is the TRTR accuracy minus the TSTR accuracy. A positive gap means the
synthetic data trains a worse model than the real data would.

## Threshold

Utility is poor when the utility gap is above 0.05. A gap at or below 0 is good: the
synthetic data lost no utility.

## Interpretation

A gap near 0 means a model trained on the synthetic data is as good on real data as one
trained on real data. A gap of 0.2 on a balanced binary task means the synthetic data has
destroyed most of the usable signal. A clearly negative gap (the synthetic-trained model
beats the real-trained one) is not a bonus: it most often means the synthetic data contains
the held-out real rows, which is a memorisation signature and a reason to assess privacy.

## Caveats

Accuracy on a few hundred test rows carries sampling noise of a few hundredths; a gap of
0.01 is not evidence of anything. Utility can be poor while every fidelity metric is good,
because fidelity is measured one column at a time and utility depends on relationships
between columns: a shuffled column or a leaked label breaks utility without moving any
marginal. In that situation the cause is not visible in the fidelity metrics and no column
should be blamed.

## Citation

Esteban, C., Hyland, S. L., & Ratsch, G. (2017). Real-valued (medical) time series
generation with recurrent conditional GANs. arXiv:1706.02633 (introduces TSTR/TRTR).
Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD 2016. Used
as the utility metric in Yavo (2026), arXiv:2602.06390.
