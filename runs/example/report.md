All three evaluations are complete. Here is the full report.

---

## Synthetic Data Evaluation Report

### Overall Verdict: **Acceptable**

The synthetic data achieves good utility and good numerical fidelity, but **categorical fidelity is poor** due to one flagged column. The overall verdict is therefore **acceptable**.

---

### 1. Categorical Fidelity — ⚠️ POOR

Jensen-Shannon (JS) distance measures how different the category frequency distributions are between real and synthetic data. A score of 0 means identical; values above **0.1** are flagged.

| Column | JS Distance | Flagged? |
|---|---|---|
| **gender** | **0.174** | ✅ Yes — exceeds 0.1 |
| city | 0.044 | No |
| education | 0.000 | No |

**`gender` is flagged** with a JS distance of 0.174, meaning its distribution in the synthetic data diverges noticeably from the real data. The other two categorical columns (`city` and `education`) are well within acceptable range.

---

### 2. Numerical Fidelity — ✅ GOOD

Cohen's d measures how far apart the means of a numerical column are, scaled by spread. Values above **0.2** are flagged.

| Column | Cohen's d | Flagged? |
|---|---|---|
| age | 0.082 | No |
| income | 0.099 | No |
| score | 0.071 | No |
| target | 0.000 | *(label — not evaluated as a feature)* |

All numerical feature columns have a Cohen's d well below 0.2. The synthetic data faithfully reproduces the numerical distributions of the real data.

---

### 3. Predictive Utility — ✅ GOOD

The TSTR (Train on Synthetic, Test on Real) evaluation trains a model on synthetic data and checks if it performs as well on real held-out data as a model trained on real data. A utility gap above **0.05** is poor.

| Metric | Value |
|---|---|
| TSTR Accuracy (trained on synthetic) | 1.000 |
| TRTR Accuracy (trained on real) | 1.000 |
| **Utility Gap** | **0.000** |

The utility gap is exactly 0.0 — the synthetic-trained model performs just as well as the real-trained model. However, a **important caveat**: the test set contains only **4 samples**, which is very small. These accuracy figures may not be reliable indicators of true generalization performance on a larger population.

---

### Summary

| Axis | Result |
|---|---|
| Categorical Fidelity | ⚠️ Poor (`gender` flagged at JS = 0.174) |
| Numerical Fidelity | ✅ Good |
| Utility | ✅ Good (gap = 0.0, but test set is only 4 samples) |
| **Overall** | **Acceptable** |

```json
{"verdict": {"categorical": "poor", "numerical": "good", "utility": "good",
             "overall": "acceptable"},
 "flagged_columns": ["gender"]}
```