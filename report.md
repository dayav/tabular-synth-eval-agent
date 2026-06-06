# Synthetic Data Evaluation Report

_Generated 2026-06-03 11:15:14_

---

Here's a full assessment of the synthetic dataset across all three evaluation axes:

---

## 🧪 Synthetic Data Evaluation Report

### 1. 📊 Categorical Fidelity (Jensen-Shannon Divergence) — *Lower is better; < 0.1 is good*

| Column | JS Distance | Rating |
|---|---|---|
| `gender` | 0.174 | ⚠️ Moderate |
| `education` | 0.000 | ✅ Perfect |
| `city` | 0.044 | ✅ Good |
| **Mean** | **0.073** | ✅ Good overall |

**Summary:** The overall mean JS distance of **0.073** is within the acceptable range (< 0.1). However, `gender` stands out with a notably higher divergence of **0.174**, suggesting its distribution in the synthetic data does not closely mirror the real data. `education` is perfectly reproduced, and `city` is well-preserved.

---

### 2. 🔢 Numerical Fidelity (Cohen's d Effect Size) — *Lower is better; < 0.2 = small/negligible effect*

| Column | Cohen's d | Rating |
|---|---|---|
| `age` | 0.082 | ✅ Excellent |
| `income` | 0.099 | ✅ Excellent |
| `score` | 0.071 | ✅ Excellent |
| `target` | 0.000 | *(label — ignore)* |
| **Mean** | **0.063** | ✅ Excellent overall |

**Summary:** All numerical feature columns show Cohen's d values well below **0.2**, indicating negligible distributional differences between real and synthetic data. Numerical fidelity is **excellent**.

---

### 3. 🤖 Predictive Utility (Train-on-Synthetic, Test-on-Real) — *Utility gap near 0 is ideal; < 0.05 is excellent*

| Metric | Value |
|---|---|
| TSTR Accuracy (synthetic) | 1.00 |
| TRTR Accuracy (real baseline) | 1.00 |
| **Utility Gap** | **0.00** |
| Test Samples | 4 |
| Classes | 2 |

**Summary:** The utility gap is **0.00**, meaning a model trained on synthetic data performs just as well as one trained on real data. This is a perfect utility score. ⚠️ *Note: The test set is very small (n=4), so this result should be interpreted with caution — it may not be fully reliable at this sample size.*

---

## 🏁 Overall Verdict

| Axis | Score | Verdict |
|---|---|---|
| Categorical Fidelity | 0.073 mean JS | ✅ Good (watch `gender`) |
| Numerical Fidelity | 0.063 mean Cohen's d | ✅ Excellent |
| Predictive Utility | 0.00 gap | ✅ Excellent (small test set caveat) |

The synthetic dataset is of **good to excellent quality overall**. The main concern is the `gender` column, which shows a moderate categorical distribution mismatch and warrants closer inspection. If `gender` is an important feature in downstream tasks, the synthetic generation process should be reviewed to better preserve its real-world distribution.