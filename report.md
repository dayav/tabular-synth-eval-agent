# Synthetic Data Evaluation Report

_Generated 2026-06-06 14:42:08_

---

Here's a comprehensive assessment of the synthetic dataset across all three evaluation axes:

---

## 🧪 Synthetic Data Evaluation Report

### 1. 📊 Categorical Fidelity (Jensen-Shannon Distance — lower is better)

| Column | JS Distance | Rating |
|---|---|---|
| `gender` | 0.174 | ⚠️ Moderate |
| `education` | 0.000 | ✅ Perfect |
| `city` | 0.044 | ✅ Good |
| **Mean** | **0.073** | ✅ **Acceptable** |

The overall mean JS distance of **0.073** is just under the 0.1 threshold, which is generally considered acceptable. However, `gender` stands out with a score of **0.174** — the synthetic distribution for this column diverges noticeably from the real one and warrants attention. `education` is perfectly replicated, and `city` is comfortably within the good range.

---

### 2. 🔢 Numerical Fidelity (Cohen's d — lower is better)

| Column | Cohen's d | Rating |
|---|---|---|
| `age` | 0.082 | ✅ Excellent |
| `income` | 0.099 | ✅ Excellent |
| `score` | 0.071 | ✅ Excellent |
| `target` | 0.000 | *(label, not evaluated as feature)* |
| **Mean** | **0.063** | ✅ **Excellent** |

All numerical feature columns show **very small effect sizes**, well below the 0.2 "small effect" threshold. The synthetic data faithfully reproduces the distributions of `age`, `income`, and `score`. The `target` column's Cohen's d of 0.0 is a label artifact and not meaningful for fidelity assessment.

---

### 3. 🤖 Predictive Utility (Train on Synthetic, Test on Real — TSTR)

| Metric | Value |
|---|---|
| TSTR Accuracy (trained on synthetic) | **1.00** |
| TRTR Accuracy (trained on real) | **1.00** |
| **Utility Gap** | **0.00** |
| Test Samples | 4 |
| Classes | 2 |

The utility gap is **0.00**, meaning a model trained on synthetic data performs identically to one trained on real data. This is an excellent result. However, it's worth noting the **test set is very small (4 samples)**, so this result should be interpreted with some caution — the perfect scores may partly reflect the simplicity or limited size of the evaluation split.

---

### 🏁 Overall Verdict

| Axis | Score | Status |
|---|---|---|
| Categorical Fidelity | 0.073 mean JS | ✅ Acceptable (watch `gender`) |
| Numerical Fidelity | 0.063 mean Cohen's d | ✅ Excellent |
| Utility Gap | 0.00 | ✅ Excellent |

The synthetic dataset performs **well overall**, with especially strong numerical fidelity and predictive utility. The one notable concern is the **`gender` column's categorical distribution**, which diverges from the real data more than ideal. If `gender` is an important feature for downstream use cases, the synthetic generation process for that column should be revisited.