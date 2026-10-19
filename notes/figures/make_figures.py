"""Generate the figures for notes/eval_harness.md from the real golden cases and
real tool output. No API key is needed: ``make figures``.

Colour roles (one palette throughout): blue = deterministic code / real data,
orange = the stochastic model / synthetic data, grey = neutral furniture, and the
reserved status red marks a cell the agent must flag (always with a text marker,
never colour alone).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evals.cases import load_cases  # noqa: E402
from evals.checks.verdict import COHEN_D_THRESHOLD, JS_THRESHOLD, UTILITY_GAP_THRESHOLD  # noqa: E402
from tools import cohen_d_numerical, js_divergence_categorical, tstr_xgboost  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
BLUE_LIGHT = "#cde2fb"
BLUE_MID = "#86b6ef"
ORANGE = "#eb6834"
ORANGE_LIGHT = "#fbd9cb"
AQUA = "#1baf7a"
FLAG = "#d03b3b"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

CASE_ORDER = ["identical", "clean_synthetic", "income_shuffled", "score_leaks_target", "age_x10", "phd_missing"]
CASE_LABEL = {
    "identical": "identical copy",
    "clean_synthetic": "clean synthetic",
    "income_shuffled": "income shuffled",
    "score_leaks_target": "score leaks target",
    "age_x10": "age x10",
    "phd_missing": "phd level missing",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    out = HERE / name
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def box(ax, xy, w, h, text, fc, ec, *, fontsize=9.5, weight="normal", color=INK, radius=0.04):
    patch = FancyBboxPatch(
        xy, w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=fc, ec=ec, lw=1.4
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, color=color, wrap=True)


def arrow(ax, a, b, *, color=INK2, lw=1.4, style="-|>", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}")
    )


# ----------------------------------------------------------------------------- data
def tool_outputs() -> dict[str, dict]:
    out = {}
    for c in load_cases():
        r, s = str(c.real_path), str(c.synth_path)
        out[c.name] = {
            "case": c,
            "js": js_divergence_categorical(r, s),
            "d": cohen_d_numerical(r, s),
            "tstr": tstr_xgboost(r, s, c.target_column),
        }
    return out


# ----------------------------------------------------------------------------- figures
def fig1_agent_loop() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, (0.2, 2.6), 1.6, 1.0, "prompt\n(paths + target)", BLUE_LIGHT, BLUE)
    box(ax, (2.6, 2.3), 2.2, 1.6, "model\n(stochastic: different\ntext every run)", ORANGE_LIGHT, ORANGE, weight="semibold")
    ys = (3.9, 2.7, 1.5)
    for y, name in zip(ys, ("js_divergence_categorical", "cohen_d_numerical", "tstr_xgboost"), strict=True):
        box(ax, (6.6, y), 2.6, 0.85, name, BLUE_LIGHT, BLUE, fontsize=8.5)
    box(ax, (2.75, 0.3), 1.9, 1.1, "report\n+ verdict block", ORANGE_LIGHT, ORANGE)
    arrow(ax, (1.8, 3.1), (2.6, 3.1))
    for y in ys:
        arrow(ax, (4.8, 3.3), (6.6, y + 0.55), color=ORANGE)
        arrow(ax, (6.6, y + 0.3), (4.8, 2.9), color=BLUE)
    arrow(ax, (3.7, 2.3), (3.7, 1.4), color=ORANGE)
    ax.text(7.9, 0.95, "tool calls (orange) and JSON results (blue)\ngo round the loop until the model stops asking", ha="center", fontsize=8.5, color=INK2)
    ax.text(7.9, 4.95, "deterministic code: ordinary unit tests", ha="center", fontsize=8.5, color=BLUE, weight="semibold")
    ax.text(3.7, 4.15, "the model: properties of the output, pass rates", ha="center", fontsize=8.5, color=ORANGE, weight="semibold")
    ax.set_title("Figure 1. The agent loop: only the blue part is ordinary code", loc="left")
    save(fig, "fig1_agent_loop.png")


def metric_table(outs: dict) -> pd.DataFrame:
    rows = []
    for name in CASE_ORDER:
        o = outs[name]
        rows.append(
            {
                "case": CASE_LABEL[name],
                "JS gender": o["js"]["per_column"]["gender"],
                "JS education": o["js"]["per_column"]["education"],
                "JS city": o["js"]["per_column"]["city"],
                "d age": o["d"]["per_column"]["age"],
                "d income": o["d"]["per_column"]["income"],
                "d score": o["d"]["per_column"]["score"],
                "TSTR": o["tstr"]["tstr_accuracy"],
                "TRTR": o["tstr"]["trtr_accuracy"],
                "gap": o["tstr"]["utility_gap"],
            }
        )
    return pd.DataFrame(rows).set_index("case")


THRESH = {"JS": JS_THRESHOLD, "d": COHEN_D_THRESHOLD, "gap": UTILITY_GAP_THRESHOLD}


def _ratio(col: str, v: float) -> float:
    key = "gap" if col == "gap" else col.split()[0]
    t = THRESH[key]
    return max(v, 0.0) / t


def fig2_tool_output_heatmap(outs: dict) -> None:
    df = metric_table(outs).drop(columns=["TSTR", "TRTR"])
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    n_rows, n_cols = df.shape
    for i in range(n_rows):
        for j, col in enumerate(df.columns):
            v = float(df.iloc[i, j])
            r = _ratio(col, v)
            step = 0 if r < 0.25 else 1 if r < 0.6 else 2 if r < 1.0 else 3 if r < 2 else 4 if r < 5 else 5
            fc = SEQ[step]
            ax.add_patch(plt.Rectangle((j, n_rows - 1 - i), 1, 1, fc=fc, ec=SURFACE, lw=2))
            flagged = r > 1.0
            txt = f"{v:+.3f}" if col == "gap" else f"{v:.3f}"
            if flagged:
                txt = "⚑ " + txt
            ax.text(j + 0.5, n_rows - 1 - i + 0.5, txt, ha="center", va="center", fontsize=8.5,
                    color=SURFACE if step >= 3 else INK, weight="semibold" if flagged else "normal")
            if flagged:
                ax.add_patch(plt.Rectangle((j + 0.04, n_rows - 1 - i + 0.04), 0.92, 0.92, fill=False, ec=FLAG, lw=2))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(df.columns, fontsize=9)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(list(df.index)[::-1], fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        f"Figure 2. What the tools return on each case (⚑ = above threshold: JS > {JS_THRESHOLD}, d > {COHEN_D_THRESHOLD}, gap > {UTILITY_GAP_THRESHOLD})",
        loc="left", fontsize=10.5,
    )
    ax.text(0, -0.9, "Shade = value relative to its threshold (one blue ramp, light to dark). The shuffled and leaky rows look exactly like the clean control to the fidelity tools.",
            fontsize=8.5, color=INK2)
    save(fig, "fig2_tool_output_heatmap.png")


def fig3_shuffle(outs: dict) -> None:
    real = pd.read_csv(outs["income_shuffled"]["case"].real_path)
    shuf = pd.read_csv(outs["income_shuffled"]["case"].synth_path)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    bins = np.linspace(min(real.income.min(), shuf.income.min()), max(real.income.max(), shuf.income.max()), 30)
    a.hist(real.income, bins=bins, color=BLUE, alpha=0.85, label="real", histtype="stepfilled")
    a.hist(shuf.income, bins=bins, color=ORANGE, histtype="step", lw=2, label="shuffled synthetic")
    a.set_title("Marginal: the two histograms coincide")
    a.set_xlabel("income")
    a.set_ylabel("rows")
    a.legend(loc="upper right")
    d_inc = outs["income_shuffled"]["d"]["per_column"]["income"]
    a.text(0.02, 0.95, f"Cohen's d = {d_inc:.3f}", transform=a.transAxes, va="top", fontsize=9, color=INK2)

    def share_by_decile(df: pd.DataFrame) -> pd.Series:
        dec = pd.qcut(df.income.rank(method="first"), 10, labels=False)
        return df.groupby(dec).target.mean()

    rs, ss = share_by_decile(real), share_by_decile(shuf)
    x = np.arange(1, 11)
    b.plot(x, rs.values, color=BLUE, lw=2, marker="o", ms=5, label="real")
    b.plot(x, ss.values, color=ORANGE, lw=2, marker="o", ms=5, label="shuffled synthetic")
    b.set_ylim(0, 1)
    b.set_xticks(x)
    b.set_xlabel("income decile")
    b.set_ylabel("share with target = 1")
    b.set_title("Joint: the relationship with the label is gone")
    b.grid(axis="y")
    b.legend(loc="upper left")
    ts = outs["income_shuffled"]["tstr"]
    b.text(0.98, 0.05, f"TSTR {ts['tstr_accuracy']:.2f} vs TRTR {ts['trtr_accuracy']:.2f}", transform=b.transAxes, ha="right", fontsize=9, color=INK2)
    fig.suptitle("Figure 3. A permutation keeps every marginal and destroys the joint distribution", x=0.01, y=1.04, ha="left", fontsize=11, weight="semibold")
    save(fig, "fig3_shuffle.png")


def fig4_leak(outs: dict) -> None:
    real = pd.read_csv(outs["score_leaks_target"]["case"].real_path)
    leak = pd.read_csv(outs["score_leaks_target"]["case"].synth_path)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=False)
    bins = np.linspace(35, 105, 36)
    for ax, df, title in ((a, real, "real: classes overlap"), (b, leak, "leaky synthetic: classes separate perfectly")):
        ax.hist(df.score[df.target == 0], bins=bins, color=BLUE_MID, alpha=0.9, label="target = 0", histtype="stepfilled")
        ax.hist(df.score[df.target == 1], bins=bins, color=BLUE, alpha=0.75, label="target = 1", histtype="stepfilled")
        ax.set_title(title)
        ax.set_xlabel("score")
        ax.text(0.02, 0.95, f"mean {df.score.mean():.1f}, sd {df.score.std():.1f}", transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    a.set_ylabel("rows")
    a.legend(loc="upper right")
    d_score = outs["score_leaks_target"]["d"]["per_column"]["score"]
    ts = outs["score_leaks_target"]["tstr"]
    b.text(0.98, 0.80, f"Cohen's d vs real = {d_score:.3f}\nutility gap = {ts['utility_gap']:+.3f}", transform=b.transAxes, ha="right", va="top", fontsize=9, color=INK2)
    fig.suptitle("Figure 4. Target leakage: same mean, same spread, a different shape", x=0.01, y=1.04, ha="left", fontsize=11, weight="semibold")
    save(fig, "fig4_leak.png")


def fig5_expected_map(outs: dict) -> None:
    cols = ["JS gender", "JS education", "JS city", "d age", "d income", "d score", "gap"]
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    n = len(CASE_ORDER)
    for i, name in enumerate(CASE_ORDER):
        exp = outs[name]["case"].expected
        flagged = set(exp["flagged_columns"])
        y = n - 1 - i
        for j, col in enumerate(cols):
            is_flag = (col == "gap" and exp["verdict"]["utility"] == "poor") or (col.split()[1] in flagged if col != "gap" else False)
            fc = "#f8d7d7" if is_flag else "#f1f0ec"
            ax.add_patch(plt.Rectangle((j, y), 1, 1, fc=fc, ec=SURFACE, lw=2))
            if is_flag:
                ax.text(j + 0.5, y + 0.5, "⚑ must flag", ha="center", va="center", fontsize=8, color=FLAG, weight="semibold")
            else:
                ax.text(j + 0.5, y + 0.5, "must not flag", ha="center", va="center", fontsize=7.5, color=MUTED)
        v = exp["verdict"]
        ax.text(len(cols) + 0.15, y + 0.5, f"{v['categorical']} / {v['numerical']} / {v['utility']}  →  {v['overall']}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, len(cols) + 3.6)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels([CASE_LABEL[c] for c in CASE_ORDER][::-1], fontsize=9)
    ax.text(len(cols) + 0.15, n + 0.15, "expected verdict (categorical / numerical / utility → overall)", fontsize=8.5, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Figure 5. The expected-answer map: one row per expected.json, written from the construction", loc="left", fontsize=10.5, pad=18)
    ax.text(0, -0.9, "A flag on the gap cell means utility must be called poor. On the shuffled and leaky rows no column may be flagged: the cause is not visible to the tools.", fontsize=8.5, color=INK2)
    save(fig, "fig5_expected_map.png")


def fig6_numeric_faithfulness() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, (0.2, 2.9), 2.6, 1.8, "ALLOWED SET\ntool results: 0.787, 0.992, −0.205,\n0.17429, 400, 0.043233 ...\nprompts: 0.1, 0.2, 0.05, 80, 20", BLUE_LIGHT, BLUE, fontsize=8.5)
    box(ax, (0.2, 0.4), 2.6, 1.8, "REPORT\nevery number, after stripping\nlist markers, numbered headings,\ntable rules, the verdict block", ORANGE_LIGHT, ORANGE, fontsize=8.5)
    box(ax, (3.4, 1.65), 2.4, 1.9, "GROUNDING RULE\n≥ 2 decimals: an allowed value\nrounds OR truncates to it\n\n< 2 decimals: exact match only\n\nsign ignored", "#f1f0ec", MUTED, fontsize=8.5)
    arrow(ax, (2.8, 3.6), (3.4, 2.9))
    arrow(ax, (2.8, 1.3), (3.4, 2.1))
    examples = [
        ("0.79", "rounds from 0.787", True),
        ("0.78", "truncates from 0.787", True),
        ("79%", "= 0.79, two decimals", True),
        ("0.21", "rounds from −0.205", True),
        ("0.3", "one decimal: nothing equals 0.3", False),
        ("0.0433", "recomputed mean; tool says 0.043233", False),
        ("10", "'factor of 10': in no tool result", False),
    ]
    y0 = 4.55
    for k, (num, why, ok) in enumerate(examples):
        y = y0 - k * 0.62
        c = AQUA if ok else FLAG
        ax.text(6.3, y, "✓" if ok else "✗", color=c, fontsize=12, weight="bold", va="center")
        ax.text(6.65, y, num, fontsize=10, weight="semibold", va="center", family="monospace")
        ax.text(7.55, y, why, fontsize=8.5, color=INK2, va="center")
    arrow(ax, (5.8, 2.6), (6.2, 2.6))
    ax.set_title("Figure 6. Numeric faithfulness: a number is grounded only if the model could have read it", loc="left")
    save(fig, "fig6_numeric_faithfulness.png")


def fig7_evals_vs_regression() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 5)
    ax.axis("off")
    box(ax, (0.3, 0.4), 4.4, 4.3, "", ORANGE_LIGHT, ORANGE)
    box(ax, (5.3, 0.4), 4.4, 4.3, "", BLUE_LIGHT, BLUE)
    ax.text(2.5, 4.3, "evals/  (forward)", ha="center", fontsize=11, weight="semibold", color=ORANGE)
    ax.text(7.5, 4.3, "regression/  (backward)", ha="center", fontsize=11, weight="semibold", color=BLUE)
    left = [
        "question: how good is the agent now?",
        "calls the live model, N times per case",
        "stochastic: scored as pass rates",
        "runs when prompt, tools or model change",
        "costs API calls and minutes",
        "output: evals/results/<run>/summary.json",
    ]
    right = [
        "question: did a known failure come back?",
        "never calls the model",
        "deterministic: plain pytest pass/fail",
        "runs on every commit, about six seconds",
        "free",
        "input: regression/fixtures/<transcript>.json",
    ]
    for k, (l_, r_) in enumerate(zip(left, right, strict=True)):
        y = 3.6 - k * 0.55
        ax.text(0.6, y, "• " + l_, fontsize=9, va="center")
        ax.text(5.6, y, "• " + r_, fontsize=9, va="center")
    arrow(ax, (4.7, 0.7), (5.3, 0.7), color=INK2, lw=1.8)
    ax.text(5.0, 0.12, "record a failing transcript, replay it through the checkers", ha="center", fontsize=8, color=INK2)
    ax.set_title("Figure 7. Evals measure the model; regression tests pin the checkers and the pipeline", loc="left")
    save(fig, "fig7_evals_vs_regression.png")


def fig8_noise_floor() -> None:
    n = np.arange(6, 241)
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.plot(n, 100 / np.sqrt(n), color=BLUE, lw=2, label="1/√n  (upper bound, p near 0.5)")
    ax.plot(n, 100 * 2 * np.sqrt(0.9 * 0.1 / n), color=ORANGE, lw=2, label="2·√(p(1−p)/n) at p = 0.9")
    for k, lab in ((18, "6 cases x 3 runs"), (60, "6 cases x 10 runs"), (180, "6 cases x 30 runs")):
        ax.axvline(k, color=GRID, lw=1)
        ax.text(k + 2, 100 / math.sqrt(k) + 2, f"{lab}\n±{100 / math.sqrt(k):.0f} pts", fontsize=8, color=INK2, va="bottom")
    ax.set_xlabel("number of scored runs n")
    ax.set_ylabel("half-width of a pass-rate interval (points)")
    ax.set_ylim(0, 45)
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    ax.set_title("Figure 8. Resolution of a pass rate: a difference smaller than the half-width is noise", loc="left")
    save(fig, "fig8_noise_floor.png")


def fig9_judge() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    box(ax, (0.2, 1.4), 1.9, 1.2, "agent model\n(under test)", ORANGE_LIGHT, ORANGE)
    box(ax, (2.9, 1.4), 1.9, 1.2, "<report>\n...text...\n</report>", "#f1f0ec", MUTED, fontsize=8.5)
    box(ax, (5.6, 1.4), 1.9, 1.2, "judge model\n(a different model)", BLUE_LIGHT, BLUE)
    box(ax, (8.2, 1.4), 1.6, 1.2, "5 scores, 1–5\n+ rationale", BLUE_LIGHT, BLUE, fontsize=8.5)
    arrow(ax, (2.1, 2.0), (2.9, 2.0))
    arrow(ax, (4.8, 2.0), (5.6, 2.0))
    arrow(ax, (7.5, 2.0), (8.2, 2.0))
    box(ax, (5.3, 3.0), 2.5, 0.75, "fixed rubric, hashed:\nscores from different rubric\nversions are never compared", SURFACE, GRID, fontsize=7.5, color=INK2)
    arrow(ax, (6.55, 3.0), (6.55, 2.6), color=MUTED)
    box(ax, (2.4, 0.2), 2.9, 0.75, "tool results go in as facts;\nreport is data, not instructions;\nlength is not rewarded", SURFACE, GRID, fontsize=7.5, color=INK2)
    arrow(ax, (3.85, 0.95), (3.85, 1.4), color=MUTED)
    ax.text(0.2, 0.45, "the score ranks models;\nit is never a gate", fontsize=8, color=INK2)
    ax.set_title("Figure 9. LLM-as-judge for clarity, with the biases it is built to resist", loc="left")
    save(fig, "fig9_judge.png")


def fig10_identical_overlap(outs: dict) -> None:
    ts = outs["identical"]["tstr"]
    fig, ax = plt.subplots(figsize=(9.6, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.3, 2.3), 6.4, 0.8, fc=BLUE_LIGHT, ec=BLUE, lw=1.4))
    ax.add_patch(plt.Rectangle((6.7, 2.3), 1.6, 0.8, fc=BLUE, ec=BLUE, lw=1.4))
    ax.text(3.5, 2.7, "real: 80% train (TRTR model learns from these)", ha="center", va="center", fontsize=9)
    ax.text(7.5, 2.7, "20% held out", ha="center", va="center", fontsize=9, color=SURFACE, weight="semibold")
    ax.add_patch(plt.Rectangle((0.3, 0.9), 8.0, 0.8, fc=ORANGE_LIGHT, ec=ORANGE, lw=1.4))
    ax.text(4.3, 1.3, "synthetic = exact copy of real: the TSTR model trains on all rows, including the held-out 20%", ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((6.7, 0.9), 1.6, 0.8, fill=False, ec=FLAG, lw=2, ls="--"))
    arrow(ax, (7.5, 1.7), (7.5, 2.3), color=FLAG)
    ax.text(8.5, 2.7, f"TRTR {ts['trtr_accuracy']:.3f}", fontsize=9.5, va="center", weight="semibold")
    ax.text(8.5, 1.3, f"TSTR {ts['tstr_accuracy']:.3f}", fontsize=9.5, va="center", weight="semibold")
    ax.text(0.3, 0.3, f"gap = TRTR − TSTR = {ts['utility_gap']:+.3f}: the synthetic-trained model has already seen the test rows. A gap at or below zero is good utility; a strongly negative one is a memorisation signature.", fontsize=8.5, color=INK2)
    ax.set_title("Figure 10. Why the identical case has a negative utility gap", loc="left")
    save(fig, "fig10_identical_overlap.png")


def main() -> None:
    outs = tool_outputs()
    fig1_agent_loop()
    fig2_tool_output_heatmap(outs)
    fig3_shuffle(outs)
    fig4_leak(outs)
    fig5_expected_map(outs)
    fig6_numeric_faithfulness()
    fig7_evals_vs_regression()
    fig8_noise_floor()
    fig9_judge()
    fig10_identical_overlap(outs)
    metric_table(outs).round(3).to_csv(HERE / "tool_output_table.csv")
    print("wrote notes/figures/tool_output_table.csv")


if __name__ == "__main__":
    main()
