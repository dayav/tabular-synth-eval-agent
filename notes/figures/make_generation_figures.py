"""Figures for notes/llm_generation_baseline.md, drawn from results/generators/comparison.json.

    python notes/figures/make_generation_figures.py      (no API key)

Colour roles: blue = CTGAN, aqua = TVAE, orange = the LLM baseline; grey = furniture;
red only for a threshold line, always labelled.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "results" / "generators" / "comparison.json"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, BLUE_LIGHT, AQUA, ORANGE, ORANGE_LIGHT, FLAG, GREY = "#2a78d6", "#cde2fb", "#1baf7a", "#eb6834", "#fbd9cb", "#d03b3b", "#f1f0ec"
COLOURS = {"ctgan": BLUE, "tvae": AQUA, "llm": ORANGE}
LABELS = {"ctgan": "CTGAN", "tvae": "TVAE", "llm": "LLM (marginals only)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})


def save(fig: plt.Figure, name: str) -> None:
    out = HERE / name
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def box(ax, xy, w, h, text, fc, ec, *, fontsize=8.5, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec=ec, lw=1.4))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, color=color)


def arrow(ax, a, b, *, color=INK2, lw=1.4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=lw, color=color))


def load() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def fig1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    box(ax, (0.2, 2.9), 2.0, 1.4, "real reference\nsample (500 rows)", BLUE_LIGHT, BLUE)
    box(ax, (2.8, 2.9), 2.2, 1.4, "schema + marginals\nlevels and frequencies,\nmin/max/mean/sd/quantiles\n(no rows)", GREY, MUTED, fontsize=8)
    box(ax, (5.6, 2.9), 2.0, 1.4, "Claude, structured\noutput: a JSON array\nof rows, 25 per batch", ORANGE_LIGHT, ORANGE, fontsize=8.5)
    box(ax, (8.2, 2.9), 2.6, 1.4, "validate every row\nknown level, numeric type,\ninteger-ness, range ±5%\n-> drop or keep", BLUE_LIGHT, BLUE, fontsize=8)
    arrow(ax, (2.2, 3.6), (2.8, 3.6))
    arrow(ax, (5.0, 3.6), (5.6, 3.6))
    arrow(ax, (7.6, 3.6), (8.2, 3.6))
    arrow(ax, (9.5, 2.9), (9.5, 2.1), color=BLUE)
    box(ax, (8.2, 1.2), 2.6, 0.9, "CSV of N valid rows\n+ counts, rejections, tokens, cost", BLUE_LIGHT, BLUE, fontsize=8)
    arrow(ax, (8.2, 1.65), (7.6, 1.65), color=MUTED)
    box(ax, (5.6, 1.2), 2.0, 0.9, "more batches until N,\nup to 3x the minimum", GREY, MUTED, fontsize=8)
    arrow(ax, (6.6, 2.1), (6.6, 2.9), color=MUTED)
    ax.text(0.2, 1.9, "the model never sees a real row:\nit cannot copy one, and any relationship\nbetween columns in its output is its prior,\nnot the data's", fontsize=8.5, color=INK2, va="top")
    ax.text(0.2, 0.35, "The validator is deterministic code; the model is the only stochastic step. The result dict is the same shape as the other tools' results.", fontsize=8.5, color=INK2)
    ax.set_title("Figure 1. generate_with_llm: marginals in, strict JSON out, validated rows to CSV", loc="left")
    save(fig, "fig_gn1_pipeline.png")


def fig2_what_each_saw() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4)
    ax.axis("off")
    cols = [
        ("CTGAN / TVAE", "trained on the full training set:\n24k to 49k real rows, every column,\nevery relationship between columns", BLUE_LIGHT, BLUE),
        ("LLM baseline", "shown the schema and the marginal\nstatistics of a 500-row sample:\nno rows, no pairwise statistics", ORANGE_LIGHT, ORANGE),
        ("the evaluation", "identical for all three: 500 rows each,\nthe same reference sample, the same\nheld-out test rows, the four tools,\nthe same agent and prompt", GREY, MUTED),
    ]
    for k, (title, body, fc, ec) in enumerate(cols):
        x = 0.3 + k * 3.45
        box(ax, (x, 0.7), 3.2, 2.9, "", fc, ec)
        ax.text(x + 1.6, 3.2, title, ha="center", fontsize=10.5, weight="semibold", color=ec if ec != MUTED else INK)
        ax.text(x + 1.6, 1.9, body, ha="center", va="center", fontsize=8.5)
    ax.text(0.3, 0.25, "The input asymmetry is the point of the baseline: how far do marginals alone get you, measured on the same footing as models that saw everything?", fontsize=8.5, color=INK2)
    ax.set_title("Figure 2. What each generator was given, and what was held equal", loc="left")
    save(fig, "fig_gn2_what_each_saw.png")


def _grouped(ax, data: dict, key: str, ylabel: str, title: str, fmt: str = "{:+.3f}", ref_lines: list[tuple[float, str, str]] | None = None) -> None:
    datasets = [r["dataset"] for r in data["results"]]
    gens = ["ctgan", "tvae", "llm"]
    x = np.arange(len(datasets))
    w = 0.26
    for i, g in enumerate(gens):
        vals = [r["generators"][g]["metrics"][key] for r in data["results"]]
        bars = ax.bar(x + (i - 1) * w, vals, width=w, color=COLOURS[g], label=LABELS[g])
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, (v if v >= 0 else 0) + 0.004, fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK2)
    for y, label, color in ref_lines or []:
        ax.axhline(y, color=color, lw=1.2, ls="--" if color == FLAG else "-")
        ax.text(len(datasets) - 0.55, y + 0.004, label, ha="right", fontsize=8, color=color)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.set_title(title, loc="left")


def fig3_utility(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    _grouped(ax, data, "utility_gap", "utility gap (TRTR − TSTR)", "Figure 3. Utility gap per generator: all nine within ±0.04, none crosses the 0.05 line",
             ref_lines=[(0.05, "poor above 0.05", FLAG)])
    ax.set_ylim(-0.05, 0.08)
    ax.legend(loc="upper left", ncol=3)
    save(fig, "fig_gn3_utility.png")


def fig4_fidelity_flags(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    datasets = [r["dataset"] for r in data["results"]]
    gens = ["ctgan", "tvae", "llm"]
    x = np.arange(len(datasets))
    w = 0.26
    for i, g in enumerate(gens):
        fr, lab = [], []
        for r in data["results"]:
            m = r["generators"][g]["metrics"]
            flagged = m["js_flagged"] + m["d_flagged"]
            total = m["n_categorical"] + m["n_numerical"]
            fr.append(flagged / total)
            lab.append(f"{flagged}/{total}")
        bars = ax.bar(x + (i - 1) * w, fr, width=w, color=COLOURS[g], label=LABELS[g])
        for b, t in zip(bars, lab, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, t, ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("fraction of columns flagged (JS > 0.1 or d > 0.2)")
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y")
    ax.legend(loc="upper right", ncol=3)
    ax.set_title("Figure 4. Univariate fidelity: columns flagged per generator (fewer is better)", loc="left")
    save(fig, "fig_gn4_fidelity_flags.png")


def fig5_identifiability(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    _grouped(ax, data, "identifiability", "identifiability", "Figure 5. Identifiability per generator: all far under 0.7; the LLM, which saw no rows, sits lowest or near it",
             fmt="{:.2f}", ref_lines=[(0.5, "honest-sample reference 0.5", INK2), (0.7, "poor above 0.7", FLAG)])
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper left", ncol=3)
    save(fig, "fig_gn5_identifiability.png")


def main() -> None:
    data = load()
    fig1_pipeline()
    fig2_what_each_saw()
    fig3_utility(data)
    fig4_fidelity_flags(data)
    fig5_identifiability(data)


if __name__ == "__main__":
    main()
