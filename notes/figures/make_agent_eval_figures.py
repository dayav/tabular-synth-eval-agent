"""Figures for notes/agent_evals_explained.md and notes/agent_evals_gap_analysis.md.

    python notes/figures/make_agent_eval_figures.py        (no API key needed)

Each figure redraws one diagram of Anthropic's "Demystifying evals for AI agents"
(2026-01-09) with this repository's own parts, or plots this repository's own eval
results next to the article's argument.

Data comes from committed files only, so a fresh clone can regenerate everything:
``evals/results/*/summary.json`` and ``evals/cases/*/expected.json``. The transcripts
are git-ignored and are deliberately not read here.

Colour roles: the three models are the first three categorical slots, in a fixed
order (blue, orange, aqua). Status colours (good, warning, critical) always come with
a word. Text is always ink, never a series colour.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "evals" / "results"
CASES = ROOT / "evals" / "cases"

SURFACE, WHITE, INK, INK2, MUTED, GRID = "#fcfcfb", "#ffffff", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
GREY = "#f1f0ec"
BLUE, BLUE_LIGHT, BLUE_TINT = "#2a78d6", "#cde2fb", "#e9f2fd"
ORANGE, ORANGE_LIGHT, ORANGE_TINT = "#eb6834", "#fbd9cb", "#fdf0ea"
AQUA, AQUA_TINT = "#1baf7a", "#def5ec"
MAGENTA, MAGENTA_TINT = "#e87ba4", "#fce9f0"
VIOLET, VIOLET_TINT = "#6250d6", "#eeecfb"
GOOD, GOOD_TINT = "#0ca30c", "#e1f4e1"
WARN, WARN_TINT = "#fab219", "#fef2d2"
CRIT, CRIT_TINT = "#d03b3b", "#f9e1e1"
MONO = "DejaVu Sans Mono"
SCALE = 0.78  # inches per drawing unit on the diagram canvases

# Fixed order: colour follows the model, never its rank in a table.
MODEL_ORDER = ["claude-sonnet-4-6", "claude-sonnet-5-5", "claude-haiku-5-5"]
MODEL_COLOR = dict(zip(MODEL_ORDER, [BLUE, ORANGE, AQUA], strict=True))

# The eval summaries carry no agent version, so the version behind each results folder
# is written here by hand from CHANGELOG.md. (That missing stamp is one of the gaps.)
AGENT_VERSION = {
    "20261018_235325": "phase-1 agent: thresholds in the prompt",
    "20261021_000623": "phase-4 agent: privacy tool added",
    "20261021_010217": "current agent: thresholds retrieved from docs",
}

STATUS = {
    "done": (GOOD, GOOD_TINT, "✓ done"),
    "partial": (WARN, WARN_TINT, "◐ partial"),
    "missing": (CRIT, CRIT_TINT, "✗ missing"),
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
        "axes.titleweight": "bold",
        "axes.titlesize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    }
)


# ----------------------------------------------------------------------------- helpers
def save(fig: plt.Figure, name: str) -> None:
    out = HERE / name
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def canvas(w: float, h: float):
    """A blank drawing surface, ``w`` by ``h`` units, filling the whole figure."""
    fig = plt.figure(figsize=(w * SCALE, h * SCALE))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def title(ax, w: float, h: float, text: str) -> None:
    ax.text(0.2, h - 0.25, text, ha="left", va="center", fontsize=11.5, weight="bold", color=INK)


def rect(ax, x, y, w, h, *, fc=SURFACE, ec=INK2, lw=1.3, dashed=False, radius=0.07, z=1):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=fc, ec=ec, lw=lw,
            linestyle=(0, (4, 3)) if dashed else "solid", zorder=z,
        )
    )


def text(ax, x, y, s, *, size=8.5, color=INK, weight="normal", ha="left", va="center", mono=False, z=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, weight=weight, zorder=z,
            family=MONO if mono else None, linespacing=1.35)


def card(ax, x, y, w, h, head, body, *, fc=GREY, ec=INK2, body_mono=False, body_size=8.0, dashed=False, lw=1.3):
    """A box with a bold heading on its first line and a left-aligned body under it."""
    rect(ax, x, y, w, h, fc=fc, ec=ec, dashed=dashed, lw=lw)
    text(ax, x + 0.14, y + h - 0.2, head, size=9.5, weight="bold")
    if body:
        text(ax, x + 0.14, y + h - 0.4, body, size=body_size, color=INK2, va="top", mono=body_mono)


def arrow(ax, a, b, *, color=INK2, lw=1.3, style="-|>", rad=0.0, dashed=False, z=4):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle=style, mutation_scale=11, lw=lw, color=color, zorder=z,
            connectionstyle=f"arc3,rad={rad}", linestyle=(0, (4, 3)) if dashed else "solid",
            shrinkA=0, shrinkB=0,
        )
    )


def chip(ax, x, y, label, *, ec, fc, dashed=False, size=7.5, h=0.3) -> float:
    """A small labelled pill; returns its width so the next one can be placed after it."""
    w = 0.602 * size / 72 / SCALE * len(label) + 0.24  # monospace advance is 0.602 em
    rect(ax, x, y, w, h, fc=fc, ec=ec, lw=1.2, dashed=dashed, radius=0.05, z=3)
    text(ax, x + w / 2, y + h / 2, label, size=size, mono=True, ha="center", color=MUTED if dashed else INK)
    return w


def pill(ax, x, y, status: str, *, w=0.98, h=0.3) -> None:
    """A status marker: the colour never stands alone, the word is always there."""
    ec, fc, label = STATUS[status]
    rect(ax, x, y, w, h, fc=fc, ec=ec, lw=1.2, radius=0.14, z=3)
    text(ax, x + w / 2, y + h / 2, label, size=8, ha="center", weight="bold")


def short(model: str) -> str:
    return model.replace("claude-", "")


# ----------------------------------------------------------------------------- data
def load_groups() -> dict[str, list[dict[str, Any]]]:
    """``{run stamp: [summary, ...]}``; one stamp is one ``make eval`` session over several models."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(RESULTS.glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        s["folder"] = path.parent.name
        groups[path.parent.name[:15]].append(s)
    for stamp in groups:
        groups[stamp].sort(key=lambda s: MODEL_ORDER.index(s["model_requested"]) if s["model_requested"] in MODEL_ORDER else 99)
    return dict(groups)


def most_trials(groups: dict[str, list[dict[str, Any]]]) -> str:
    """The session with the most trials per task (latest wins a tie)."""
    return max(groups, key=lambda g: (min(s["runs_per_case"] for s in groups[g]), g))


def trials_passing_all(per_case: dict[str, Any]) -> int:
    """Trials of one task that passed every check.

    The summary stores passes per check, not per trial, so this is the minimum over the
    three checks: exact whenever failures of different checks do not fall on different trials.
    """
    return min(int(per_case[k]) for k in ("tool_sequence", "numeric_faithfulness", "verdict"))


def lower_bound_all_pass(n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) lower confidence bound on p when n of n trials passed (Clopper-Pearson)."""
    return float(alpha ** (1.0 / n))


# ----------------------------------------------------------------------------- figures
def fig1_single_turn_vs_agent() -> None:
    W, H = 12.0, 7.7
    fig, ax = canvas(W, H)
    title(ax, W, H, "Figure 1. A single-turn eval grades one answer; this agent's eval grades a loop")

    # single-turn row
    rect(ax, 0.2, 5.0, 11.6, 2.1, fc=GREY, ec=GREY, radius=0.15, z=0)
    text(ax, 0.45, 6.05, "Single-turn\neval", size=11, color=INK2, weight="bold")
    card(ax, 2.0, 6.1, 3.0, 0.85, "Prompt", "Is a Cohen's d of {D} above\nthe flagging threshold of 0.2?", fc=WHITE)
    card(ax, 2.0, 5.15, 3.0, 0.75, "Data", "D = 0.31", fc=WHITE, body_mono=True)
    card(ax, 6.3, 5.3, 2.0, 1.5, "LLM", "one call\nno tools\nno state", fc=AQUA_TINT)
    rect(ax, 8.75, 5.82, 1.0, 0.46, fc=WHITE)
    text(ax, 9.25, 6.05, "Response", ha="center", color=INK2)
    card(ax, 10.1, 5.55, 1.55, 1.0, "Grading logic", 'response ==\n"yes"', fc=WHITE, body_mono=True)
    arrow(ax, (5.0, 6.5), (6.3, 6.25))
    arrow(ax, (5.0, 5.52), (6.3, 5.85))
    arrow(ax, (8.3, 6.05), (8.75, 6.05))
    arrow(ax, (9.75, 6.05), (10.1, 6.05))

    # agent row
    rect(ax, 0.2, 0.2, 11.6, 4.6, fc=GREY, ec=GREY, radius=0.15, z=0)
    text(ax, 0.45, 2.5, "This agent's\neval", size=11, color=INK2, weight="bold")
    card(
        ax, 2.0, 3.35, 3.0, 1.3, "Tools",
        "js_divergence_categorical()\ncohen_d_numerical()\ntstr_xgboost()\nidentifiability_heom()\nlookup_metric_guidance()",
        fc=WHITE, body_mono=True, body_size=7.2,
    )
    card(
        ax, 2.0, 1.8, 3.0, 1.35, "Environment",
        "real.csv and synth.csv, copied to a\nfresh temp folder under neutral names\ndocs/metrics index\nall of it read-only",
        fc=WHITE, body_size=7.8,
    )
    card(
        ax, 2.0, 0.4, 3.0, 1.2, "Task",
        "Evaluate the synthetic dataset at\n…/synth.csv against …/real.csv.\nTarget column: 'target'.",
        fc=WHITE, body_size=7.8,
    )
    rect(ax, 6.3, 0.6, 2.75, 3.9, fc=AQUA_TINT)
    text(ax, 6.44, 4.28, "Agent", size=9.5, weight="bold")
    text(ax, 7.15, 4.28, "run_agent()", size=7.5, mono=True, color=INK2)
    text(
        ax, 6.44, 4.02,
        "js_divergence_categorical(…)\ncohen_d_numerical(…)\ntstr_xgboost(…, 'target')\nlookup_metric_guidance(…)\nlookup_metric_guidance(…)\nlookup_metric_guidance(…)\n…",
        size=7.2, mono=True, color=INK2, va="top",
    )
    rect(ax, 6.44, 0.78, 2.47, 0.62, fc=ORANGE_LIGHT, ec=ORANGE_LIGHT, radius=0.05, z=2)
    text(ax, 7.675, 1.09, "final message: report\n+ JSON verdict block", size=7.8, ha="center")
    card(
        ax, 9.5, 2.55, 2.15, 1.45, "Grading logic",
        "tool_sequence\nnumeric_faithfulness\nverdict\njudge (clarity rubric)",
        fc=WHITE, body_mono=True, body_size=7.2,
    )
    text(ax, 10.575, 2.4, "graded on the trajectory\nand the final message", size=7.5, color=INK2, ha="center", va="top")
    arrow(ax, (5.0, 4.0), (6.3, 4.0))
    arrow(ax, (5.0, 2.75), (6.3, 2.75))
    arrow(ax, (5.0, 1.0), (6.3, 1.0))
    arrow(ax, (9.05, 3.3), (9.5, 3.3))
    arrow(ax, (6.3, 2.3), (5.0, 2.3), dashed=True, color=MUTED)
    text(ax, 5.65, 2.14, "no environment\nupdate: the tools\nonly read", size=7, color=MUTED, ha="center", va="top")
    save(fig, "fig_ae1_single_turn_vs_agent.png")


def fig2_components(groups: dict[str, list[dict[str, Any]]]) -> None:
    n_tasks = len([d for d in CASES.iterdir() if (d / "expected.json").exists()])
    W, H = 12.0, 7.1
    fig, ax = canvas(W, H)
    title(ax, W, H, "Figure 2. Components of the evaluation of this agent (dashed = absent here)")

    rect(ax, 0.2, 0.2, 11.6, 6.3, fc=WHITE, radius=0.18, z=0)
    text(ax, 0.45, 6.2, "Evaluation harness", size=11, weight="bold")
    text(ax, 2.9, 6.2, "evals/run_eval.py", size=8, mono=True, color=INK2)

    rect(ax, 0.45, 1.75, 8.85, 4.15, fc=VIOLET_TINT, radius=0.1, z=1)
    text(ax, 0.65, 5.63, "Evaluation suite", size=10.5, weight="bold")
    text(ax, 2.75, 5.63, f"evals/cases/  ·  {n_tasks} tasks", size=8, mono=True, color=INK2)

    # the task card
    rect(ax, 0.65, 1.95, 6.3, 3.4, fc=SURFACE, ec=ORANGE, lw=1.6, z=2)
    text(ax, 0.85, 5.1, "Task", size=10.5, weight="bold")
    text(ax, 1.5, 5.1, "phd_missing  ·  real.csv + synth.csv + expected.json", size=7.2, mono=True, color=INK2)
    text(ax, 0.85, 4.78, "Graders", color=INK2)
    x = 0.85
    for label in ("tool_sequence", "numeric_faithfulness", "verdict", "judge"):
        x += chip(ax, x, 4.3, label, ec=MAGENTA, fc=MAGENTA_TINT, size=7) + 0.08
    chip(ax, x, 4.3, "state_check", ec=MUTED, fc=SURFACE, dashed=True, size=7)
    text(ax, 0.85, 4.02, "Tracked metrics", color=INK2)
    x = 0.85
    for label in ("tokens", "latency", "cost", "n_turns *", "n_toolcalls *"):
        x += chip(ax, x, 3.54, label, ec=BLUE, fc=BLUE_TINT, size=7) + 0.08
    text(ax, x + 0.05, 3.69, "* kept per trial,\n  not in the summary", size=6.8, color=MUTED)
    text(ax, 0.85, 3.26, "Trials", color=INK2)
    for k in (2, 1, 0):
        rect(ax, 0.85 + 0.07 * k, 2.1 + 0.07 * k, 3.5, 0.9, fc=AQUA_TINT, ec=AQUA, lw=1.2, radius=0.05, z=3)
    text(ax, 1.0, 2.76, "Trial #2", size=8.5, weight="bold")
    text(ax, 1.0, 2.5, "transcript", size=7.5, mono=True, color=INK2)
    text(ax, 1.0, 2.28, "prompt, tool_calls, steps, report", size=7.8)
    text(ax, 4.85, 2.6, "3 trials per task\nby default\n(--runs N)", size=7.5, color=INK2)

    # the other tasks
    rect(ax, 7.1, 1.95, 1.1, 3.4, fc=SURFACE, ec=ORANGE, lw=1.4, z=2)
    text(ax, 7.22, 5.1, "Task", size=10.5, weight="bold", color=INK2)
    text(ax, 7.22, 4.82, "age_x10", size=7.2, mono=True, color=INK2)
    for k in (2, 1, 0):
        rect(ax, 7.22 + 0.06 * k, 2.1 + 0.06 * k, 0.72, 0.8, fc=AQUA_TINT, ec=AQUA, lw=1.2, radius=0.05, z=3)
    rect(ax, 8.33, 1.95, 0.85, 3.4, fc=SURFACE, ec=ORANGE_LIGHT, lw=1.4, z=2)
    text(ax, 8.43, 5.1, "Task", size=10.5, weight="bold", color=MUTED)
    text(ax, 8.43, 4.82, f"+ {n_tasks - 2} more", size=7.2, mono=True, color=MUTED)

    # agent harness
    rect(ax, 9.75, 3.45, 1.85, 0.8, fc=GREY)
    text(ax, 10.675, 3.97, "Agent harness", size=9.5, weight="bold", ha="center")
    text(ax, 10.675, 3.7, "src/agent.py", size=7.5, mono=True, color=INK2, ha="center")
    arrow(ax, (9.32, 3.85), (9.73, 3.85), style="<|-|>")

    # outcome and grading
    arrow(ax, (4.75, 1.75), (4.75, 1.42))
    rect(ax, 3.3, 0.42, 2.9, 1.0, fc=VIOLET_TINT)
    text(ax, 3.45, 1.19, "Outcome", size=10.5, weight="bold")
    text(ax, 3.45, 0.78, "the final message: report + verdict\nblock (no environment state)", size=7.6, color=INK2)
    arrow(ax, (6.2, 0.92), (6.6, 0.92))
    rect(ax, 6.6, 0.42, 2.7, 1.0, fc=MAGENTA_TINT, ec=MAGENTA)
    text(ax, 6.75, 1.19, "Grader evaluate", size=10.5, weight="bold")
    text(ax, 6.75, 0.78, "transcript + final message →\n3 pass/fail checks + judge scores", size=7.6, color=INK2)

    # legend
    rect(ax, 9.6, 0.42, 2.05, 2.55, fc=SURFACE)
    entries = [
        (ORANGE, "Task = inputs + answer\nkey. The graders are the\nsame for every task."),
        (AQUA, "Trial = one run_agent()\ncall. Transcript = its\nrecord."),
        (MAGENTA, "Graders score aspects\nof one trial."),
    ]
    y = 2.66
    for color, label in entries:
        ax.add_patch(Circle((9.8, y), 0.065, fc=SURFACE, ec=color, lw=1.8, zorder=4))
        text(ax, 9.97, y + 0.08, label, size=7, color=INK2, va="top")
        y -= 0.8
    save(fig, "fig_ae2_components.png")


def fig3_grading_target() -> None:
    W, H = 11.0, 6.0
    fig, ax = canvas(W, H)
    title(ax, W, H, "Figure 3. What each grader reads from one trial")
    cols = ["tool_sequence", "numeric_faithfulness", "verdict", "judge"]
    col_x = [5.6, 7.35, 8.9, 10.0]
    for cx, name in zip(col_x, cols, strict=True):
        text(ax, cx, 5.2, name, size=8, mono=True, ha="center")
    G, E = "graded", "evidence"
    rows = [
        ("Trajectory", "prompts: system, user, tool descriptions", {1: E, 3: E}, None),
        ("", "tool calls: names and arguments", {0: G}, None),
        ("", "tool results", {0: G, 1: E, 3: E}, None),
        ("", "steps: tokens, latency, stop reason", {}, "tracked, not graded"),
        ("Final message", "report prose", {1: G, 2: G, 3: G}, None),
        ("", "JSON verdict block", {2: G}, None),
        ("Environment", "state after the run", {}, "nothing to read: the agent writes nothing"),
        ("Answer key", "expected.json", {0: E, 2: E}, None),
    ]
    y = 4.75
    for group, label, marks, note in rows:
        if group:
            ax.plot([0.2, 10.8], [y + 0.26, y + 0.26], color=GRID, lw=1, zorder=1)
            text(ax, 0.25, y, group, size=8.5, weight="bold")
        text(ax, 1.75, y, label, size=8.5, color=MUTED if note and not marks and group == "Environment" else INK)
        for k, kind in marks.items():
            if kind == G:
                ax.add_patch(Circle((col_x[k], y), 0.105, fc=BLUE, ec=SURFACE, lw=2, zorder=4))
            else:
                ax.add_patch(Circle((col_x[k], y), 0.095, fc=SURFACE, ec=BLUE, lw=1.7, zorder=4))
        if note:
            text(ax, (col_x[0] + col_x[-1]) / 2, y, note, size=8, color=MUTED, ha="center")
        y -= 0.5
    ax.plot([0.2, 10.8], [y + 0.26, y + 0.26], color=GRID, lw=1, zorder=1)
    ax.add_patch(Circle((1.85, 0.42), 0.105, fc=BLUE, ec=SURFACE, lw=2, zorder=4))
    text(ax, 2.05, 0.42, "graded: the check passes or fails on this", size=8, color=INK2)
    ax.add_patch(Circle((5.55, 0.42), 0.095, fc=SURFACE, ec=BLUE, lw=1.7, zorder=4))
    text(ax, 5.75, 0.42, "evidence: read to decide what counts as grounded or expected", size=8, color=INK2)
    save(fig, "fig_ae3_grading_target.png")


def fig4_grader_types(groups: dict[str, list[dict[str, Any]]]) -> None:
    stamp = most_trials(groups)
    runs = groups[stamp]
    n = runs[0]["n_scored"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.5, 1.2, 0.8], "wspace": 0.55})
    fig.subplots_adjust(left=0.2, right=0.99, top=0.8, bottom=0.25)
    fig.suptitle(
        f"Figure 4. The three grader types, as measured on {stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} ({n} trials per model)",
        x=0.01, y=0.97, ha="left", fontsize=11.5, weight="bold",
    )

    # deterministic checks
    ax = axes[0]
    checks = [
        ("tool_selection", "tool sequence"),
        ("runs_with_no_invented_number", "numeric faithfulness\n(no invented number)"),
        ("verdict", "verdict"),
    ]
    for i, (key, _) in enumerate(checks):
        for j, s in enumerate(runs):
            rate = 100 * s[key]["rate"]
            y = i + (j - 1) * 0.22
            ax.plot([rate], [y], "o", ms=9, color=MODEL_COLOR[s["model_requested"]], mec=SURFACE, mew=2, zorder=3, clip_on=False)
            if s[key]["passed"] < s[key]["n"] or (i == 0 and j == 0):
                ax.annotate(f"{s[key]['passed']}/{s[key]['n']}", (rate, y), xytext=(-9, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(checks)))
    ax.set_yticklabels([label for _, label in checks], fontsize=8.5)
    ax.set_ylim(len(checks) - 0.45, -0.55)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.5)
    ax.set_xlabel("share of trials that pass", fontsize=9)
    ax.grid(axis="x")
    ax.tick_params(length=0)
    ax.set_title("Deterministic checks", loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=MODEL_COLOR[s["model_requested"]], mec=SURFACE, mew=1.5,
                          label=short(s["model_requested"])) for s in runs]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.19, 0.0), ncol=3, fontsize=8.5, handletextpad=0.2)

    # model judge
    ax = axes[1]
    for j, s in enumerate(runs):
        mean = s["clarity"]["mean"]
        ax.plot([mean], [j], "o", ms=9, color=MODEL_COLOR[s["model_requested"]], mec=SURFACE, mew=2, zorder=3)
        ax.annotate(f"{mean:.2f}", (mean, j), xytext=(-9, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([short(s["model_requested"]) for s in runs], fontsize=8.5)
    ax.set_ylim(len(runs) - 0.4, -0.6)
    ax.set_xlim(1, 5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("mean clarity score, 1 to 5", fontsize=9)
    ax.grid(axis="x")
    ax.tick_params(length=0, labelsize=8.5)
    ax.set_title("Model judge", loc="left")
    fig.text(0.985, 0.02, "Judge: uncalibrated, and scored under two prompt\nvariants. Read the ranking, not the values.",
             ha="right", fontsize=8, color=INK2, linespacing=1.35)

    # human
    ax = axes[2]
    ax.axis("off")
    ax.set_title("Human review", loc="left")
    ax.text(0.0, 0.66, "0", fontsize=40, weight="bold", transform=ax.transAxes, va="center")
    ax.text(0.0, 0.3, "reports scored\nby a person", fontsize=9, color=INK2, transform=ax.transAxes, va="center")
    save(fig, "fig_ae4_grader_types.png")


def fig5_pass_at_k(groups: dict[str, list[dict[str, Any]]]) -> None:
    stamp = most_trials(groups)
    runs = groups[stamp]
    ks = list(range(1, 11))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"wspace": 0.22})
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.13)
    fig.suptitle("Figure 5. pass@k and pass^k: one success in k tries, or k successes in k tries",
                 x=0.01, y=0.97, ha="left", fontsize=11.5, weight="bold")

    # the article's worked example
    ax = axes[0]
    p = 0.75
    at_k = [100 * (1 - (1 - p) ** k) for k in ks]
    pow_k = [100 * p**k for k in ks]
    ax.axhline(100 * p, color=MUTED, lw=1)
    ax.plot(ks, at_k, "-o", color=AQUA, lw=2, ms=7, mec=SURFACE, mew=1.5, label="pass@k: at least one of k trials succeeds")
    ax.plot(ks, pow_k, "-o", color=ORANGE, lw=2, ms=7, mec=SURFACE, mew=1.5, label="pass^k: all k trials succeed")
    ax.annotate(f"{at_k[2]:.0f}% at k = 3", (3, at_k[2]), xytext=(6, -15), textcoords="offset points", fontsize=8.5)
    ax.annotate(f"{pow_k[2]:.0f}% at k = 3", (3, pow_k[2]), xytext=(-8, -12), textcoords="offset points", ha="right", fontsize=8.5)
    ax.text(10.2, 100 * p - 2.5, "per-trial success rate 75%", ha="right", va="top", fontsize=8, color=INK2)
    ax.set_title("The article's example: 75% per-trial success", loc="left")
    ax.legend(loc="center right", fontsize=8.5, bbox_to_anchor=(1.0, 0.5))

    # this project
    ax = axes[1]
    perfect = [s for s in runs if s["all_three_checks"]["passed"] == s["all_three_checks"]["n"]]
    others = [s for s in runs if s not in perfect]
    if perfect:
        n = perfect[0]["all_three_checks"]["n"]
        lb = lower_bound_all_pass(n)
        low = [100 * lb**k for k in ks]
        ax.fill_between(ks, low, [100] * len(ks), color=ORANGE, alpha=0.10, lw=0)
        ax.plot(ks, [100] * len(ks), "-o", color=ORANGE, lw=2, ms=7, mec=SURFACE, mew=1.5, clip_on=False, zorder=3)
        ax.plot(ks, low, color=ORANGE, lw=1.2, ls=(0, (4, 3)))
        names = ", ".join(short(s["model_requested"]) for s in perfect)
        ax.text(10.2, 94, f"{n}/{n} passed ({names})", ha="right", va="center", fontsize=8.5)
        ax.text(1.0, 30, f"dashed: 95% lower bound for {n}/{n}.\nA true rate of {100 * lb:.0f}% per trial is\nstill consistent with a perfect score.",
                ha="left", va="top", fontsize=8.5, color=INK2)
        ax.annotate(f"{low[2]:.0f}%", (3, low[2]), xytext=(-6, -10), textcoords="offset points", ha="right", fontsize=8.5)
        ax.annotate(f"{low[-1]:.0f}%", (10, low[-1]), xytext=(-2, -13), textcoords="offset points", ha="right", fontsize=8.5)
    for s in others:
        a = s["all_three_checks"]
        rate = a["passed"] / a["n"]
        vals = [100 * rate**k for k in ks]
        ax.plot(ks, vals, "-o", color=INK2, lw=2, ms=7, mec=SURFACE, mew=1.5, zorder=3)
        ax.text(10.2, vals[-1] - 7, f"{a['passed']}/{a['n']} passed ({short(s['model_requested'])}):\npoint estimate {vals[-1]:.0f}% at k = 10",
                ha="right", va="top", fontsize=8.5)
    ax.set_title("This project: pass^k, all three checks passing", loc="left")

    for ax in axes:
        ax.set_xlim(0.7, 10.3)
        ax.set_ylim(0, 104)
        ax.set_xticks(ks)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("number of trials (k)")
        ax.grid(axis="y")
        ax.tick_params(length=0, labelsize=8.5)
    save(fig, "fig_ae5_pass_at_k.png")


def _trial_panel(ax, x0: float, top: float, runs: list[dict[str, Any]], heading: str, sub: str) -> None:
    tasks = sorted({t for s in runs for t in s["per_case"]})
    text(ax, x0, top + 0.72, heading, size=9.5, weight="bold")
    text(ax, x0, top + 0.46, sub, size=8, color=INK2)
    col_w, side, gap = 1.25, 0.2, 0.06
    cols = [x0 + 2.05 + col_w * j for j in range(len(runs))]
    for cx, s in zip(cols, runs, strict=True):
        text(ax, cx + col_w / 2 - 0.1, top + 0.12, short(s["model_requested"]), size=7.5, mono=True, ha="center", color=INK2)
    passed_total = trials_total = 0
    for i, task in enumerate(tasks):
        y = top - 0.3 - 0.42 * i
        text(ax, x0, y, task, size=7.8, mono=True)
        for cx, s in zip(cols, runs, strict=True):
            pc = s["per_case"].get(task)
            if not pc:
                continue
            n, ok = int(pc["n"]), trials_passing_all(pc)
            passed_total += ok
            trials_total += n
            start = cx + col_w / 2 - 0.1 - (n * side + (n - 1) * gap) / 2
            for t in range(n):
                sx = start + t * (side + gap)
                if t < ok:
                    ax.add_patch(Rectangle((sx, y - side / 2), side, side, fc=BLUE, ec="none", zorder=3))
                else:
                    ax.add_patch(Rectangle((sx, y - side / 2), side, side, fc=SURFACE, ec=CRIT, lw=1.5, zorder=3))
                    text(ax, sx + side / 2, y - 0.005, "✗", size=8, ha="center", color=INK)
    y_end = top - 0.3 - 0.42 * len(tasks)
    text(ax, x0, y_end - 0.05, f"{passed_total} of {trials_total} trials pass every check", size=8.5, color=INK2)


def fig6_trial_grid(groups: dict[str, list[dict[str, Any]]]) -> None:
    multi, latest = most_trials(groups), max(groups)
    W, H = 12.0, 5.4
    fig, ax = canvas(W, H)
    title(ax, W, H, "Figure 6. Every trial that has been run on the full suite, one square per trial")

    def heading(stamp: str) -> tuple[str, str]:
        runs = groups[stamp]
        date = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
        k = runs[0]["runs_per_case"]
        return (f"{date}: {len(runs[0]['per_case'])} tasks × {k} trial{'s' if k > 1 else ''}",
                AGENT_VERSION.get(stamp, "agent version not recorded"))

    _trial_panel(ax, 0.3, 3.75, groups[multi], *heading(multi))
    _trial_panel(ax, 6.6, 3.75, groups[latest], *heading(latest))
    ax.add_patch(Rectangle((0.3, 0.22), 0.2, 0.2, fc=BLUE, ec="none"))
    text(ax, 0.6, 0.32, "trial passed all three checks", size=8, color=INK2)
    ax.add_patch(Rectangle((3.0, 0.22), 0.2, 0.2, fc=SURFACE, ec=CRIT, lw=1.5))
    text(ax, 3.1, 0.315, "✗", size=8, ha="center")
    text(ax, 3.3, 0.32, "trial failed a check", size=8, color=INK2)
    save(fig, "fig_ae6_trial_grid.png")


def fig7_headroom(groups: dict[str, list[dict[str, Any]]]) -> None:
    stamp = most_trials(groups)
    runs = groups[stamp]
    fig, ax = plt.subplots(figsize=(9.6, 2.9))
    ax.axvspan(0, 70, color=BLUE, alpha=0.08, lw=0)
    ax.axvspan(95, 100, color=INK2, alpha=0.10, lw=0)
    ax.text(35, 1.42, "a capability eval starts down here:\nlow pass rates leave a hill to climb", ha="center", va="center", fontsize=8.5, color=INK2)
    ax.text(94.3, 1.42, "a regression eval\nsits up here", ha="right", va="center", fontsize=8.5, color=INK2)
    for j, s in enumerate(runs):
        a = s["all_three_checks"]
        rate = 100 * a["passed"] / a["n"]
        y = 0.62 - 0.26 * j
        ax.plot([rate], [y], "o", ms=9, color=MODEL_COLOR[s["model_requested"]], mec=SURFACE, mew=2, zorder=3, clip_on=False)
        ax.annotate(f"{short(s['model_requested'])}: {a['passed']}/{a['n']}", (rate, y), xytext=(-10, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.1, 1.8)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.5)
    ax.set_xlabel(f"share of trials passing every check, live suite, {stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}")
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Figure 7. The live suite has no headroom: every model sits at the regression end",
                 loc="left", fontsize=11.5, pad=10)
    save(fig, "fig_ae7_headroom.png")


def fig8_roadmap() -> None:
    W, H = 12.0, 6.5
    fig, ax = canvas(W, H)
    title(ax, W, H, "Figure 8. The article's roadmap to good evals, with where this project stands on each step")
    columns = [
        ("Evaluation suite development", AQUA_TINT, [
            ("0.  Start now,\n     start early", "done"),
            ("1.  Start with\n     manual tests", "partial"),
            ("2.  Write\n     unambiguous tasks", "done"),
            ("3.  Cover positive\n     and negative cases", "done"),
        ]),
        ("Harness development", ORANGE_TINT, [
            ("4.  Build robust\n     eval harness", "done"),
            ("5.  Design graders\n     thoughtfully", "partial"),
        ]),
        ("Eval maintenance", VIOLET_TINT, [
            ("6.  Check the\n     trajectories", "partial"),
            ("7.  Monitor for\n     saturation", "missing"),
            ("8.  Maintain\n     long-term", "partial"),
        ]),
    ]
    box_w, box_h, top, step = 3.15, 0.8, 4.25, 1.12
    anchors: list[list[tuple[float, float]]] = []
    for c, (head, tint, steps) in enumerate(columns):
        px = 0.25 + 3.9 * c
        rect(ax, px, 0.25, 3.7, 5.6, fc=GREY, ec=GREY, radius=0.15, z=0)
        text(ax, px + 1.85, 5.38, head, size=10.5, weight="bold", color=INK2, ha="center", va="baseline")
        col_anchor = []
        for i, (label, status) in enumerate(steps):
            bx, by = px + 0.275, top - step * i
            rect(ax, bx, by, box_w, box_h, fc=tint)
            text(ax, bx + 0.15, by + box_h / 2, label, size=8.8)
            pill(ax, bx + box_w - 1.1, by + box_h / 2 - 0.15, status)
            if i:
                arrow(ax, (bx + box_w / 2, by + step), (bx + box_w / 2, by + box_h + 0.02))
            col_anchor.append((bx, by))
        anchors.append(col_anchor)
    # elbow connectors between columns, as in the original
    for c in (0, 1):
        sx, sy = anchors[c][-1]
        tx, ty = anchors[c + 1][0]
        mid = sx + box_w + (tx - sx - box_w) / 2
        y0, y1 = sy + box_h / 2, ty + box_h / 2
        ax.plot([sx + box_w, mid, mid], [y0, y0, y1], color=INK2, lw=1.3, zorder=4, solid_capstyle="round")
        arrow(ax, (mid, y1), (tx - 0.02, y1))
    save(fig, "fig_ae8_roadmap.png")


def fig9_layers() -> None:
    W, H = 12.0, 6.9
    fig, ax = canvas(W, H)
    title(ax, W, H, "Figure 9. Layered defences: which slices of the Swiss cheese this project has")
    side = 3.3
    slabs = [  # drawn back to front
        ((3.8, 0.25), SURFACE, MUTED, True, 1),
        ((2.5, 1.45), BLUE_TINT, INK2, False, 2),
        ((1.2, 2.65), BLUE_LIGHT, INK2, False, 3),
    ]
    for (x, y), fc, ec, dashed, z in slabs:
        rect(ax, x, y, side, side, fc=fc, ec=ec, dashed=dashed, radius=0.06, z=z)
    holes = [  # (x, y, radius, fill, z): a hole in the front slice shows what is behind it
        (2.45, 5.4, 0.2, SURFACE, 3.5),
        (3.8, 5.25, 0.32, SURFACE, 3.5),
        (1.9, 3.35, 0.42, SURFACE, 3.5),
        (3.8, 3.35, 0.32, BLUE_TINT, 3.5),
        (5.0, 4.05, 0.3, SURFACE, 2.5),
        (5.25, 1.95, 0.32, SURFACE, 2.5),
    ]
    for x, y, r, fc, z in holes:
        ax.add_patch(Circle((x, y), r, fc=fc, ec=INK2, lw=1.2, zorder=z))

    def caught(x: float, y: float) -> None:
        ax.add_patch(Circle((x, y), 0.2, fc=CRIT_TINT, ec=CRIT, lw=1.4, zorder=7))
        text(ax, x, y - 0.005, "✗", size=10, ha="center", z=8)

    # four failures travelling down and to the right
    arrow(ax, (0.35, 5.85), (1.72, 4.48), color=CRIT, z=6)   # stopped by the first slice
    caught(1.9, 4.3)
    arrow(ax, (0.35, 4.9), (2.92, 2.33), color=CRIT, z=6)    # through a hole, stopped by the second slice
    caught(3.1, 2.15)
    arrow(ax, (0.95, 6.2), (4.57, 2.58), color=CRIT, z=6)    # through a hole, stopped by the second slice
    caught(4.75, 2.4)
    arrow(ax, (2.75, 6.3), (7.55, 1.5), color=CRIT, z=6)     # through both; the third slice is not there
    text(ax, 7.62, 1.32, "reaches\na user", size=8, color=INK2, ha="center", va="top")

    notes = [  # (leader y, slice edge x, heading, status, body)
        (5.55, 4.5, "Automated evals", "done",
         "Unit tests on the tools, a replay regression\nsuite on every PR, live evals on main.\nCaught: a ratio the model computed itself,\na JS distance of 0.0 on a changed column."),
        (3.95, 5.8, "Manual transcript review", "partial",
         "Done twice, with no routine. Caught: a path\nthat leaked the answer, six checker false\npositives. Found only by this review: the\njudge punishing a grounded \"80/20\"."),
        (2.3, 7.1, "Production monitoring, A/B tests,\nuser feedback, human studies", "missing",
         "Not deployed, so no users and no traffic.\nRun logs record cost and latency per run,\nbut nothing watches them."),
    ]
    for y, edge_x, head, status, body in notes:
        ax.plot([edge_x, 8.15], [y, y], color=MUTED, lw=1, zorder=1)
        ax.add_patch(Circle((8.15, y), 0.05, fc=SURFACE, ec=INK2, lw=1.2, zorder=4))
        lines = head.count("\n") + 1
        text(ax, 8.35, y + (0.1 if lines > 1 else 0.0), head, size=9, weight="bold", va="top" if lines > 1 else "center")
        top = y - (0.18 if lines == 1 else 0.42)
        pill(ax, 8.35, top - 0.34, status)
        text(ax, 8.35, top - 0.46, body, size=7.6, color=INK2, va="top")
    arrow(ax, (0.35, 0.72), (0.95, 0.72), color=CRIT)
    text(ax, 1.1, 0.72, "a failure", size=8, color=INK2)
    caught(0.55, 0.3)
    text(ax, 1.1, 0.3, "caught by a layer", size=8, color=INK2)
    save(fig, "fig_ae9_layers.png")


def main() -> None:
    groups = load_groups()
    fig1_single_turn_vs_agent()
    fig2_components(groups)
    fig3_grading_target()
    fig4_grader_types(groups)
    fig5_pass_at_k(groups)
    fig6_trial_grid(groups)
    fig7_headroom(groups)
    fig8_roadmap()
    fig9_layers()


if __name__ == "__main__":
    main()
