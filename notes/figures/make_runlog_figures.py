"""Figures for notes/run_logging.md, drawn from one logged run (runs/example by default).

    python notes/figures/make_runlog_figures.py [--run runs/example]

No API key needed. Colour roles follow make_figures.py: blue = deterministic code,
orange = the model / API, grey = furniture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, BLUE_LIGHT, BLUE_MID = "#2a78d6", "#cde2fb", "#86b6ef"
ORANGE, ORANGE_LIGHT = "#eb6834", "#fbd9cb"
AQUA, FLAG = "#1baf7a", "#d03b3b"

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


def box(ax, xy, w, h, text, fc, ec, *, fontsize=9, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec=ec, lw=1.4))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, color=color)


def arrow(ax, a, b, *, color=INK2, lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def load(run_dir: Path) -> tuple[dict, list[dict], dict]:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    trace = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))
    usage = json.loads((run_dir / "usage.json").read_text(encoding="utf-8"))
    return meta, trace, usage


def fig1_data_flow() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, (0.2, 2.6), 1.7, 1.3, "run_agent()\nloop", ORANGE_LIGHT, ORANGE, weight="semibold")
    box(ax, (2.5, 2.3), 2.3, 1.9, "AgentRun\nsteps[]  (per API call)\ntool_calls[]  (per tool)\nusage, stop_reason,\nmodel_served, error", BLUE_LIGHT, BLUE, fontsize=8.5)
    box(ax, (5.5, 3.5), 1.6, 0.7, "write_run()", BLUE_LIGHT, BLUE)
    box(ax, (5.5, 2.3), 1.6, 0.7, "run_to_dict()", BLUE_LIGHT, BLUE)
    files = [("meta.json", "who / when / what\ninputs + hashes"), ("trace.json", "steps + tool calls\nlatency, request id"),
             ("usage.json", "tokens, cost,\npricing date"), ("report.md", "the report")]
    for k, (name, desc) in enumerate(files):
        y = 4.3 - k * 0.95
        box(ax, (7.8, y), 2.0, 0.75, f"{name}\n{desc}", "#f1f0ec", MUTED, fontsize=7.5)
        arrow(ax, (7.1, 3.85), (7.8, y + 0.37), color=MUTED)
    box(ax, (7.8, 0.25), 2.0, 0.6, "evals/results/.../transcripts/*.json", "#f1f0ec", MUTED, fontsize=7.5)
    arrow(ax, (7.1, 2.65), (7.8, 0.55), color=MUTED)
    box(ax, (5.5, 0.4), 1.6, 0.7, "summarize.py", BLUE_LIGHT, BLUE)
    arrow(ax, (7.8, 1.0), (7.1, 0.75), color=MUTED)
    arrow(ax, (1.9, 3.25), (2.5, 3.25))
    arrow(ax, (4.8, 3.45), (5.5, 3.85))
    arrow(ax, (4.8, 3.05), (5.5, 2.65))
    ax.text(0.2, 1.75, "written in a finally block: a failed run is\nlogged with its error, then the exception\nis re-raised", fontsize=8.5, color=INK2, va="top")
    ax.text(4.95, 2.0, "same serialiser:\nan eval transcript and a run folder\ndescribe a run in the same words", fontsize=7.5, color=INK2, va="top")
    ax.text(3.3, 0.55, "one row per folder", fontsize=8, color=INK2, ha="center")
    arrow(ax, (5.5, 0.75), (1.6, 0.75), color=MUTED)
    ax.set_title("Figure 1. From the loop to the folder: the run record and the four files it becomes", loc="left")
    save(fig, "fig_rl1_data_flow.png")


def fig2_latency_waterfall(trace: list[dict], usage: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    t = 0.0
    rows: list[tuple[str, float, float, str]] = []
    for step in trace:
        rows.append((f"API call {step['turn']} ({step['stop_reason']})", t, step["latency_s"], ORANGE))
        t += step["latency_s"]
        for tc in step["tool_calls"]:
            rows.append((f"  tool {tc['name']}", t, tc["latency_s"], BLUE))
            t += tc["latency_s"]
    n = len(rows)
    for i, (_label, start, dur, color) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, max(dur, t * 0.004), left=start, height=0.6, color=color)
        ax.text(start + max(dur, t * 0.004) + 0.1, y, f"{dur:.3f} s" if dur < 1 else f"{dur:.1f} s", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5)
    ax.set_xlabel("seconds since the run started")
    ax.set_xlim(0, t * 1.15)
    ax.grid(axis="x")
    total, api, tools = usage["latency_s"], usage["api_latency_s"], usage["tool_latency_s"]
    ax.text(0.99, 0.97, f"wall {total:.1f} s = API {api:.1f} s + tools {tools:.2f} s ({100 * tools / total:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK2)
    ax.set_title("Figure 2. Where the time goes in one run: a waterfall from trace.json (tool bars widened to stay visible)", loc="left", fontsize=10.5)
    save(fig, "fig_rl2_latency_waterfall.png")


def fig3_tokens_cost(trace: list[dict], usage: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    turns = [s["turn"] for s in trace]
    x = range(len(turns))
    a.bar([i - 0.2 for i in x], [s["input_tokens"] for s in trace], width=0.4, color=BLUE_MID, label="input tokens")
    a.bar([i + 0.2 for i in x], [s["output_tokens"] for s in trace], width=0.4, color=ORANGE, label="output tokens")
    for i, s in enumerate(trace):
        a.text(i - 0.2, s["input_tokens"], str(s["input_tokens"]), ha="center", va="bottom", fontsize=8, color=INK2)
        a.text(i + 0.2, s["output_tokens"], str(s["output_tokens"]), ha="center", va="bottom", fontsize=8, color=INK2)
    a.set_xticks(list(x))
    a.set_xticklabels([f"API call {t}" for t in turns])
    a.set_ylabel("tokens")
    a.set_title("Input grows every turn: the whole conversation is resent")
    a.legend(loc="upper left")
    a.grid(axis="y")

    cost = usage["cost"]
    parts = [("input", cost["input_usd"], BLUE_MID), ("output", cost["output_usd"], ORANGE),
             ("cache read", cost["cache_read_usd"], BLUE_LIGHT), ("cache write", cost["cache_write_usd"], "#f1f0ec")]
    left = 0.0
    for name, v, c in parts:
        if v > 0:
            b.barh(0, v, left=left, color=c, height=0.5)
            b.text(left + v / 2, 0, f"{name}\n${v:.4f}", ha="center", va="center", fontsize=8.5)
            left += v
    b.set_yticks([])
    b.set_xlim(0, left * 1.05 if left else 1)
    b.set_xlabel("USD, list prices as of " + cost["pricing_date"])
    b.set_title(f"Estimated cost of the run: ${cost['total_usd']:.4f}")
    b.spines["left"].set_visible(False)
    fig.suptitle("Figure 3. Tokens per step and what they cost, from trace.json and usage.json", x=0.01, y=1.04, ha="left", fontsize=11, weight="semibold")
    save(fig, "fig_rl3_tokens_cost.png")


def fig4_pillars() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 4.2))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    cols = [
        ("PROVENANCE", "meta.json", "which inputs (sha256, rows, cols)\nwhich prompt, which system prompt\nwhich model asked for / served\nwhich git revision", BLUE_LIGHT, BLUE),
        ("TRACES", "trace.json", "one span per API call\nchild spans per tool call\nlatency, stop reason\nrequest id for support tickets", ORANGE_LIGHT, ORANGE),
        ("METRICS", "usage.json + summarize.py", "tokens in / out / cached\nestimated cost, dated\nAPI vs tool latency\naggregated across runs", BLUE_LIGHT, BLUE),
        ("LOGS", "report.md + tool results", "what the model said\nwhat the tools returned\nthe error, if any", "#f1f0ec", MUTED),
    ]
    for k, (title, file, body, fc, ec) in enumerate(cols):
        x = 0.3 + k * 2.8
        box(ax, (x, 0.5), 2.65, 3.6, "", fc, ec)
        ax.text(x + 1.325, 3.75, title, ha="center", fontsize=10.5, weight="semibold", color=ec if ec != MUTED else INK)
        ax.text(x + 1.325, 3.3, file, ha="center", fontsize=8, family="monospace", color=INK2)
        ax.text(x + 1.325, 1.9, body, ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(0.3, 0.1, "Answers, left to right: can I reproduce it?  where did the time go?  what does it cost and is it drifting?  what actually happened?", fontsize=8.5, color=INK2)
    ax.set_title("Figure 4. The four questions a run folder answers, and which file answers each", loc="left")
    save(fig, "fig_rl4_pillars.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "runs" / "example")
    args = ap.parse_args()
    meta, trace, usage = load(args.run)
    fig1_data_flow()
    fig2_latency_waterfall(trace, usage)
    fig3_tokens_cost(trace, usage)
    fig4_pillars()
    print("from run", meta["run_id"])


if __name__ == "__main__":
    main()
