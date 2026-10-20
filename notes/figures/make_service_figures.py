"""Figures for notes/service_and_deployment.md. Pure diagrams, no data, no API key.

    python notes/figures/make_service_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, BLUE_LIGHT = "#2a78d6", "#cde2fb"
ORANGE, ORANGE_LIGHT = "#eb6834", "#fbd9cb"
AQUA, FLAG = "#1baf7a", "#d03b3b"
GREY = "#f1f0ec"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlesize": 11,
})


def save(fig: plt.Figure, name: str) -> None:
    out = HERE / name
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def box(ax, xy, w, h, text, fc, ec, *, fontsize=8.5, weight="normal", color=INK, lw=1.4):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec=ec, lw=lw))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, color=color)


def arrow(ax, a, b, *, color=INK2, lw=1.4, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def canvas(w: float, h: float, xmax: float, ymax: float):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def fig1_request_flow() -> None:
    fig, ax = canvas(11, 4.8, 11, 5)
    y = 3.1
    steps = [
        ("client\nPOST /evaluate\n2 CSVs + target", GREY, MUTED),
        ("read with limit\n413 if too big", BLUE_LIGHT, BLUE),
        ("parse + validate\n400: not CSV, rows,\ncolumns, target", BLUE_LIGHT, BLUE),
        ("temp dir\nreal.csv / synth.csv\n(neutral names)", BLUE_LIGHT, BLUE),
        ("run_agent()\nlog_dir=RUNS_DIR", ORANGE_LIGHT, ORANGE),
        ("response\nrun id, verdict,\nusage, cost, report", GREY, MUTED),
    ]
    w, gap = 1.55, 0.3
    for k, (t, fc, ec) in enumerate(steps):
        x = 0.2 + k * (w + gap)
        box(ax, (x, y), w, 1.3, t, fc, ec, fontsize=8)
        if k:
            arrow(ax, (x - gap, y + 0.65), (x, y + 0.65))
    # gates before the work
    box(ax, (0.2, 1.3), 3.4, 0.8, "gates checked first: model in allowlist (400),\nfree slot in the semaphore (503 if busy)", GREY, MUTED, fontsize=8)
    arrow(ax, (1.9, 2.1), (1.9, 3.1), color=MUTED)
    # side effects
    box(ax, (7.6, 1.2), 1.55, 0.9, "runs/<id>/\nmeta, trace,\nusage, report", BLUE_LIGHT, BLUE, fontsize=7.5)
    arrow(ax, (8.375, 3.1), (8.375, 2.1), color=BLUE)
    box(ax, (5.75, 1.3), 1.55, 0.8, "finally: temp dir\nremoved, slot freed", GREY, MUTED, fontsize=7.5)
    arrow(ax, (6.525, 3.1), (6.525, 2.1), color=MUTED)
    box(ax, (9.3, 1.3), 1.55, 0.8, "upstream errors\n502 / 503", ORANGE_LIGHT, ORANGE, fontsize=7.5)
    arrow(ax, (9.3, 3.4), (9.55, 2.1), color=ORANGE)
    ax.text(0.2, 0.55, "Everything left of the orange box is deterministic and runs before any money is spent. "
            "The model is called once per request, synchronously; the HTTP response waits for the report.", fontsize=8.5, color=INK2)
    ax.set_title("Figure 1. One request through POST /evaluate: validate, isolate, run, record, respond", loc="left")
    save(fig, "fig_sv1_request_flow.png")


def fig2_pipeline() -> None:
    fig, ax = canvas(11, 4.4, 11, 4.6)
    box(ax, (0.3, 1.6), 2.3, 1.5, "test\nruff, mypy, unit,\nregression, API tests\n(every PR and push)", BLUE_LIGHT, BLUE, weight="normal")
    box(ax, (3.6, 2.7), 2.4, 1.3, "evals\nHaiku x1 on main,\nany model by hand\n(needs API key secret)", ORANGE_LIGHT, ORANGE)
    box(ax, (3.6, 0.6), 2.4, 1.3, "docker\nbuild, push to GHCR\n:sha and :latest,\nsmoke /health", BLUE_LIGHT, BLUE)
    box(ax, (7.2, 0.6), 2.4, 1.3, "deploy\nOIDC login, set secret,\nupdate image, wait\nfor /health", BLUE_LIGHT, BLUE)
    arrow(ax, (2.6, 2.6), (3.6, 3.3))
    arrow(ax, (2.6, 2.1), (3.6, 1.3))
    arrow(ax, (6.0, 1.25), (7.2, 1.25))
    ax.text(6.6, 1.45, "main only", fontsize=7, color=INK2, ha="center")
    ax.text(3.0, 3.25, "main or manual", fontsize=7.5, color=INK2, ha="center", rotation=22)
    ax.text(3.0, 1.45, "main only", fontsize=7.5, color=INK2, ha="center", rotation=-22)
    box(ax, (7.2, 2.7), 3.5, 1.3, "gate: vars.AZURE_DEPLOY == 'true'\nand the three OIDC secrets;\nabsent, the job is skipped and the\nimage is still published", GREY, MUTED, fontsize=7.5)
    arrow(ax, (8.4, 2.7), (8.4, 1.9), color=MUTED)
    ax.text(0.3, 0.25, "Cost guard: the regression suite never calls the model; the automatic eval uses the cheapest model once per case; "
            "Sonnet-scale evals are a manual button.", fontsize=8.5, color=INK2)
    ax.set_title("Figure 2. The CI pipeline: tests gate everything, money and deployment are opt-in", loc="left")
    save(fig, "fig_sv2_pipeline.png")


def fig3_image_layers() -> None:
    fig, ax = canvas(10, 4.4, 10, 4.6)
    # builder
    box(ax, (0.3, 0.5), 4.0, 3.6, "", ORANGE_LIGHT, ORANGE)
    ax.text(2.3, 3.75, "builder stage", ha="center", fontsize=10.5, weight="semibold", color=ORANGE)
    for k, t in enumerate(["python:3.12-slim", "python -m venv /opt/venv", "COPY requirements.lock", "pip install -r requirements.lock\n(compilers, caches, wheels: all stay here)"]):
        box(ax, (0.6, 2.75 - k * 0.72), 3.4, 0.55, t, SURFACE, ORANGE, fontsize=8)
    # runtime
    box(ax, (5.7, 0.5), 4.0, 3.6, "", BLUE_LIGHT, BLUE)
    ax.text(7.7, 3.75, "runtime stage (what ships)", ha="center", fontsize=10.5, weight="semibold", color=BLUE)
    for k, t in enumerate(["python:3.12-slim + libgomp1", "useradd app  ->  USER app", "COPY --from=builder /opt/venv", "COPY src/   (nothing else)", "HEALTHCHECK /health   CMD uvicorn"]):
        box(ax, (6.0, 2.85 - k * 0.56), 3.4, 0.44, t, SURFACE, BLUE, fontsize=8)
    arrow(ax, (4.3, 1.6), (5.7, 1.55), color=INK2, lw=1.8)
    ax.text(5.0, 1.75, "venv only", ha="center", fontsize=8, color=INK2)
    ax.text(0.3, 0.15, "The key is never in the image: ANTHROPIC_API_KEY arrives from the environment at run time (a Container Apps secret in Azure, .env locally).", fontsize=8.5, color=INK2)
    ax.set_title("Figure 3. The multi-stage image: build tools stay in the builder, the runtime is slim and non-root", loc="left")
    save(fig, "fig_sv3_image_layers.png")


def fig4_defence_layers() -> None:
    fig, ax = canvas(10.5, 4.6, 10.5, 4.8)
    rings = [
        ("size limit (413)", "bytes read with a cap, before parsing"),
        ("format + shape (400)", "CSV only, 10..200k rows, same columns, target with 2+ classes"),
        ("model allowlist (400)", "only models the operator priced and approved"),
        ("concurrency cap (503)", "a bounded semaphore; a flood waits or is told to retry"),
        ("neutral paths", "uploads renamed, so a file name cannot hint the model"),
        ("run id pattern (404)", "strict regex before any path is built: no traversal"),
        ("secret by reference", "key from the environment / secretref, never in image or logs"),
        ("least privilege", "non-root user, slim image, only src/ copied"),
    ]
    for k, (title, body) in enumerate(rings):
        y = 4.2 - k * 0.5
        box(ax, (0.3, y - 0.2), 2.6, 0.42, title, BLUE_LIGHT if k < 6 else GREY, BLUE if k < 6 else MUTED, fontsize=8.5)
        ax.text(3.1, y + 0.01, body, va="center", fontsize=8.5, color=INK)
    ax.text(0.3, 0.1, "Order matters: the cheap checks run first, and nothing reaches the model until every one has passed.", fontsize=8.5, color=INK2)
    ax.set_title("Figure 4. Defence in depth for a public endpoint that spends money on every request", loc="left")
    save(fig, "fig_sv4_defence_layers.png")


def main() -> None:
    fig1_request_flow()
    fig2_pipeline()
    fig3_image_layers()
    fig4_defence_layers()


if __name__ == "__main__":
    main()
