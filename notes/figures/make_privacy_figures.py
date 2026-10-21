"""Figures for notes/privacy_tool.md, computed from the golden real data with the real tool.

    python notes/figures/make_privacy_figures.py      (no API key; about a minute)

Colour roles follow make_figures.py: blue = real data / deterministic code, orange =
synthetic data, grey = furniture, red only for a threshold line, always labelled.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from tools import _heom_encode, identifiability_heom  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, BLUE_LIGHT, BLUE_MID = "#2a78d6", "#cde2fb", "#86b6ef"
ORANGE, ORANGE_LIGHT = "#eb6834", "#fbd9cb"
AQUA, FLAG, GREY = "#1baf7a", "#d03b3b", "#f1f0ec"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})

REAL = ROOT / "evals/cases/identical/real.csv"
CLEAN = ROOT / "evals/cases/clean_synthetic/synth.csv"
NUMERIC = ["age", "income", "score"]


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


def measure(synth: pd.DataFrame, tmp: Path) -> dict:
    p = tmp / "synth.csv"
    synth.to_csv(p, index=False)
    return identifiability_heom(str(REAL), str(p))


# ----------------------------------------------------------------------------- figures
def fig1_heom() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    box(ax, (0.3, 2.6), 2.4, 1.3, "numeric column\nmin-max scaled on the\nreal data: 0 .. 1", BLUE_LIGHT, BLUE)
    box(ax, (0.3, 0.6), 2.4, 1.3, "categorical column\none-hot block, each\nentry scaled by 1/√2", BLUE_LIGHT, BLUE)
    box(ax, (3.6, 1.6), 2.3, 1.3, "one vector per row\n[ num | cat blocks ]\nEuclidean distance\n= HEOM distance", GREY, MUTED)
    arrow(ax, (2.7, 3.25), (3.6, 2.5))
    arrow(ax, (2.7, 1.25), (3.6, 2.0))
    rows = [
        ("same category", "0", AQUA),
        ("different category", "1   (√(½ + ½))", ORANGE),
        ("numeric apart by half the real range", "0.5", BLUE),
        ("unseen category", "\u2248 0.71 to every level", MUTED),
    ]
    ax.text(6.5, 3.95, "distance contributed by one column", fontsize=9, weight="semibold")
    for k, (what, d, c) in enumerate(rows):
        y = 3.45 - k * 0.6
        ax.text(6.5, y, what, fontsize=8.5, va="center")
        ax.text(10.4, y, d, fontsize=8.5, va="center", ha="right", color=c, weight="semibold")
    ax.text(6.5, 0.75, "So a categorical mismatch weighs the same as a numeric\ncolumn at opposite ends of its range: the HEOM\n(heterogeneous Euclidean-overlap) rule of Wilson & Martinez.", fontsize=8, color=INK2, va="top")
    ax.set_title("Figure 1. The HEOM encoding: mixed-type rows become vectors whose Euclidean distance is the HEOM distance", loc="left", fontsize=10.5)
    save(fig, "fig_pv1_heom.png")


def fig2_two_statistics() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    rng = np.random.default_rng(3)
    real = rng.uniform(0.5, 4.5, size=(14, 2))
    nn = NearestNeighbors(n_neighbors=2).fit(real)
    d, _ = nn.kneighbors(real)
    radii = d[:, 1]
    for ax, title in ((a, "identifiability: real side"), (b, "εANY: synthetic side")):
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title)
    # left: one real point, its nearest real neighbour, and the nearest synthetic point
    i = 5
    a.scatter(real[:, 0], real[:, 1], s=28, color=BLUE_MID, zorder=2)
    a.scatter(*real[i], s=70, color=BLUE, zorder=3)
    a.add_patch(Circle(real[i], radii[i], fill=False, ec=BLUE, lw=1.6, ls="--"))
    syn_in = real[i] + np.array([0.3, 0.1]) * radii[i]
    a.scatter(*syn_in, s=60, color=ORANGE, marker="s", zorder=4)
    a.text(real[i][0], real[i][1] + radii[i] + 0.15, "real record xᵢ and its radius rᵢ\n(distance to its nearest other real)", ha="center", fontsize=8, color=BLUE)
    a.text(syn_in[0] + 0.1, syn_in[1] - 0.25, "nearest synthetic\ncloser than rᵢ:\nxᵢ counts as identified", fontsize=8, color=ORANGE, va="top")
    a.set_xlabel("score = identified real records / all real records\nhonest sample of equal size: about 0.5 by symmetry", fontsize=8, color=INK2)
    # right: all real balls, synthetic points inside or outside
    b.scatter(real[:, 0], real[:, 1], s=28, color=BLUE_MID, zorder=2)
    for p, r in zip(real, radii, strict=True):
        b.add_patch(Circle(p, r, fill=True, fc=BLUE_LIGHT, ec=BLUE, lw=0.8, alpha=0.6))
    syn = rng.uniform(0.5, 4.5, size=(12, 2))
    dists, idx = NearestNeighbors(radius=float(radii.max())).fit(real).radius_neighbors(syn)
    unsafe = np.array([len(ii) > 0 and bool(np.any(dd < radii[ii])) for dd, ii in zip(dists, idx, strict=True)])
    b.scatter(syn[unsafe, 0], syn[unsafe, 1], s=60, color=ORANGE, marker="s", zorder=4, label="inside some ball: unsafe")
    b.scatter(syn[~unsafe, 0], syn[~unsafe, 1], s=60, facecolors=SURFACE, edgecolors=ORANGE, marker="s", zorder=4, label="outside all balls")
    b.legend(loc="lower left", fontsize=8)
    b.set_xlabel("score = unsafe synthetic rows / all synthetic rows\nhonest level depends on how the balls tile the space", fontsize=8, color=INK2)
    fig.suptitle("Figure 2. Two statistics from the same neighbourhoods: who is being measured decides what 'honest' looks like", x=0.01, y=1.02, ha="left", fontsize=11, weight="semibold")
    save(fig, "fig_pv2_two_statistics.png")


def fig3_calibration(tmp: Path) -> None:
    real = pd.read_csv(REAL)
    clean = pd.read_csv(CLEAN)
    rng = np.random.default_rng(0)
    sds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    ident, eps = [], []
    for sd in sds:
        j = real.copy()
        for col in NUMERIC:
            j[col] = j[col] + rng.normal(0, sd * j[col].std(), len(j))
        r = measure(j, tmp)
        ident.append(r["identifiability"])
        eps.append(r["epsilon_any"])
    honest = measure(clean, tmp)
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.plot(sds, ident, color=BLUE, lw=2, marker="o", ms=6, label="identifiability (thresholded at 0.7)")
    ax.plot(sds, eps, color=ORANGE, lw=2, marker="o", ms=6, label="εANY (reported, not thresholded)")
    ax.axhline(0.7, color=FLAG, lw=1.2, ls="--")
    ax.text(1.0, 0.715, "0.7 threshold", ha="right", fontsize=8, color=FLAG)
    ax.axhline(0.5, color=GRID, lw=1.2)
    ax.text(0.1, 0.515, "honest-sample reference 0.5", ha="center", fontsize=8, color=INK2)
    ax.axhline(honest["identifiability"], color=BLUE, lw=1, ls=":")
    ax.axhline(honest["epsilon_any"], color=ORANGE, lw=1, ls=":")
    ax.text(0.0105, honest["identifiability"] - 0.04, f"honest re-draw: identifiability {honest['identifiability']:.2f}", fontsize=8, color=BLUE)
    ax.text(0.0105, honest["epsilon_any"] - 0.045, f"honest re-draw: εANY {honest['epsilon_any']:.2f}", fontsize=8, color=ORANGE)
    ax.set_xscale("log")
    ax.set_xlabel("noise added to a copy of the real data (standard deviations of each numeric column)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y")
    ax.legend(loc="lower left")
    ax.set_title("Figure 3. Calibration on the golden data: a jittered copy becomes honest-looking around half a standard deviation", loc="left", fontsize=10.5)
    save(fig, "fig_pv3_calibration.png")


def fig4_partial_copy(tmp: Path) -> None:
    real = pd.read_csv(REAL)
    clean = pd.read_csv(CLEAN)
    fracs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    ident, eps, exact = [], [], []
    for f in fracs:
        n = int(round(2000 * f))
        mix = pd.concat([real.sample(n, random_state=1), clean.sample(2000 - n, random_state=1)]).reset_index(drop=True)
        r = measure(mix, tmp)
        ident.append(r["identifiability"])
        eps.append(r["epsilon_any"])
        exact.append(r["exact_match_fraction"])
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    ax.plot(fracs, ident, color=BLUE, lw=2, marker="o", ms=6, label="identifiability")
    ax.plot(fracs, eps, color=ORANGE, lw=2, marker="o", ms=6, label="εANY")
    ax.plot(fracs, exact, color=INK2, lw=2, marker="o", ms=6, label="exact-match fraction")
    ax.axhline(0.7, color=FLAG, lw=1.2, ls="--")
    ax.text(0.0, 0.715, "identifiability threshold 0.7", fontsize=8, color=FLAG)
    ax.axhline(0.05, color=FLAG, lw=1.2, ls=":")
    ax.text(0.0, 0.065, "exact-match threshold 0.05", fontsize=8, color=FLAG)
    ax.set_xlabel("fraction of synthetic rows that are verbatim real rows (the rest an honest re-draw)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y")
    ax.legend(loc="center right")
    ax.set_title("Figure 4. Partial copying: identifiability moves slowly, the exact-match rule catches it at once", loc="left")
    save(fig, "fig_pv4_partial_copy.png")


def fig5_size_ratio(tmp: Path) -> None:
    """Identifiability with and without the size-matched subsample, as an honest synthetic set grows.

    The larger sets are fresh draws from the known generative process (not resampled rows),
    so the only thing that changes is how many honest candidates each real record has nearby.
    """
    sys.path.insert(0, str(ROOT))
    from evals.generate_cases import draw_population

    real = pd.read_csv(REAL)
    ratios = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    matched, raw = [], []
    cat = list(real.select_dtypes(include=["object", "category", "string"]).columns)
    num = [c for c in real.columns if c not in cat]
    for q in ratios:
        synth = draw_population(np.random.default_rng(11), n=int(2000 * q))
        matched.append(measure(synth, tmp)["identifiability"])
        xr, xs = _heom_encode(real, synth, num, cat)
        rr, _ = NearestNeighbors(n_neighbors=2).fit(xr).kneighbors(xr)
        rs, _ = NearestNeighbors(n_neighbors=1).fit(xs).kneighbors(xr)
        raw.append(float(np.mean(rs[:, 0] < rr[:, 1])))
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    ax.plot(ratios, raw, color=ORANGE, lw=2, marker="o", ms=6, label="nearest synthetic among all synthetic rows")
    ax.plot(ratios, matched, color=BLUE, lw=2, marker="o", ms=6, label="size-matched subsample (what the tool reports)")
    for x, y in zip(ratios, raw, strict=True):
        ax.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=8, color=ORANGE)
    ax.axhline(0.5, color=GRID, lw=1.2)
    ax.text(4.0, 0.515, "honest reference 0.5", ha="right", fontsize=8, color=INK2)
    ax.axhline(0.7, color=FLAG, lw=1.2, ls="--")
    ax.text(4.0, 0.715, "threshold 0.7", ha="right", fontsize=8, color=FLAG)
    ax.set_xscale("log")
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{r:g}\u00d7" for r in ratios])
    ax.set_xlabel("honest synthetic rows as a multiple of real rows (fresh draws from the generative process)")
    ax.set_ylabel("identifiability")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    ax.set_title("Figure 5. Why identifiability is computed on a size-matched subsample: more honest rows alone move the raw score", loc="left", fontsize=10.5)
    save(fig, "fig_pv5_size_ratio.png")
    print("fig5 raw:", dict(zip(ratios, [round(v, 3) for v in raw], strict=True)), "matched:", dict(zip(ratios, [round(v, 3) for v in matched], strict=True)))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        fig1_heom()
        fig2_two_statistics()
        fig3_calibration(t)
        fig4_partial_copy(t)
        fig5_size_ratio(t)


if __name__ == "__main__":
    main()
