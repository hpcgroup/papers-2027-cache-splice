"""Shared setup for the GT3D paper figures: the PSSG plotting template (pssg-plots), the series styles that every
figure uses, and small helpers for things the template's LinePlot / BarPlot do not cover (OOM markers, ideal lines,
power-of-two axes).

The template is imported from a checkout next to this repository (../pssg-plots); set PSSG_PLOTS to use another one.
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

PSSG = Path(os.environ.get("PSSG_PLOTS", Path(__file__).resolve().parents[2] / "pssg-plots"))
sys.path.insert(0, str(PSSG))
from pssgplot import PlotEnvironment, LinePlot, BarPlot, PALETTE, HATCHES, MARKERS, LINESTYLES  # noqa: E402,F401

FONT = PSSG / "fonts" / "gillsans.ttf"
FIGS = Path(__file__).resolve().parent

# one panel of a three-panel figure* (0.32 \textwidth) or a one-column figure (0.9 \columnwidth)
PANEL = (4.0, 3.0)
WIDE = (8.0, 3.0)

# the same series has the same color, dash, and marker in every figure
SERIES = {
    "GT3D": PALETTE[0],
    "2D x head": PALETTE[0],
    "2D": PALETTE[1],
    "GP-AG": PALETTE[2],
    "Ring (x)": PALETTE[3],
    "Ring (K|V)": PALETTE[4],
    "GP-A2A": PALETTE[6],
    "PyG": PALETTE[2],
    "cuSPARSE": PALETTE[1],
    "DF-GNN": PALETTE[5],
    "TorchGT": PALETTE[6],
    "TorchGT (random blocks)": PALETTE[6],
    "TorchGT (real topology)": PALETTE[3],
    "1D": PALETTE[2],
    "Head only": PALETTE[6],
    "Ring": PALETTE[3],
}

# fixed hatch per series, so that a series looks the same in every bar chart
HATCH = {"GT3D": "//", "2D x head": "//", "PyG": "xxx", "cuSPARSE": "|||", "DF-GNN": "OO", "2D": "++", "GP-AG": "OO",
         "TorchGT": "\\\\\\", "TorchGT (random blocks)": "\\\\\\", "TorchGT (real topology)": "**",
         "TorchGT (random)": "\\\\\\", "TorchGT (release)": "\\\\\\", "TorchGT (real)": "**"}   # short names of figs/samebatch


class env(PlotEnvironment):
    """PSSG plot environment with the bundled Gill Sans font; `scale` multiplies matplotlib's default font sizes.
    The bundled Gill Sans has an empty multiplication sign, so write it as $\\times$ (math text uses DejaVu Sans)."""

    def __init__(self, scale=1.0):
        super().__init__(font_path=FONT)
        self._scale = scale

    def __enter__(self):
        super().__enter__()
        s = self._scale
        plt.rcParams.update({"mathtext.fontset": "dejavusans", "font.size": 11 * s,
                             "axes.labelsize": 12 * s, "axes.titlesize": 12 * s, "xtick.labelsize": 11 * s,
                             "ytick.labelsize": 11 * s, "legend.fontsize": 10 * s, "pdf.fonttype": 42, "ps.fonttype": 42})
        return self


def palette(names):
    return {n: SERIES[n] for n in names}


def pow2_axis(ax, ticks, axis="x"):
    """Log2 axis with exactly the given ticks, labelled as plain integers."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log", base=2)
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)([f"{t:g}" for t in ticks])
    lim = (ticks[0] / 1.25, ticks[-1] * 1.25)
    (ax.set_xlim if axis == "x" else ax.set_ylim)(lim)


def log10_axis(ax, ticks, axis="y", fmt="{:g}"):
    """Log10 axis with exactly the given ticks."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log", base=10)
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    (ax.set_yticklabels if axis == "y" else ax.set_xticklabels)([fmt.format(t) for t in ticks])
    (ax.set_ylim if axis == "y" else ax.set_xlim)(ticks[0] / 1.3, ticks[-1] * 1.3)


def decade_axis(ax, lo, hi, axis="y"):
    """Log10 axis in the style of the Plexus strong-scaling figure: limits exactly (lo, hi), major ticks at the powers of
    ten labelled 10^k, unlabelled minor ticks at 2..9 x 10^k, and dotted grid lines at both."""
    import math
    from matplotlib.ticker import LogLocator, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log", base=10)
    ks = range(math.ceil(math.log10(lo) - 1e-9), math.floor(math.log10(hi) + 1e-9) + 1)
    a.set_major_locator(FixedLocator([10.0 ** k for k in ks]))
    (ax.set_yticklabels if axis == "y" else ax.set_xticklabels)([rf"$10^{{{k}}}$" for k in ks])
    a.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=100))
    a.set_minor_formatter(NullFormatter())
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)
    a.grid(True, which="both", linestyle="dotted", zorder=0)


def finish(chart, path, legend=None, **legend_kw):
    """Template look (dotted y grid, grey spines, ticks in), optional legend, tight layout, save as PDF."""
    ax = chart.ax
    ax.yaxis.grid(True, linestyle="dotted", zorder=0)
    ax.tick_params(axis="both", which="both", direction="in", color="#606060")
    if legend is False:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    elif legend is not None:
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(n) for n in legend if n in labels]
        ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, **legend_kw)
    chart.fig.tight_layout()
    chart.save(path, format="pdf", bbox_inches="tight")
    plt.close(chart.fig)
    print("wrote", path)


def hatch_bars(chart, hue_order, pal):
    """Template hatches for a hue-grouped bar plot, matched to the bars by their face color (robust to missing bars),
    and the same hatches on the legend patches."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle
    key = {to_rgba(pal[h])[:3]: HATCH.get(h, HATCHES[i % len(HATCHES)]) for i, h in enumerate(hue_order)}
    for p in chart.ax.patches:
        if isinstance(p, Rectangle):
            h = key.get(tuple(round(c, 6) for c in p.get_facecolor()[:3]))
            if h is None:
                h = next((v for k, v in key.items() if all(abs(a - b) < 1e-3 for a, b in zip(k, p.get_facecolor()[:3]))), None)
            if h is not None:
                p.set_hatch(h)
                p.set_edgecolor("black")
    leg = chart.ax.get_legend()
    if leg is not None:
        for p, h in zip(leg.get_patches(), hue_order):
            p.set_hatch(key[to_rgba(pal[h])[:3]])
            p.set_edgecolor("black")


def mark_missing(ax, x_order, hue_order, missing, text="OOM", y=None):
    """Write `text` (rotated) where a dodged bar is missing; `missing` is a set of (x, hue)."""
    n = len(hue_order)
    w = 0.8 / n
    y0 = y if y is not None else ax.get_ylim()[0]
    for (xv, hv) in missing:
        xi = x_order.index(xv) - 0.4 + w * (hue_order.index(hv) + 0.5)
        ax.text(xi, y0, text, rotation=90, ha="center", va="bottom", fontsize=plt.rcParams["font.size"] * 0.75,
                color="#606060")


class Canvas:
    """A bare figure with the template look, for chart types the template does not provide (e.g. stacked bars)."""

    def __init__(self, figsize=PANEL):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        for side in ("top", "right"):
            self.ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            self.ax.spines[side].set_color("#606060")

    def save(self, path, **kwargs):
        self.fig.savefig(path, **kwargs)
