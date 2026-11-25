"""Sec. 7.3: time per epoch of GT3D (the model's pick) against GP-AG and GP-A2A, one line panel per graph, in the
PSSG template's own style: pssgplot.LinePlot with hue = style = series, so colors, markers and dashes come from PALETTE /
MARKERS / LINESTYLES in the order of SERIES_ORDER (the same order as figs/vs_torchgt).  The model has 8 heads since 2026-11-22
(data.csv; the 4-head data is data_heads4.csv), so GP-A2A, which needs P <= H = 8, has points at 4 and 8 GPUs.  The TorchGT rows of data.csv are not drawn since 2026-11-21: the
comparison with TorchGT is the same-batch figure (figs/samebatch)."""
from pathlib import Path
import sys

import pandas as pd
from matplotlib.colors import to_hex

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, LinePlot, PALETTE, PANEL, pow2_axis, log10_axis, finish  # noqa: E402

HERE = Path(__file__).resolve().parent
# PSSG style slots 0, 1, 2 in the order of the curves from top to bottom (time per epoch), as in the Plexus scaling figure
SERIES_ORDER = ["GP-AG", "GP-A2A", "GT3D"]
PAL = dict(zip(SERIES_ORDER, PALETTE))
df = pd.read_csv(HERE / "data.csv", comment="#")
df = df[df["series"].isin(SERIES_ORDER)]


# Proportions of the three panels of Figure 8 of the Plexus paper (papers-2025-plexus-sc/figs/scaling/plot_scaling.py): the
# default 6.4 x 4.8 inch canvas, a plot area of height : width = 3 : 5 (its set_aspect_ratio(3/5)), 20 pt tick labels, axis
# labels and title, and a 16 pt legend.
CANVAS, BOX_ASPECT, TEXT_PT, LEGEND_PT = (6.4, 4.8), 3 / 5, 20, 16


def panel(graph, gpus, yticks, out, legend, loc="upper right"):
    """The time axis runs exactly from the first to the last tick."""
    d = df[df["graph"] == graph]
    if d.empty:
        print("skip", graph, "(no data)")
        return
    chart = LinePlot()
    chart.plot(data=d, x="gpus", y="ms", hue="series", style="series", hue_order=SERIES_ORDER,
               style_order=SERIES_ORDER, palette=PAL, xlabel="Number of GPUs", ylabel="Time per epoch (ms)",
               title=f"Strong scaling ({graph}), Perlmutter", figsize=CANVAS)     # title style of 2026-11-24 (user)
    for line in chart.ax.lines:          # GP-A2A coincides with GT3D at 4 GPUs: draw it on top; markers at an axis end stay whole
        if len(line.get_xdata()):        # not seaborn's empty legend proxies: unclipped, they pull the tight bounding box to
            line.set_clip_on(False)      # the figure origin and leave blank margins on the left and at the bottom
        if to_hex(line.get_color()).lower() == PAL["GP-A2A"].lower():
            line.set_zorder(5)
    pow2_axis(chart.ax, gpus)
    log10_axis(chart.ax, yticks)
    chart.ax.set_ylim(yticks[0], yticks[-1])
    chart.ax.set_box_aspect(BOX_ASPECT)
    present = [s for s in SERIES_ORDER if s in set(d["series"])]
    finish(chart, HERE / out, legend=present if legend else False, loc=loc, fontsize=LEGEND_PT)


with env(scale=TEXT_PT / 11):
    panel("ogbn-arxiv", [4, 8, 16], [20, 30, 40], "epoch_time_arxiv.pdf", False)
    panel("Reddit", [4, 8, 16, 32, 64], [20, 50, 100, 200, 500], "epoch_time_reddit.pdf", False)
    panel("ogbn-products", [4, 8, 16, 32, 64, 128, 256], [50, 100, 200, 500, 1000], "epoch_time_products.pdf", True,
          loc="lower left")
