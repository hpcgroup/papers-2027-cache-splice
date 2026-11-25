"""Kernel figure (fig:kernel, results section): kernel time of graph attention (one forward call on the projected Q, K, V) of the fused kernel against PyG gather/scatter, the cuSPARSE
SDDMM + SpMM pipeline and DF-GNN on the per-GPU blocks of our grids, for a plain GT (graph edges + self loops, dot scores,
exact softmax, the only normalisation DF-GNN has), one A100.  Data: make_data.py -> data_softmax.csv.
The clamp-exp version without DF-GNN (the figure until 2026-11-19) is plot_clampexp.py, the earlier three-panel Exphormer
version (forward, backward, global token) is plot_exphormer.py."""
from pathlib import Path
import sys

import re

import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, BarPlot, palette, log10_axis, finish, hatch_bars, mark_missing  # noqa: E402

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "data_softmax.csv", comment="#")
# The x axis groups the blocks by head configuration: all blocks with H = 2, d_h = 40 first, then H = 8, d_h = 32. A
# tick names only the block, and one line under each group carries the head configuration once.
CONFIGS = list(dict.fromkeys(df["config"]))                     # "2x40", "8x32" = heads x head dimension
BLOCKS = list(dict.fromkeys(df["block"]))
df["label"] = df["config"] + "|" + df["block"]
x_order = [f"{c}|{b}" for c in CONFIGS for b in BLOCKS]


def block_label(b):
    """'products 4x4' -> two lines with a proper multiplication sign (Gill Sans has none, see common.env). The arxiv
    block is the whole graph, labelled as the 1x1 grid so that every tick has the same two lines."""
    b = "arxiv 1x1" if b == "arxiv" else b
    return re.sub(r"(\d)x(\d)", r"\1$\\times$\2", b).replace(" ", "\n")


IMPLS = ["PyG", "cuSPARSE", "DF-GNN", "GT3D"]
YTICKS = [1, 10, 100, 1000]                                     # the time axis ends at the last tick
GROUP_Y = -0.27                                                # axes fraction of the head-configuration lines

with env():
    pal = palette(IMPLS)
    chart = BarPlot()
    chart.plot(data=df, x="label", y="fwd_ms", hue="impl", hue_order=IMPLS, order=x_order, palette=pal, hatch=False,
               saturation=1, linewidth=0.5, xlabel="", ylabel="Kernel time (ms)", figsize=(5.6, 2.8),
               ylim=(YTICKS[0], YTICKS[-1]))
    log10_axis(chart.ax, YTICKS)
    chart.ax.set_ylim(top=YTICKS[-1])
    missing = {(r.label, r.impl) for r in df.itertuples() if np.isnan(r.fwd_ms)}
    mark_missing(chart.ax, x_order, IMPLS, {m for m in missing if m[1] != "DF-GNN"}, y=YTICKS[0] * 1.1)
    mark_missing(chart.ax, x_order, IMPLS, {m for m in missing if m[1] == "DF-GNN"}, text="N/A", y=YTICKS[0] * 1.1)
    chart.ax.tick_params(axis="x", labelsize=9)
    chart.ax.set_xticks(range(len(x_order)))
    chart.ax.set_xticklabels([block_label(k.split("|")[1]) for k in x_order])
    trans = blended_transform_factory(chart.ax.transData, chart.ax.transAxes)
    for i, c in enumerate(CONFIGS):                                # one line and one label per head configuration
        lo, hi = i * len(BLOCKS) - 0.4, (i + 1) * len(BLOCKS) - 0.6
        h, dh = c.split("x")
        chart.ax.plot([lo, hi], [GROUP_Y, GROUP_Y], transform=trans, color="black", linewidth=0.8, clip_on=False)
        chart.ax.text((lo + hi) / 2, GROUP_Y - 0.03, rf"$H = {h},\ d_h = {dh}$", transform=trans, ha="center", va="top",
                      fontsize=10)
    chart.ax.legend(frameon=False, loc="upper left", ncol=len(IMPLS), columnspacing=1.0, handletextpad=0.4)
    hatch_bars(chart, IMPLS, pal)
    chart.ax.set_title("Graph attention kernel time (single GPU), Perlmutter")      # title added 2026-11-24 (user)
    finish(chart, HERE / "kernel_fwd.pdf")
