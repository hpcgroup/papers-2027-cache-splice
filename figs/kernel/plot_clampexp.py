"""Clamp-exp version of the kernel figure (the paper's figure until 2026-11-19, now plot.py with softmax and DF-GNN): forward time of the fused kernel against PyG gather/scatter and the cuSPARSE SDDMM + SpMM pipeline on the
per-GPU blocks of our grids, for a plain GT (graph edges + self loops, dot scores, clamped exponential), one A100.
The earlier three-panel Exphormer version (forward, backward, global token) is plot_exphormer.py."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, BarPlot, palette, log10_axis, finish, hatch_bars, mark_missing  # noqa: E402

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "data_plain.csv", comment="#")
df["label"] = df["block"].str.replace(" ", "\n") + "\n" + df["config"].str.replace("x", r"$\times$")
x_order = list(dict.fromkeys(df["label"]))
IMPLS = ["PyG", "cuSPARSE", "GT3D"]
YTICKS = [1, 10, 100, 1000, 10000]

with env():
    pal = palette(IMPLS)
    chart = BarPlot()
    chart.plot(data=df, x="label", y="fwd_ms", hue="impl", hue_order=IMPLS, order=x_order, palette=pal, hatch=False,
               saturation=1, linewidth=0.5, xlabel="", ylabel="Forward time (ms)", figsize=(5.6, 2.8),
               ylim=(YTICKS[0], YTICKS[-1]))
    log10_axis(chart.ax, YTICKS)
    missing = {(r.label, r.impl) for r in df.itertuples() if np.isnan(r.fwd_ms)}
    mark_missing(chart.ax, x_order, IMPLS, missing, y=YTICKS[0] * 1.1)
    chart.ax.tick_params(axis="x", labelsize=9)
    chart.ax.legend(frameon=False, loc="upper left", ncol=len(IMPLS))
    hatch_bars(chart, IMPLS, pal)
    finish(chart, HERE / "kernel_fwd_clampexp.pdf")
