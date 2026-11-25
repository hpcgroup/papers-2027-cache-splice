"""Sec. 7.4: strong scaling of GT3D (the configuration the model picks) on ogbn-papers100M, ogbn-products and Reddit, in
the PSSG template's own style: pssgplot.LinePlot with hue = style = graph (PALETTE / MARKERS / LINESTYLES in GRAPHS
order, which is also the order of the curves from top to bottom), power-of-two GPU axis, and a log10 time axis in the
style of the Plexus strong-scaling figure (10 ms to 10 s, ticks labelled at the powers of ten, minor grid lines)."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, LinePlot, PALETTE, PANEL, pow2_axis, decade_axis, finish  # noqa: E402

HERE = Path(__file__).resolve().parent
GPUS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
FIGSIZE = (5.0, 3.0)                                             # 5:3, as figs/kernelmodel
GRAPHS = ["ogbn-papers100M", "ogbn-products", "Reddit"]          # PSSG style slots 0, 1, 2
df = pd.read_csv(HERE / "data.csv", comment="#")
df["ms"] = df["s"] * 1000

with env():
    chart = LinePlot()
    chart.plot(data=df, x="gpus", y="ms", hue="graph", style="graph", hue_order=GRAPHS, style_order=GRAPHS,
               palette=dict(zip(GRAPHS, PALETTE)), xlabel="Number of GPUs", ylabel="Time per epoch (ms)",
               title="Strong scaling of GT3D (Perlmutter)", figsize=FIGSIZE)
    pow2_axis(chart.ax, GPUS)
    decade_axis(chart.ax, 10, 10000)         # papers100M at width 128 starts at 8.4 s, above the earlier 5 s end
    chart.ax.tick_params(axis="x", labelsize=9)
    finish(chart, HERE / "scaling_best.pdf", legend=GRAPHS, loc="upper left", fontsize=9)
