"""Strong scaling of the attention-only Exphormer (width 128) on ogbn-products and Reddit, 4-256 GPUs: GT3D (configuration
picked by the model) vs the best 2D grid, the 1D grid (GP-AG), head-only parallelism (GP-A2A) and the ring."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, LinePlot, PANEL, palette, pow2_axis, log10_axis, finish  # noqa: E402

HERE = Path(__file__).resolve().parent
ORDER = ["GT3D", "2D", "1D", "Head only", "Ring"]
df = pd.read_csv(HERE / "data.csv", comment="#")


def panel(graph, yticks, out, legend):
    d = df[(df["graph"] == graph) & df["ms"].notna()]
    order = [s for s in ORDER if s in set(d["series"])]
    chart = LinePlot()
    chart.plot(data=d, x="gpus", y="ms", hue="series", style="series", hue_order=order, style_order=order,
               palette=palette(order), xlabel="Number of GPUs", ylabel="Time per epoch (ms)", title=graph,
               figsize=PANEL, xlim=(3.2, 320), ylim=(yticks[0], yticks[-1]))
    g = d[d["series"] == "GT3D"].sort_values("gpus")
    for r in g.itertuples():   # the configuration the model picked
        label = r.config.replace(";", ",").replace("x", r"$\times$")
        chart.ax.annotate(label, (r.gpus, r.ms), textcoords="offset points", xytext=(4, 6), fontsize=8,
                          color="#404040")
    pow2_axis(chart.ax, [4, 16, 64, 256])
    log10_axis(chart.ax, yticks)
    finish(chart, HERE / out, legend=order if legend else False, loc="lower left", fontsize=9)


with env():
    panel("ogbn-products", [50, 100, 300, 1000], "scaling_products.pdf", True)
    panel("Reddit", [30, 100, 300, 1000], "scaling_reddit.pdf", True)
