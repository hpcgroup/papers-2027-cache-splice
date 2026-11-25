"""Fig. 3: the parallel strategies on the attention operator, 4-64 GPUs (time for ogbn-products and Reddit at width 256,
memory for ogbn-products)."""
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, LinePlot, PANEL, palette, pow2_axis, log10_axis, finish  # noqa: E402

HERE = Path(__file__).resolve().parent
ORDER = ["Ring (x)", "Ring (K|V)", "GP-AG", "2D", "2D x head"]
LABEL = {"2D x head": "2D $\\times$ head"}

df = pd.read_csv(HERE / "data.csv", comment="#")
df["strategy"] = df["strategy"].map(lambda s: LABEL.get(s, s))
order = [LABEL.get(s, s) for s in ORDER]
pal = {LABEL.get(k, k): v for k, v in palette(ORDER).items()}


def panel(graph, value, ylabel, yticks, out, legend):
    d = df[(df["graph"] == graph) & df[value].notna()]
    chart = LinePlot()
    chart.plot(data=d, x="gpus", y=value, hue="strategy", style="strategy", hue_order=order, style_order=order,
               palette=pal, xlabel="Number of GPUs", ylabel=ylabel, title=graph.replace(" h", ", width "),
               figsize=PANEL, xlim=(3.2, 80), ylim=(yticks[0], yticks[-1]))
    pow2_axis(chart.ax, [4, 8, 16, 32, 64])
    log10_axis(chart.ax, yticks)
    finish(chart, HERE / out, legend=order if legend else False, loc="lower left", ncol=1)


with env():
    panel("products h256", "ms", "Time per layer (ms)", [50, 100, 200, 400, 800], "strategies_products.pdf", False)
    panel("Reddit h256", "ms", "Time per layer (ms)", [10, 20, 50, 100, 200], "strategies_reddit.pdf", False)
    panel("products h256", "gb", "Memory per GPU (GB)", [0.5, 1, 2, 5, 10, 20], "strategies_memory.pdf", True)
