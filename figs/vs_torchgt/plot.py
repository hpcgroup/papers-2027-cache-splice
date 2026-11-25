"""Strong scaling of GT3D against TorchGT on ogbn-products (comparison-only setting with 32 heads, so that TorchGT's
head-only parallelism can reach 32 GPUs): time per epoch, in the PSSG template's own style with the series slots of
figs/epoch_time (GT3D = slot 0, TorchGT = slot 3).  One TorchGT epoch covers only the subgraph induced by the training
nodes (random dense blocks in the released code), GT3D trains on the whole graph."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, LinePlot, PALETTE, PANEL, pow2_axis, log10_axis, finish  # noqa: E402

HERE = Path(__file__).resolve().parent
SERIES_ORDER = ["GT3D", "GP-AG", "GP-A2A", "TorchGT"]          # same slots as figs/epoch_time
df = pd.read_csv(HERE / "data.csv", comment="#")
df = df[df["graph"] == "ogbn-products"].rename(columns={"system": "series"})

with env():
    chart = LinePlot()
    chart.plot(data=df, x="gpus", y="ms", hue="series", style="series", hue_order=SERIES_ORDER,
               style_order=SERIES_ORDER, palette=dict(zip(SERIES_ORDER, PALETTE)), xlabel="Number of GPUs",
               ylabel="Time per epoch (ms)", title="ogbn-products, 32 heads", figsize=PANEL)
    pow2_axis(chart.ax, [4, 8, 16, 32])
    log10_axis(chart.ax, [50, 100, 200, 500, 1000])
    finish(chart, HERE / "vs_torchgt.pdf", legend=["GT3D", "TorchGT"], loc="upper right", fontsize=9)
