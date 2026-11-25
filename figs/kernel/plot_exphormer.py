"""Fig. 4: the fused kernel against PyG gather/scatter and the cuSPARSE SDDMM + SpMM pipeline on the per-GPU blocks of
our grids (forward and backward time, one A100), and the effect of a global token."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, BarPlot, palette, log10_axis, finish, hatch_bars, mark_missing  # noqa: E402

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "data.csv", comment="#")
df["label"] = df["block"].str.replace(" ", "\n") + "\n" + df["config"].str.replace("x", r"$\times$")
x_order = list(dict.fromkeys(df["label"]))


def panel(value, impls, ylabel, yticks, out, figsize, legend):
    d = df[df["impl"].isin(impls)]
    pal = palette(impls)
    chart = BarPlot()
    chart.plot(data=d, x="label", y=value, hue="impl", hue_order=impls, order=x_order, palette=pal, hatch=False,
               saturation=1, linewidth=0.5, xlabel="", ylabel=ylabel, figsize=figsize, ylim=(yticks[0], yticks[-1]))
    log10_axis(chart.ax, yticks)
    missing = {(r.label, r.impl) for r in d.itertuples() if np.isnan(getattr(r, value))}
    if value == "bwd_ms":   # cuSPARSE has no sparse backward in PyTorch; PyG ran out of memory
        missing = {m for m in missing if m[1] == "PyG"}
    mark_missing(chart.ax, x_order, impls, missing, y=yticks[0] * 1.1)
    chart.ax.tick_params(axis="x", labelsize=9)
    chart.ax.legend(frameon=False, loc="upper left", ncol=len(impls))
    hatch_bars(chart, impls, pal)
    finish(chart, HERE / out)


hub = pd.read_csv(HERE / "hub.csv", comment="#")
hub["slowdown"] = hub["with_ms"] / hub["without_ms"]
hub["case"] = hub["impl"] + "\n" + hub["pass"].str.replace("forward", "fwd").str.replace("backward", "bwd")

with env():
    panel("fwd_ms", ["PyG", "cuSPARSE", "GT3D"], "Forward time (ms)", [1, 10, 100, 1000, 10000],
          "kernel_fwd.pdf", (6.0, 3.0), True)
    panel("bwd_ms", ["PyG", "GT3D"], "Backward time (ms)", [1, 10, 100, 1000, 10000],
          "kernel_bwd.pdf", (6.0, 3.0), False)

    cases = list(dict.fromkeys(hub["case"]))
    hub["config"] = hub["config"].str.replace("x", r"$\times$")
    cfgs = list(dict.fromkeys(hub["config"]))
    pal = {cfgs[0]: "#56B4E9", cfgs[1]: "#0072B2"}
    chart = BarPlot()
    chart.plot(data=hub, x="case", y="slowdown", hue="config", hue_order=cfgs, order=cases, palette=pal, hatch=False,
               saturation=1, linewidth=0.5, xlabel="", ylabel="Slowdown with a global token", figsize=(4.0, 3.0), ylim=(0, 11),
               labels="slowdown", label_fmt="{:.1f}", label_fontsize=8)
    chart.ax.axhline(1.0, color="#606060", linewidth=1, linestyle=(0, (4, 2)))
    chart.ax.set_yticks([0, 2, 4, 6, 8, 10])
    chart.ax.tick_params(axis="x", labelsize=9)
    chart.ax.legend(frameon=False, loc="upper right", title="Heads $\\times$ width")
    hatch_bars(chart, cfgs, pal)
    finish(chart, HERE / "kernel_hub.pdf")
