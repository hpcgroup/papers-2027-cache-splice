"""Fig. 1: where TorchGT spends its time on real graphs -- (a) CPU-side block construction and serial evaluation,
(b) the backward pass of the global token, (c) strong scaling from 4 to 16 GPUs."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (env, BarPlot, LinePlot, Canvas, PANEL, PALETTE, palette, pow2_axis, log10_axis, finish,  # noqa: E402
                    hatch_bars)

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "data.csv", comment="#")

with env():
    # (a) stacked wall-clock breakdown per epoch
    d = df[df["kind"] == "breakdown"]
    parts = ["GPU training", "CPU block construction", "Serial evaluation", "Other"]
    colors = {"GPU training": PALETTE[1], "CPU block construction": PALETTE[0], "Serial evaluation": PALETTE[6],
              "Other": "#BBBBBB"}
    hatches = {"GPU training": "//", "CPU block construction": "xxx", "Serial evaluation": "OO", "Other": ""}
    batches = list(dict.fromkeys(d["batch"]))
    c = Canvas(PANEL)
    for i, b in enumerate(batches):
        left = 0.0
        for p in parts:
            v = d[(d["batch"] == b) & (d["series"] == p)]["seconds"]
            if len(v) == 0:
                continue
            v = float(v.iloc[0])
            c.ax.barh(i, v, left=left, color=colors[p], hatch=hatches[p], edgecolor="black", linewidth=0.5,
                      label=p if i == 0 or p == "Serial evaluation" else None, zorder=3)
            left += v
        c.ax.text(left + 0.05, i, f"{left:.1f} s", va="center", fontsize=10)
    c.ax.set_yticks(range(len(batches)))
    c.ax.set_yticklabels([b.replace(" (", "\n(") for b in batches])
    c.ax.invert_yaxis()
    c.ax.set_xlim(0, 5)
    c.ax.set_xlabel("Wall-clock time per epoch (s)")
    c.ax.xaxis.grid(True, linestyle="dotted", zorder=0)
    c.ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    c.ax.tick_params(axis="both", direction="in", color="#606060")
    c.fig.tight_layout()
    c.save(HERE / "torchgt_breakdown.pdf", format="pdf", bbox_inches="tight")
    print("wrote", HERE / "torchgt_breakdown.pdf")

    # (b) the global token in the backward pass
    d = df[df["kind"] == "globaltoken"].copy()
    d[["mode", "token"]] = d["series"].str.split(";", expand=True)
    tokens = ["with global token", "without"]
    pal = {"with global token": PALETTE[0], "without": PALETTE[1]}
    chart = BarPlot()
    chart.plot(data=d, x="mode", y="seconds", hue="token", hue_order=tokens, palette=pal, hatch=False, saturation=1,
               linewidth=0.5, xlabel="", ylabel="GPU time per epoch (s)", figsize=PANEL, ylim=(0, 0.7),
               labels="seconds", label_fmt="{:.2f}", label_fontsize=9)
    chart.ax.set_yticks([0, 0.2, 0.4, 0.6])
    chart.ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    hatch_bars(chart, tokens, pal)
    finish(chart, HERE / "torchgt_globaltoken.pdf")

    # (c) strong scaling, 4 -> 16 GPUs: speedup over the smallest GPU count that runs (dashed = ideal)
    d = df[df["kind"] == "scaling"].copy()
    d["line"] = d["batch"] + ", " + d["series"].str.replace("TorchGT (", "").str.replace(")", "")
    d["speedup"] = d.groupby("line")["seconds"].transform(lambda t: t.iloc[0] / t)
    order = list(dict.fromkeys(d["line"]))
    pal = {l: palette(["TorchGT (random blocks)", "TorchGT (real topology)"])[
        "TorchGT (random blocks)" if "random" in l else "TorchGT (real topology)"] for l in order}
    chart = LinePlot()
    chart.plot(data=d, x="gpus", y="speedup", hue="line", style="line", hue_order=order, style_order=order,
               palette=pal, xlabel="Number of GPUs", ylabel="Speedup", figsize=PANEL, xlim=(3.2, 20), ylim=(0, 4.4))
    chart.ax.plot([4, 16], [1, 4], color="#909090", linestyle=(0, (4, 2)), linewidth=1, zorder=1, label="Ideal")
    pow2_axis(chart.ax, [4, 8, 16])
    chart.ax.set_ylim(0, 4.4)
    chart.ax.set_yticks([0, 1, 2, 3, 4])
    finish(chart, HERE / "torchgt_scaling.pdf", legend=order + ["Ideal"], loc="upper left", fontsize=8.5)
