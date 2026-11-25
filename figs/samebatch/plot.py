"""Same model, same batches: GT3D against TorchGT (random blocks and real topology), in the PSSG template's own style.

The paper (Figure 10, 2026-11-13) shows ogbn-products only, as two panels side by side at 0.49 \\columnwidth, on 4 to 32
GPUs (no 1-GPU point, user 2026-11-13).  Both panels use template slots 0, 1, 2 (GT3D, TorchGT random, TorchGT real), so
that the template's own bars (user: "use the PSSG style") and the lines share colors.  The arxiv and one-panel plots keep the
slots of figs/epoch_time (GT3D = 0, TorchGT = 3, real topology = 4).
  samebatch_products_time.pdf  time per epoch, A100-40GB nodes except 4 GPUs, where both systems come from the same A100-80GB
                               job (TorchGT runs out of memory on four 40 GB GPUs)
  samebatch_products_mem.pdf   peak memory per GPU (max over ranks), hatched bars, dashed line = A100-40GB capacity.
                               Allocated memory does not depend on the GPU, so a TorchGT run on A100-80GB (hw = 80GB) shows
                               how far it exceeds 40 GB where the 40GB nodes ran out of memory.
samebatch_arxiv.pdf / samebatch_products.pdf are the earlier one-panel time plots (not in the paper)."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, LinePlot, BarPlot, PALETTE, PANEL, pow2_axis, log10_axis, decade_axis, finish  # noqa: E402

HERE = Path(__file__).resolve().parent
GPUS = [1, 4, 8, 16, 32]
PAPER_GPUS = [4, 8, 16, 32]
PAPER_HW = {4: "80GB"}                  # GPU count -> node type taken for that point (default 40GB)
SERIES_ORDER = ["GT3D", "GP-AG", "GP-A2A", "TorchGT (release)", "TorchGT (real)"]
PAL = dict(zip(SERIES_ORDER, PALETTE))
# template style slots 0, 1, 2 in the order of the curves from top to bottom (user, 2026-11-22); the bars follow the same order
SHOWN = ["TorchGT (real)", "TorchGT (release)", "GT3D"]
# legend names; "release" = the default mode of the released code (random blocks), renamed from "random" on 2026-11-21
SHORT = {"TorchGT (random blocks)": "TorchGT (release)", "TorchGT (real topology)": "TorchGT (real)"}
PAPER_PAL = dict(zip(SHOWN, PALETTE))     # Figure 10: template slots 0, 1, 2 in both panels, as the template's bars use them
HALF = (3.3, 3.1)                       # one of two panels side by side at 0.49 \columnwidth, drawn with env(scale=1.3)
CAPACITY_GB = 40
df = pd.read_csv(HERE / "data.csv", comment="#")
df["series"] = df["series"].replace(SHORT)


def time_panel(graph, yticks, out, legend, figsize=PANEL, fs=9, gpus=GPUS, hw_at=None, order=SERIES_ORDER, pal=PAL,
               decades=None, **legend_kw):
    """decades = (lo, hi): time axis in the Plexus strong-scaling style (common.decade_axis) instead of the 1-2-5 ticks."""
    hw_at = hw_at or {}
    d = df[(df["graph"] == graph) & df["gpus"].isin(gpus)]
    d = d[d["hw"] == d["gpus"].map(lambda g: hw_at.get(g, "40GB"))]
    ok = d[d["status"] == "ok"]
    chart = LinePlot()
    chart.plot(data=ok, x="gpus", y="ms", hue="series", style="series", hue_order=order,
               style_order=order, palette=pal, xlabel="Number of GPUs", ylabel="Time per epoch (ms)",
               title=graph, figsize=figsize)
    pow2_axis(chart.ax, gpus)
    if decades:
        decade_axis(chart.ax, *decades)
    else:
        log10_axis(chart.ax, yticks)
    for _, r in d[d["status"] == "OOM"].iterrows():
        nxt = ok[(ok["series"] == r["series"])].sort_values("gpus")
        y = nxt["ms"].iloc[0] if len(nxt) else yticks[0]           # at the level of the series' first measured point
        chart.ax.text(r["gpus"], y, "OOM", ha="center", va="center", fontsize=fs, color=pal[r["series"]])
    finish(chart, HERE / out, legend=SHOWN if legend else False, fontsize=fs, **legend_kw)


def mem_rows(graph, gpus=GPUS):
    """One memory value per (series, gpus): max over ranks, from the 40GB nodes if there, else from the 80GB nodes."""
    d = df[(df["graph"] == graph) & df["mem_gb"].notna() & df["gpus"].isin(gpus)].copy()
    d["pref"] = (d["mem_kind"] != "max-rank") * 2 + (d["hw"] != "40GB")
    d = d.sort_values("pref").drop_duplicates(["series", "gpus"])
    # every (series, gpus) pair present, NaN where unmeasured, so that the three series keep fixed slots in each group
    full = pd.MultiIndex.from_product([SHOWN, gpus], names=["series", "gpus"]).to_frame(index=False)
    return full.merge(d, on=["series", "gpus"], how="left")


def mem_panel(graph, out, figsize=HALF, fs=8, gpus=GPUS):
    """The PSSG template's own grouped bars (user, 2026-11-13): BarPlot defaults, i.e. PALETTE in hue order, HATCHES in hue
    order with black hatch lines, seaborn's default saturation and gap.  The colors match the time panel (PAPER_PAL)."""
    d = mem_rows(graph, gpus)
    chart = BarPlot()
    top = max(CAPACITY_GB * 1.25, d["mem_gb"].max() * 1.15)
    chart.plot(data=d, x="gpus", y="mem_gb", hue="series", hue_order=SHOWN, order=gpus, palette=PAPER_PAL,
               xlabel="Number of GPUs", ylabel="Peak memory per GPU (GB)", title=graph, figsize=figsize, ylim=(0, top))
    chart.ax.axhline(CAPACITY_GB, color="#606060", linestyle="--", linewidth=1, zorder=2)
    chart.ax.text(len(gpus) - 0.5, CAPACITY_GB, "40 GB", ha="right", va="bottom", fontsize=fs, color="#606060")
    finish(chart, HERE / out, legend=False)          # the legend is in the time panel


with env():
    time_panel("ogbn-arxiv", [50, 100, 200, 500, 1000], "samebatch_arxiv.pdf", True, loc="center right")
    time_panel("ogbn-products", [100, 200, 500, 1000, 2000, 5000, 10000], "samebatch_products.pdf", True,
               loc="upper right", bbox_to_anchor=(1.0, 0.8))
with env(scale=1.3):        # Figure 10: two panels at 0.49 \columnwidth need larger type than one panel at 0.9 \columnwidth
    # the legend sits in the time panel, in the empty band between 1 and 4 s
    time_panel("ogbn-products", [100, 200, 500, 1000, 2000, 5000, 10000], "samebatch_products_time.pdf", True,
               figsize=HALF, fs=11, gpus=PAPER_GPUS, hw_at=PAPER_HW, order=SHOWN, pal=PAPER_PAL, decades=(100, 10000),
               loc="upper right",
               bbox_to_anchor=(1.04, 0.84),
               handlelength=1.6, borderpad=0.2)
    mem_panel("ogbn-products", "samebatch_products_mem.pdf", fs=11, gpus=PAPER_GPUS)
