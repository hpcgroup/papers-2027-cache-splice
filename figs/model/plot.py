"""Model validation: predicted vs. measured epoch time (model_time_<graph>_<width>.pdf) and peak memory
(model_memory_<graph>_<width>.pdf), one panel per (graph, width), for every measured configuration of the attention-only model, colored
by the total number of GPUs and shaped by the layout (ring, 1D, head, 1D + head, 2D, 2D + head); and the memory breakdown
(model_memory_breakdown.pdf)."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, Canvas, PANEL, PALETTE, log10_axis  # noqa: E402

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "data.csv", comment="#")

GPU_COLOR = {4: PALETTE[1], 16: PALETTE[2], 64: PALETTE[0]}
DIM_MARKER = {"Ring": "D", "1D": "^", "Head": "P", "1D + head": "v", "2D": "s", "2D + head": "o"}
GRAPH_NAME = {"products": "ogbn-products", "reddit": "Reddit"}
# axes of the panels that are in the paper (Figures 6 and 7, width 128); the other panels keep the automatic limits
TIME_AXIS = {("products", 128): (1000, 200), ("reddit", 128): (300, 100)}       # linear from 0: (end in ms, tick step)
MEM_TOP = {("products", 128): 40, ("reddit", 128): 5}                           # log axes: end in GB
# data.csv holds width 128 with 4 heads and, since 2026-11-22, with 8 heads (HEADS8_MODEL).  The paper's figures show the
# 8-head runs since 2026-11-25 (the paper's model): the per-graph kernel tables now hold the d_h = 16 configurations
# (gt3d/scripts/run_calib_d16.sh), and the model takes each run's --bwd_sub (make_data.py).  Set 128 to 4 for the 4-head runs.
HEADS_SHOWN = {128: 8, 256: 8}


def _ticks(lo, hi):
    """1-2-5 ticks inside (lo, hi), thinned to 1-3 or decades when there are too many."""
    for mant in ((1, 2, 5), (1, 3), (1,)):
        t = [m * 10.0 ** k for k in range(-2, 5) for m in mant if lo <= m * 10.0 ** k <= hi]
        if len(t) <= 5:
            return t
    return t


def scatter(p, meas, pred, label, out, title=None, gpu_legend=True, dim_legend=True, figsize=PANEL, top=None,
            linear_step=None):
    """top: end of both axes (default: 1.35 x the largest value).  linear_step: linear axes from 0 to `top` with a tick
    every `linear_step` instead of the log axes."""
    c = Canvas(figsize)
    ax = c.ax
    p = p[p[meas].notna()]
    vals = np.concatenate([p[meas].values, p[pred].values])
    lim = (0.0 if linear_step else vals.min() / 1.35, top if top is not None else vals.max() * 1.35)
    over = (p[meas] > lim[1]) | (p[pred] > lim[1])                # points beyond the end of an axis, see below
    xs = np.array(lim)
    ax.plot(xs, xs, color="#606060", linewidth=1, zorder=1)
    for gpus, color in GPU_COLOR.items():
        for dim, marker in DIM_MARKER.items():
            s = p[(p["gpus"] == gpus) & (p["dim"] == dim) & ~over]
            if len(s):
                ax.scatter(s[meas], s[pred], marker=marker, s=40, color=color, edgecolors="white", linewidths=0.5,
                           zorder=3, clip_on=False)
    # a point beyond the end of an axis is drawn on the border, with its true value written next to it: it is never dropped
    # or moved silently (Reddit, ring on 64 GPUs: 500 ms measured on an axis that ends at 300 ms)
    for r in p[over].itertuples():
        vm, vp = getattr(r, meas), getattr(r, pred)
        ax.scatter([min(vm, lim[1])], [min(vp, lim[1])], marker=DIM_MARKER[r.dim], s=40, color=GPU_COLOR[r.gpus],
                   edgecolors="white", linewidths=0.5, zorder=3, clip_on=False)
        note = f"{vm:.0f} $\\rightarrow$" if vm > lim[1] else f"{vp:.0f} $\\uparrow$"
        ax.annotate(note, (min(vm, lim[1]), min(vp, lim[1])), xytext=(-7, 6), textcoords="offset points", ha="right",
                    va="bottom", fontsize=8, color="#404040")
    if linear_step:
        ticks = list(np.arange(0, lim[1] + linear_step / 2, linear_step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    else:
        if top is None:
            ticks = _ticks(*lim)
        else:                                                  # 1-2-5 ticks up to the end of the axis, which gets a label
            ticks = [m * 10.0 ** k for k in range(-2, 5) for m in (1, 2, 5) if lim[0] <= m * 10.0 ** k <= top]
            if top not in ticks:
                ticks = [t for t in ticks if t < top / 1.6] + [top]
        log10_axis(ax, ticks, axis="x")
        log10_axis(ax, ticks, axis="y")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"Measured {label}")
    ax.set_ylabel(f"Predicted {label}")
    if title:
        ax.set_title(title)
    ax.yaxis.grid(True, linestyle="dotted", zorder=0)
    ax.tick_params(axis="both", which="both", direction="in", color="#606060")
    # legends: color = total GPUs, shape = layout
    if gpu_legend:
        h_gpu = [ax.scatter([], [], marker="o", s=40, color=c_) for c_ in GPU_COLOR.values()]
        l1 = ax.legend(h_gpu, [f"{g} GPUs" for g in GPU_COLOR], frameon=False, loc="upper left", fontsize=9)
        ax.add_artist(l1)
    if dim_legend:
        h_dim = [ax.scatter([], [], marker=m_, s=40, color="#606060") for m_ in DIM_MARKER.values()]
        ax.legend(h_dim, list(DIM_MARKER), frameon=False, loc="lower right", fontsize=9)
    c.fig.tight_layout()
    c.save(HERE / out, format="pdf", bbox_inches="tight")
    print("wrote", HERE / out)


with env():
    for i, (gname, hidden) in enumerate((("products", 128), ("products", 256), ("reddit", 128), ("reddit", 256))):
        sub = d[(d["graph"] == gname) & (d["hidden"] == hidden) & (d["heads"] == HEADS_SHOWN[hidden])]
        t_top, t_step = TIME_AXIS.get((gname, hidden), (None, None))
        # the paper shows only width 128 (Figures 6 and 7), so those titles name just the graph
        title = GRAPH_NAME[gname] if hidden == 128 else f"{GRAPH_NAME[gname]}, width {hidden}"
        scatter(sub, "measured_ms", "model_ms", "time (ms)", f"model_time_{gname}_{hidden}.pdf",
                title=title, gpu_legend=i == 0, dim_legend=i == 0, figsize=(3.6, 3.4),
                top=t_top, linear_step=t_step)
        scatter(sub, "measured_gb", "model_gb", "memory (GB)", f"model_memory_{gname}_{hidden}.pdf",
                title=title, gpu_legend=i == 0, dim_legend=i == 0, figsize=(3.6, 3.4),
                top=MEM_TOP.get((gname, hidden)))


def breakdown(out):
    """Stacked memory model terms of the four layouts (ogbn-products, width 128) at 4 and 16 GPUs; markers: measured peak."""
    b = pd.read_csv(HERE / "breakdown.csv", comment="#")
    parts = [("base", "Graph structures, features", "#BBBBBB", ""), ("owned", "Owned rows (all layers)", PALETTE[1], "//"),
             ("gathered", "Gathered blocks (all layers)", PALETTE[0], "xxx"), ("transient", "Backward transient", PALETTE[6], "OO")]
    c = Canvas((4.6, 3.0))
    ax = c.ax
    xs, labels = [], []
    for gi, gpus in enumerate(sorted(b["gpus"].unique())):
        sub = b[b["gpus"] == gpus]
        for li, r in enumerate(sub.itertuples()):
            x = gi * 5 + li
            xs.append(x); labels.append(r.layout)
            bottom = 0.0
            for key, name, color, hatch in parts:
                v = getattr(r, key)
                ax.bar(x, v, bottom=bottom, width=0.75, color=color, hatch=hatch, edgecolor="black", linewidth=0.5,
                       label=name if (gi == 0 and li == 0) else None, zorder=3)
                bottom += v
            ax.scatter([x], [r.measured_gb], marker="_", s=260, color="black", linewidths=2, zorder=4,
                       label="Measured peak" if (gi == 0 and li == 0) else None)
        ax.text(gi * 5 + 1.5, -6.5, f"{gpus} GPUs", ha="center", va="top", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Peak memory per GPU (GB)")
    ax.set_ylim(0, 40)
    ax.set_xlim(-0.7, 8.7)
    ax.yaxis.grid(True, linestyle="dotted", zorder=0)
    ax.tick_params(axis="both", direction="in", color="#606060")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    c.fig.tight_layout()
    c.save(HERE / out, format="pdf", bbox_inches="tight")
    print("wrote", HERE / out)


with env():
    breakdown("model_memory_breakdown.pdf")
