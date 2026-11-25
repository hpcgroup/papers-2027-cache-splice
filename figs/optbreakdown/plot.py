"""Breakdown of the epoch time with the communication optimizations of Sections 3.3 (sparsity-aware communication) and 3.4
(communication / computation overlap) switched on one after the other, in the PSSG template's style (stacked bars).

data.csv comes from gt3d/bench/parse_opt_breakdown.py (jobs of gt3d/scripts/sbatch_opt_breakdown{,_papers}.sh): one grid per
graph, the grid of the paper's other figures, width 128, 8 heads.  Every bar is the timer tree of one run (ms per epoch, mean
over ranks, self time of every region): attention kernels, the remaining computation, and the collectives by kind WITH
their waits, so an overlapped collective contributes only its exposed wait.  The bar's height is the timed step
(forward + backward + all-reduce + optimizer), which is within 1 % of the run's epoch time.

optbreakdown.pdf: one figure at \\columnwidth with two panels and a shared legend
  left   ogbn-products, 64 GPUs, (4,2,8): base and + overlap.  The plan's rule keeps this grid dense, so the paper's
         configuration ("full") is the overlap one; forcing the sparse collectives is 8 % slower than base (not drawn).
  right  ogbn-papers100M, 256 GPUs, 32x8 streamed: base, + sparse, + overlap, + both (the paper's configuration).
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, PALETTE, HATCHES  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGSIZE = (6.6, 3.5)                    # shown at \columnwidth, drawn with env(scale=1.3) like the half-width panels
# stack from the bottom: computation first, then the collectives.  Template colors and hatches; the black slot 3 is skipped,
# since a black bar hides its hatch.
PARTS = [("attn", "Attention kernel", PALETTE[0], HATCHES[0]), ("other", "Other computation", PALETTE[1], HATCHES[1]),
         ("ag", "All-gather", PALETTE[2], HATCHES[2]), ("rs", "Reduce-scatter", PALETTE[6], HATCHES[6]),
         ("a2a_ar", "All-to-all, all-reduce", PALETTE[4], HATCHES[4])]
PANELS = [  # graph, variants, tick labels, title, y max, y ticks
    ("ogbn-products", ["base", "full"], ["Base", "+Overlap"], "ogbn-products (64 GPUs)", 150, [0, 50, 100, 150]),
    ("ogbn-papers100M", ["base", "sparse", "overlap", "full"], ["Base", "+Sparse", "+Overlap", "+Both"],
     "ogbn-papers100M (256 GPUs)", 5000, [0, 1000, 2000, 3000, 4000, 5000]),
]

df = pd.read_csv(HERE / "data.csv", comment="#")
df["a2a_ar"] = df["a2a"] + df["ar"]

with env(scale=1.3):
    fs = plt.rcParams["font.size"]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [2.3, 4]})
    for ax, (graph, variants, labels, title, ymax, yticks) in zip(axes, PANELS):
        d = df[df["graph"] == graph].set_index("variant")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#606060")
        for i, v in enumerate(variants):
            bottom = 0.0
            for key, _, color, hatch in PARTS:
                val = float(d.loc[v, key])
                if val > 0:
                    ax.bar(i, val, bottom=bottom, width=0.62, color=color, hatch=hatch, edgecolor="black", linewidth=0.6, zorder=3)
                    bottom += val
            ax.text(i, bottom + ymax * 0.015, f"{bottom:,.0f}", ha="center", va="bottom", fontsize=fs * 0.8)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels, fontsize=fs * 0.9)
        ax.set_xlim(-0.6, len(variants) - 0.4)
        ax.set_ylim(0, ymax)
        ax.set_yticks(yticks)
        ax.set_title(title, fontsize=fs)
        ax.yaxis.grid(True, linestyle="dotted", zorder=0)
        ax.tick_params(axis="both", which="both", direction="in", color="#606060")
    axes[0].set_ylabel("Time per epoch (ms)")
    handles = [Patch(facecolor=c, hatch=h, edgecolor="black", linewidth=0.6, label=n) for _, n, c, h in PARTS]
    fig.legend(handles=handles, frameon=False, fontsize=fs * 0.78, loc="upper center", ncol=3, handlelength=1.4,
               columnspacing=1.0, labelspacing=0.3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = HERE / "optbreakdown.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", out)
