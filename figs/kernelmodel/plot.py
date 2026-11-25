"""Time per row vs. non-zeros per row on regular blocks, measured (hollow markers) and the analytic model (lines),
for the three passes of the attention kernel with 8 heads of width 16 (the paper's model; 4 heads of width 32 until
2026-11-25): kernel_model.pdf.
The legend lists the markers (measured) and the lines (model) of the three passes (user, 2026-11-24)."""
from pathlib import Path
import sys

from matplotlib.lines import Line2D
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, Canvas, PANEL, PALETTE, pow2_axis, log10_axis  # noqa: E402

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "data.csv", comment="#")
# template style slots 0, 1, 2 in the order of the curves from top to bottom
PASSES = [("row", "Row pass", PALETTE[0], "o"), ("col", "Column pass", PALETTE[1], "^"),
          ("fwd", "Forward", PALETTE[2], "s")]
TITLE = "Kernel model (synthetic blocks), Perlmutter"
FIGSIZE = (5.0, 3.0)                                           # 5:3, shown at \columnwidth
Y_TOP = 150                                                    # ns, end of the time axis

with env(scale=1.0):
    c = Canvas(FIGSIZE)
    ax = c.ax
    s = d[(d["heads"] == 8) & (d["d_head"] == 16)]             # the head configuration of the paper's model
    for p, label, color, marker in PASSES:
        q = s[s["pass"] == p].sort_values("degree")
        ax.plot(q["degree"], q["model_ns"], color=color, linewidth=1.5, zorder=2)
        ax.scatter(q["degree"], q["measured_ns"], marker=marker, s=30, facecolors="none", edgecolors=color,
                   linewidths=1.1, zorder=3, clip_on=False)                # hollow markers
    pow2_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128], axis="x")
    log10_axis(ax, [5, 10, 20, 50, 100, Y_TOP], axis="y")          # the end of the axis carries a tick and a label
    ax.set_ylim(ax.get_ylim()[0], Y_TOP)
    ax.set_xlabel("NNZs per row")
    ax.set_ylabel("Time per row (ns)")
    ax.yaxis.grid(True, linestyle="dotted", zorder=0)
    ax.tick_params(axis="both", which="both", direction="in", color="#606060")
    ax.set_title(TITLE)
    # legend: the measured markers of the three passes in the first column, their model lines in the second
    handles = [Line2D([], [], linestyle="none", marker=m, markersize=5.5, markerfacecolor="none", markeredgecolor=c,
                      markeredgewidth=1.1, label=f"{lab} (measured)") for _, lab, c, m in PASSES]
    handles += [Line2D([], [], color=c, linewidth=1.5, label=f"{lab} (model)") for _, lab, c, _ in PASSES]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=8, ncol=2, columnspacing=1.0,
              handlelength=1.6, handletextpad=0.5, labelspacing=0.3, borderaxespad=0.3)
    c.fig.tight_layout()
    out = HERE / "kernel_model.pdf"
    c.save(out, format="pdf", bbox_inches="tight")
    print("wrote", out)
