"""Breakdown of the epoch time (white-box model terms, ms per epoch per GPU; black tick = measured epoch time) of the
configuration the paper reports at every GPU count: breakdown_<graph>.pdf (absolute) and breakdown_<graph>_share.pdf
(normalised to the model total), and the three layouts at 64 GPUs, breakdown_layouts_<graph>.pdf.  Where the measured
epoch exceeds the model, the excess is drawn as a hatched grey segment ("sync / wait": ranks waiting inside collectives,
which the model does not contain).  breakdown_preview.png shows every panel at once."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, Canvas, PANEL  # noqa: E402

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "breakdown.csv", comment="#")
df["other_ar"] = df["other"] + df["ar"]                       # optimiser + gradient all-reduce, one segment in the first figure

# stacking order (bottom -> top), legend name, color (validated: Okabe-Ito with a lighter purple and sienna), hatch
PARTS = [("attn", "Attention kernels", "#0072B2", ""),
         ("gemm", "GEMM (projections, FFN)", "#D55E00", "//"),
         ("spmm", "GCN SpMM", "#009E73", "xx"),
         ("ew", "Elementwise (norm, dropout, casts)", "#7B4EA3", ""),
         ("other_ar", "Optimiser, gradient all-reduce", "#BBBBBB", ""),
         ("gather_col", "Gather, column group", "#E69F00", "\\\\"),
         ("rs_col", "Reduce-scatter, column group", "#CC79A7", "//"),
         ("row", "Row-group collectives", "#56B4E9", "xx"),
         ("a2a", "All-to-all, head group", "#A0522D", "\\\\"),
         ("sync", "Measured $-$ model (sync / wait)", "#FFFFFF", "...")]
GRAPH_NAME = {"products": "ogbn-products, width 128", "reddit": "Reddit, width 128", "papers": "papers100M, width 64 (GCN + GT)"}
LW = 0.5


def stacked(ax, sub, xlabels, share=False, width=0.72):
    """Stacked bars of the model terms of the rows `sub` (one bar per row) at x = 0, 1, ...; the measured epoch as a tick."""
    for i, r in enumerate(sub.itertuples()):
        scale = 100.0 / r.model_ms if share else 1.0
        bottom = 0.0
        for key, name, color, hatch in PARTS:
            v = getattr(r, "sync_ms" if key == "sync" else key) * scale
            if v <= 0:
                continue
            ax.bar(i, v, bottom=bottom, width=width, color=color, hatch=hatch, edgecolor="black", linewidth=LW, zorder=3)
            bottom += v
        if not share and pd.notna(r.measured_ms):
            ax.scatter([i], [r.measured_ms], marker="_", s=220, color="black", linewidths=1.6, zorder=5)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(xlabels)
    ax.set_xlim(-0.6, len(sub) - 0.4)
    ax.yaxis.grid(True, linestyle="dotted", zorder=0)
    ax.tick_params(axis="both", direction="in", color="#606060")


def grid_labels(ax, sub, y=-0.085):
    """The grid of each bar under its tick label (small, grey)."""
    for i, r in enumerate(sub.itertuples()):
        lab = r.label.replace("x", r"$\times$")
        if pd.isna(r.measured_ms):
            lab += "*"
        ax.text(i, y, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color="#404040")


def legend_handles():
    hs = [Patch(facecolor=c, hatch=h, edgecolor="black", linewidth=LW, label=n) for _, n, c, h in PARTS]
    hs.append(Line2D([], [], marker="_", color="black", markersize=12, markeredgewidth=1.6, linestyle="none", label="Measured epoch"))
    return hs


def panel_scaling(gname, out, share, legend, figsize=PANEL):
    sub = df[(df["panel"] == "scaling") & (df["graph"] == gname)].sort_values("gpus")
    c = Canvas(figsize)
    stacked(c.ax, sub, [str(g) for g in sub["gpus"]], share=share)
    grid_labels(c.ax, sub)
    c.ax.set_xlabel("Number of GPUs", labelpad=20)
    c.ax.set_ylabel("Share of the epoch (%)" if share else "Time per epoch (ms)")
    c.ax.set_title(GRAPH_NAME[gname])
    if share:
        c.ax.set_ylim(0, 100)
    else:
        c.ax.set_ylim(0, max(sub["model_ms"].max(), sub["measured_ms"].max()) * 1.08)
    if legend:
        c.ax.legend(handles=legend_handles(), frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    labelspacing=0.35, handlelength=1.6)
    c.fig.tight_layout()
    c.save(HERE / out, format="pdf", bbox_inches="tight")
    c.save(HERE / out.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(c.fig)
    print("wrote", HERE / out)


def panel_layouts(gname, out, legend, figsize=PANEL):
    sub = df[(df["panel"] == "layouts") & (df["graph"] == gname)]
    c = Canvas(figsize)
    stacked(c.ax, sub, [l.split(" ")[0] for l in sub["label"]], width=0.6)
    grid_labels(c.ax, sub)
    c.ax.set_xlabel("Layout, 64 GPUs", labelpad=20)
    c.ax.set_ylabel("Time per epoch (ms)")
    c.ax.set_title(GRAPH_NAME[gname])
    c.ax.set_ylim(0, max(sub["model_ms"].max(), sub["measured_ms"].max()) * 1.08)
    if legend:
        c.ax.legend(handles=legend_handles(), frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    labelspacing=0.35, handlelength=1.6)
    c.fig.tight_layout()
    c.save(HERE / out, format="pdf", bbox_inches="tight")
    c.save(HERE / out.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(c.fig)
    print("wrote", HERE / out)


# breakdown_comm.pdf: the 2D and the 3D configuration of every GPU count, both graphs in one figure, the epoch split into
# compute (attention, GEMM, elementwise + optimiser) and the four collective kinds (AG = all-gathers over the row and column
# groups, RS = reduce-scatters over both, A2A = the head group's all-to-alls, AR = the gradient all-reduce)
COMM = [("attn", "Attention kernels", "#0072B2", ""),
        ("gemm", "GEMM (projections, FFN)", "#D55E00", "//"),
        ("ewo", "Elementwise, optimiser", "#7B4EA3", ""),
        ("ag", "All-gather (AG)", "#E69F00", "\\\\"),
        ("rs", "Reduce-scatter (RS)", "#CC79A7", "//"),
        ("a2a", "All-to-all (A2A)", "#A0522D", "xx"),
        ("ar", "All-reduce (AR)", "#56B4E9", "xx"),
        ("sync", "Measured $-$ model (sync / wait)", "#FFFFFF", "...")]


def panel_comm(out, share=False, figsize=(8.4, 3.3)):
    """One figure, two panels (ogbn-products, Reddit): at every GPU count the 2D bar (best fused grid) and the 3D bar (best
    head grid), stacked by COMM; black tick = measured epoch; the per-bar labels name the family and the grid."""
    sub = df[df["panel"] == "comm"].copy()
    sub["ewo"] = sub["ew"] + sub["other"] + sub["spmm"]
    sub["ag"] = sub["ag_col"] + sub["ag_row"]
    sub["rs"] = sub["rs_col"] + sub["rs_row"]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, gname in zip(axes, ("products", "reddit")):
        s = sub[sub["graph"] == gname].sort_values(["gpus", "family"])          # "fused" (2D) before "head" (3D)
        gpus = sorted(s["gpus"].unique())
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#606060")
        ymax = 0.0
        for r in s.itertuples():
            x = gpus.index(r.gpus) + (-0.2 if r.family == "fused" else 0.2)
            scale = 100.0 / r.model_ms if share else 1.0
            bottom = 0.0
            for key, name, color, hatch in COMM:
                v = (r.sync_ms if key == "sync" else getattr(r, key)) * scale
                if v <= 0:
                    continue
                ax.bar(x, v, bottom=bottom, width=0.36, color=color, hatch=hatch, edgecolor="black", linewidth=LW, zorder=3)
                bottom += v
            if not share:
                ax.scatter([x], [r.measured_ms], marker="_", s=150, color="black", linewidths=1.6, zorder=5)
                ymax = max(ymax, r.measured_ms, r.model_ms)
            ax.text(x, -0.025, "2D" if r.family == "fused" else "3D", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7.5, color="#404040")
            ax.text(x, -0.105, r.label.replace("x", r"$\times$"), transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=6.5, color="#404040")
        ax.set_xticks(range(len(gpus)))
        ax.set_xticklabels([str(g) for g in gpus])
        ax.tick_params(axis="x", pad=30)
        ax.set_xlim(-0.6, len(gpus) - 0.4)
        ax.yaxis.grid(True, linestyle="dotted", zorder=0)
        ax.tick_params(axis="both", direction="in", color="#606060")
        ax.set_title(GRAPH_NAME[gname])
        ax.set_xlabel("Number of GPUs")
        ax.set_ylabel("Share of the epoch (%)" if share else "Time per epoch (ms)")
        ax.set_ylim(0, 100 if share else ymax * 1.08)
    hs = [Patch(facecolor=c, hatch=h, edgecolor="black", linewidth=LW, label=n) for _, n, c, h in COMM]
    if not share:
        hs.append(Line2D([], [], marker="_", color="black", markersize=12, markeredgewidth=1.6, linestyle="none", label="Measured epoch"))
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.2),
               handlelength=1.6, columnspacing=1.2, labelspacing=0.35)
    fig.tight_layout()
    fig.savefig(HERE / out, format="pdf", bbox_inches="tight")
    fig.savefig(HERE / out.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", HERE / out)


def legend_figure(out, ncol=4):
    """The shared legend of the panels, as its own figure (placed under the three panels in the paper)."""
    fig = plt.figure(figsize=(8.0, 0.9))
    fig.legend(handles=legend_handles(), loc="center", ncol=ncol, frameon=False, fontsize=9, handlelength=1.8,
               columnspacing=1.4, labelspacing=0.4)
    fig.savefig(HERE / out, format="pdf", bbox_inches="tight")
    fig.savefig(HERE / out.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", HERE / out)


def preview(out):
    """Every panel in one PNG: row 1 absolute, row 2 shares, row 3 the 64-GPU layouts."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
    for j, gname in enumerate(("products", "reddit", "papers")):
        sub = df[(df["panel"] == "scaling") & (df["graph"] == gname)].sort_values("gpus")
        for i, share in enumerate((False, True)):
            ax = axes[i][j]
            stacked(ax, sub, [str(g) for g in sub["gpus"]], share=share)
            grid_labels(ax, sub)
            ax.set_title(GRAPH_NAME[gname])
            ax.set_xlabel("Number of GPUs", labelpad=20)
            ax.set_ylabel("Share of the epoch (%)" if share else "Time per epoch (ms)")
            ax.set_ylim(0, 100 if share else max(sub["model_ms"].max(), sub["measured_ms"].max()) * 1.08)
        ax = axes[2][j]
        if gname == "papers":
            ax.axis("off")
            ax.legend(handles=legend_handles(), frameon=False, fontsize=9, loc="center", labelspacing=0.5)
            continue
        sub = df[(df["panel"] == "layouts") & (df["graph"] == gname)]
        stacked(ax, sub, [l.split(" ")[0] for l in sub["label"]], width=0.6)
        grid_labels(ax, sub)
        ax.set_title(GRAPH_NAME[gname] + ", 64 GPUs")
        ax.set_xlabel("Layout", labelpad=20)
        ax.set_ylabel("Time per epoch (ms)")
        ax.set_ylim(0, max(sub["model_ms"].max(), sub["measured_ms"].max()) * 1.08)
    fig.suptitle("Epoch time breakdown: white-box model terms (bars), measured epoch (tick); * = not measured", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", HERE / out)


with env():
    for gname in ("products", "reddit", "papers"):
        panel_scaling(gname, f"breakdown_{gname}.pdf", False, False)
        panel_scaling(gname, f"breakdown_{gname}_share.pdf", True, False)
    for gname in ("products", "reddit"):
        panel_layouts(gname, f"breakdown_layouts_{gname}.pdf", False)
    legend_figure("breakdown_legend.pdf")
    panel_comm("breakdown_comm.pdf")
    panel_comm("breakdown_comm_share.pdf", share=True)
    preview("breakdown_preview.png")
