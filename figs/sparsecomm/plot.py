"""Fig. 11: sparsity-aware communication -- (a) measured cost of the sparse collective relative to the dense one as a
function of the referenced share, for a column group with one GPU per node and one with four; (b) the referenced
share of a column block as a function of P_d under the random node order (analytic, with the expander floor)."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env, LinePlot, PANEL, PALETTE, pow2_axis, finish  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP_DEG = 3

with env():
    # (a) measured: sparse / dense time vs share
    c = pd.read_csv(HERE / "curve.csv", comment="#")
    rows = []
    name = {"col4:1pn": "1 GPU/node", "col16:4pn": "4 GPUs/node"}
    for r in c.itertuples():
        rows.append((r.share, r.sparse_gather_ms / r.dense_ms, f"All-gather, {name[r.group]}"))
        rows.append((r.share, r.sparse_rs_ms / r.dense_ms, f"Reduce-scatter, {name[r.group]}"))
    d = pd.DataFrame(rows, columns=["share", "ratio", "collective"])
    order = ["All-gather, 1 GPU/node", "Reduce-scatter, 1 GPU/node", "All-gather, 4 GPUs/node",
             "Reduce-scatter, 4 GPUs/node"]
    pal = dict(zip(order, [PALETTE[0], PALETTE[0], PALETTE[1], PALETTE[1]]))
    chart = LinePlot()
    chart.plot(data=d, x="share", y="ratio", hue="collective", style="collective", hue_order=order,
               style_order=order, palette=pal, xlabel="Share of rows referenced", ylabel="Sparse / dense time",
               figsize=PANEL, xlim=(0, 1.02), ylim=(0, 4))
    chart.ax.axhline(1.0, color="#606060", linewidth=1, linestyle=(0, (4, 2)), zorder=1)
    chart.ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    chart.ax.set_yticks([0, 1, 2, 3, 4])
    finish(chart, HERE / "sparse_measured.pdf", legend=order, loc="upper left", fontsize=8.5)

    # (b) analytic share of a column block that a GPU references, vs the number of row blocks P_d
    g = pd.read_csv(HERE / "graphs.csv", comment="#")
    pd_vals = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
    rows = []
    for r in g.itertuples():
        d_real = r.d_attn - 2 * EXP_DEG
        for p in pd_vals:
            pgf = (1 + d_real / (p * r.kappa)) ** (-r.kappa)          # E[(1 - 1/p)^deg], negative-binomial out-degree
            need = 1 - pgf * (1 - 1 / p) ** (2 * EXP_DEG)              # a real edge or one of the 2 d_e expander edges lands in the block
            rows.append((p, 1 / p + (1 - 1 / p) * need, r.graph))      # the plan's share: own chunk included
    d = pd.DataFrame(rows, columns=["P_d", "share", "graph"])
    order = ["Reddit", "ogbn-products", "ogbn-papers100M"]
    pal = dict(zip(order, [PALETTE[2], PALETTE[1], PALETTE[0]]))
    chart = LinePlot()
    chart.plot(data=d, x="P_d", y="share", hue="graph", style="graph", hue_order=order, style_order=order,
               palette=pal, xlabel="Row blocks $P_d$", ylabel="Share of column block referenced", figsize=PANEL,
               xlim=(3.2, 1280), ylim=(0, 1.05))
    pow2_axis(chart.ax, [4, 16, 64, 256, 1024])
    chart.ax.set_ylim(0, 1.05)
    # sparse pays when share / (a2a/AG) <= 0.9; a2a/AG = 0.89 for a column group with one GPU per node
    chart.ax.axhline(0.9 * 0.89, color="#606060", linewidth=1, linestyle=(0, (4, 2)), zorder=1)
    chart.ax.text(4.2, 0.9 * 0.89 - 0.07, "sparse pays below (1 GPU/node)", fontsize=9, color="#606060")
    chart.ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    finish(chart, HERE / "sparse_share.pdf", legend=order, loc="lower left", fontsize=9)
