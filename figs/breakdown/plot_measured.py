"""Measured breakdown of the epoch by kernel category (measured.csv, from the profiling runs; no model term): at every GPU
count the 2D bar (best fused grid) and the 3D bar (head grid) for ogbn-products and Reddit in one figure.  The stack is the
mean device kernel time per rank of one training step: attention kernels, GEMM, elementwise, optimizer, and the NCCL kernels
by collective kind (AG, RS, A2A, AR).  The compute categories are the kernel times as measured; the communication block is
the rest of the measured epoch (main table 2026-10-31, same chain), epoch - compute kernels, split into the four kinds in the
proportions of their NCCL kernel times.  An NCCL kernel's time includes the wait for its peers, and under the profiler that
wait also absorbs the launch skew the profiler itself causes (and the pipelined reduce-scatter overlaps compute on purpose),
so the raw NCCL sum can exceed the epoch's communication share; the proportions are kept, the total is the measured one.
When the NCCL kernels sum to less than epoch - compute, the remainder is drawn hatched (GPU idle: launch gaps, skew outside
NCCL kernels).  The bars therefore add up to the measured epoch exactly.  breakdown_comm_measured.pdf (ms), _share.pdf (%)."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import env  # noqa: E402

HERE = Path(__file__).resolve().parent
# compute categories from the profiler runs (CUDA kernel time per category, mean over ranks), the collectives from the timer
# runs (ms per epoch with their waits, mean over ranks); the bar is the measured epoch of the main table: what neither
# accounts for is GPU idle; when compute + collectives exceed the epoch (the pipelined reduce-scatter overlaps compute, and a
# collective's wait can overlap another rank's compute), the collectives are scaled down to the remainder
prof = pd.read_csv(HERE / "measured.csv", comment="#")
tim = pd.read_csv(HERE / "timers.csv", comment="#")
tim = tim.rename(columns={f"{k}_{s}": f"t_{k}_{s}" for k in ("ag", "rs", "a2a", "ar", "attn") for s in ("mean", "max")})
df = prof.merge(tim[["graph", "gpus", "family"] + [c for c in tim.columns if c.startswith("t_")] + ["epoch_timers_ms"]],
                on=["graph", "gpus", "family"], how="inner")
COMP = ["attn_mean", "gemm_mean", "ew_mean", "optim_mean"]
KINDS = ["ag", "rs", "a2a", "ar"]
df["epoch_ms"] = df["epoch_ms"].fillna(df["epoch_timers_ms"])            # 8 / 32 GPUs: no main-table point, the timer run's own epoch
df["comp"] = df[COMP].sum(axis=1)
df["comm"] = sum(df[f"t_{k}_mean"] for k in KINDS)
df["rest"] = (df["epoch_ms"] - df["comp"]).clip(lower=0.0)
df["scale"] = (df["rest"] / df["comm"]).where(df["comm"] > df["rest"], 1.0).fillna(1.0)
for k in KINDS:
    df[k] = df[f"t_{k}_mean"] * df["scale"]
df["idle"] = (df["rest"] - df["comm"] * df["scale"]).clip(lower=0.0)
df["commx"] = df["comm"] * df["scale"]                                    # communication, as drawn (after the overlap scaling)

PARTS = [("attn_mean", "Attention kernels", "#0072B2", ""),
         ("gemm_mean", "GEMM (projections, FFN)", "#D55E00", "//"),
         ("ew_mean", "Elementwise (norm, dropout, casts)", "#7B4EA3", ""),
         ("optim_mean", "Optimiser", "#BBBBBB", ""),
         ("ag", "All-gather (AG)", "#E69F00", "\\\\"),
         ("rs", "Reduce-scatter (RS)", "#CC79A7", "//"),
         ("a2a", "All-to-all (A2A)", "#A0522D", "xx"),
         ("ar", "All-reduce (AR)", "#56B4E9", "xx"),
         ("idle", "GPU idle (launch gaps, skew)", "#FFFFFF", "...")]
# the two-way split: computation (attention, GEMM, elementwise, optimizer kernels) vs communication (the collectives with their waits)
PARTS2 = [("comp", "Computation", "#0072B2", ""),
          ("commx", "Communication", "#E69F00", "\\\\"),
          ("idle", "GPU idle (launch gaps, skew)", "#FFFFFF", "...")]
GRAPH_NAME = {"products": "ogbn-products, width 128", "reddit": "Reddit, width 128"}
LW = 0.5


def panel(out, share=False, figsize=(8.4, 3.3), parts=PARTS):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, gname in zip(axes, ("products", "reddit")):
        s = df[df["graph"] == gname].sort_values(["gpus", "family"])          # "fused" (2D) before "head" (3D)
        gpus = sorted(s["gpus"].unique())
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#606060")
        ymax = 0.0
        for r in s.itertuples():
            x = gpus.index(r.gpus) + (-0.2 if r.family == "fused" else 0.2)
            total = r.epoch_ms
            scale = 100.0 / total if share else 1.0
            bottom = 0.0
            for key, name, color, hatch in parts:
                v = getattr(r, key) * scale
                if v <= 0:
                    continue
                ax.bar(x, v, bottom=bottom, width=0.36, color=color, hatch=hatch, edgecolor="black", linewidth=LW, zorder=3)
                bottom += v
            ymax = max(ymax, total)
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
    hs = [Patch(facecolor=c, hatch=h, edgecolor="black", linewidth=LW, label=n) for _, n, c, h in parts if (df[_] > 0).any()]
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.2 if len(parts) > 4 else -0.08),
               handlelength=1.6, columnspacing=1.2, labelspacing=0.35)
    fig.tight_layout()
    fig.savefig(HERE / out, format="pdf", bbox_inches="tight")
    fig.savefig(HERE / out.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", HERE / out)


with env():
    panel("breakdown_comm_measured.pdf")
    panel("breakdown_comm_measured_share.pdf", share=True)
    panel("breakdown_comm_measured_cc.pdf", parts=PARTS2)                      # computation vs communication only
    panel("breakdown_comm_measured_cc_share.pdf", share=True, parts=PARTS2)
