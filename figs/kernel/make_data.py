"""data_softmax.csv of the kernel figure: forward ms of one plain-GT sparse attention layer with exact softmax, parsed from
the log of gt3d/scripts/sbatch_kernel_softmax.sh (bench/bench_kernel_external.py --plain --norm softmax), one A100-40GB.
DF-GNN = the fastest of its variants that ran and returned the correct output for the configuration (the benchmark
refuses the variants that are wrong or do not fit the block, see DFGNNImpl there).

    python make_data.py [path/to/sbatch_kernel_softmax_<job>.log]
"""
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "gt3d/logs/sbatch_kernel_softmax_61598117.log"
BLOCKS = {"arxiv": "arxiv", "products16": "products 4x4", "reddit4": "Reddit 2x2", "products4": "products 2x2"}
CONFIGS = ["2x40", "8x32"]
IMPLS = {"PyG": ["official"], "cuSPARSE": ["cusparse"], "DF-GNN": ["dfgnn_hyper_inf", "dfgnn_tiling", "dfgnn_hyper"],
         "GT3D": ["triton"]}

fwd, edges, why, graph = {}, {}, {}, None
for line in LOG.read_text().splitlines():
    m = re.match(r"=== (\S+) plain: rows \d+ cols \d+ edges (\d+)", line)
    if m:
        graph, edges[m.group(1)] = m.group(1), int(m.group(2)) / 1e6
        continue
    m = re.match(r"\s+(\d+x\d+)\s+(\S+)\s+(.*)", line)
    if not m or graph is None:
        continue
    cfg, impl, rest = m.groups()
    t = re.match(r"fwd\s+([\d.]+) ms", rest)
    if t:
        fwd[graph, cfg, impl] = float(t.group(1))
    else:
        why[graph, cfg, impl] = "OOM" if rest.startswith("OOM") else rest[:60]

rows = ["# One sparse attention layer of a plain GT (graph edges + self loops, one edge type, dot-product scores, exact per-row",
        "# softmax), forward ms of one call on one A100-SXM4-40GB (mean of 10 after 3 warm-up calls), fp32. Empty = out of memory",
        "# (PyG) or no DF-GNN variant that is correct and fits the block (variant = the DF-GNN kernel of the reported time).",
        "# Rank-(0,0) blocks of the grids: arxiv = full graph, products 4x4 = products16, Reddit 2x2 = reddit4, products 2x2 = products4.",
        f"# Source: gt3d/logs/{LOG.name} (bench/bench_kernel_external.py --plain --norm softmax), make_data.py.",
        "block,edges_M,config,impl,fwd_ms,variant"]
for g, block in BLOCKS.items():
    for cfg in CONFIGS:
        for name, keys in IMPLS.items():
            got = {k: fwd[g, cfg, k] for k in keys if (g, cfg, k) in fwd}
            if got:
                k = min(got, key=got.get)
                rows.append(f"{block},{edges[g]:.1f},{cfg},{name},{got[k]:.2f},{k.replace('dfgnn_', '') if name == 'DF-GNN' else ''}")
            else:
                rows.append(f"{block},{edges[g]:.1f},{cfg},{name},,")
                print(f"missing {block} {cfg} {name}: " + "; ".join(why.get((g, cfg, k), "not run") for k in keys))
(HERE / "data_softmax.csv").write_text("\n".join(rows) + "\n")
print("wrote", HERE / "data_softmax.csv")
