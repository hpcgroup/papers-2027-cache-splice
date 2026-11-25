"""Same model, same batches: GT3D against TorchGT, parsed from gt3d/logs/sbatch_samebatch_<job>.log and samebatch80g_<job>.log
(scripts/sbatch_samebatch.sh; the latter on A100-80GB nodes).

Model: TorchGT's node-level Graphormer, 4 layers, hidden 128, ffn 128, 32 heads (a setting for this comparison only, so that
TorchGT's head-only parallelism reaches 32 GPUs).  Batches: TorchGT's own dataset files, its random 6/2/2 split (seed 42) and
its batches, 2 per epoch on both graphs (arxiv seq_len 64000, products seq_len 870000).

GT3D: bench/torchgt_batch_bench.py, exact attention on the real batch topology + global token, each batch timed on its own
(--batch_index, mean of 10 iterations); epoch = batch 0 + batch 1; per GPU count the faster of the two 2D grids the job ran.
TorchGT: the released code, --attn_type sparse with --reorder (random blocks, its default mode) and without (real topology),
12 epochs, no evaluation.  Its printed "Epoch Time" is the running mean over epochs 5..k (epochs 1-4 are not recorded), so the
epoch-12 line is the mean of epochs 5-12.  Both time forward + backward + gradient all-reduce + optimizer step only.

Memory (mem_gb): peak torch.cuda.max_memory_allocated per GPU.  GT3D = max over the two batches of the timed iterations of
that grid; TorchGT = over the whole 12-epoch run (appended to its epoch-12 line, measurement-only patch of 2026-11-13).
mem_kind "max-rank" = max over all ranks (both systems from 2026-11-13); "rank0" = GPU 0 only (older GT3D lines).

hw: "40GB" or "80GB" from the nvidia-smi line at the top of each log.  Rows are keyed by (graph, series, gpus, hw).

Logs are read in job order and a later job replaces an earlier entry, except that an error never replaces a number or an OOM.
GT3D errors (the Triton cache hitting the home quota, 2026-11-12 morning) give no row.  TorchGT rows with status "error" are
failures other than out-of-memory: at 32 GPUs on products, gt_sp/utils.py:504 asserts seq_len % P == 0 and 870000 % 32 = 16
(job 61202462); the unbound attn_bias of job 61173940 was fixed before the later jobs.  So 32 GPUs on products use seq_len
870016 (the nearest multiple of 32, batches of 870016 + 599401 nodes) for BOTH systems (job 61227746, 2026-11-13); the column
seq_len records the batch size of every row.
"""
import re
from pathlib import Path

LOGS = Path(__file__).resolve().parents[3] / "gt3d" / "logs"
HERE = Path(__file__).resolve().parent
GRAPH = {"arxiv": "ogbn-arxiv", "products": "ogbn-products"}
TGT_MODE = {"random blocks": "TorchGT (random blocks)", "real topology": "TorchGT (real topology)"}


def parse(path, job):
    gpus, hw = None, "40GB"
    gt3d = {}            # (graph, gpus, grid) -> {batch: (ms, mem_gb, max_rank)}
    tgt = {}             # (graph, series, gpus) -> (status, ms, mem_gb)
    sl = {}              # gt3d (graph, gpus, grid) or tgt (graph, series, gpus) -> seq_len
    sec = None
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"### samebatch nodes=(\d+) gpus=(\d+)", line)
        if m:
            gpus = int(m.group(2))
            continue
        m = re.match(r"NVIDIA A100-SXM4-(\d+)GB", line)
        if m:
            hw = f"{m.group(1)}GB"
            continue
        m = re.match(r"### GT3D (\w+) (?:grid (\d+)x(\d+)|1 GPU) batch (\d)", line)
        if m:
            ds, r, c, b = m.groups()
            sec = ("gt3d", GRAPH[ds], 1 if r is None else gpus, "1" if r is None else f"{r}x{c}", int(b))
            continue
        m = re.match(r"### TorchGT (\w+) (random blocks|real topology)", line)
        if m:
            sec = ("tgt", GRAPH[m.group(1)], TGT_MODE[m.group(2)], gpus)
            tgt[sec[1:]] = ("error", None, None)
            continue
        if line.startswith("###"):
            sec = None
            continue
        if sec is None:
            continue
        if sec[0] == "gt3d":
            m = re.search(r"iteration ([\d.]+) ms", line)
            if line.startswith("RESULT") and m:
                mm = re.search(r"peak mem ([\d.]+) GB( max-rank)?", line)
                gt3d.setdefault(sec[1:4], {})[sec[4]] = (float(m.group(1)), float(mm.group(1)), bool(mm.group(2)))
                sl[sec[1:4]] = int(re.search(r"seq_len=(\d+)", line).group(1))
        else:
            if line.startswith("[torchgt_srun]"):
                sl[sec[1:]] = int(re.search(r"--seq_len (\d+)", line).group(1))
            m = re.match(r"Epoch: (\d+), Loss: [\d.]+, Epoch Time: ([\d.]+)s", line)
            if m:
                mm = re.search(r"Peak memory: ([\d.]+) GB", line)
                tgt[sec[1:]] = ("ok" if int(m.group(1)) == 12 else "partial", float(m.group(2)) * 1000,
                                float(mm.group(1)) if mm else None)
            elif "AssertionError" in line and tgt[sec[1:]][0] == "error":
                tgt[sec[1:]] = ("error", None, None)
            elif "OutOfMemory" in line or "out of memory" in line:
                status, ms, mem = tgt[sec[1:]]
                # an OOM after an epoch line is the released code's rank-0 evaluation, not training
                tgt[sec[1:]] = ("eval-OOM", ms, mem) if status == "partial" else ("OOM", None, None)
    rows = {}
    for (graph, g, grid), bt in gt3d.items():
        if set(bt) == {0, 1}:
            mem = max(bt[0][1], bt[1][1])
            kind = "max-rank" if (bt[0][2] and bt[1][2]) or g == 1 else "rank0"
            rows.setdefault((graph, "GT3D", g, hw), []).append(
                (bt[0][0] + bt[1][0], grid, job, sl[(graph, g, grid)], mem, kind))
    for (graph, series, g), (status, ms, mem) in tgt.items():
        if status in ("ok", "OOM", "error"):
            rows[(graph, series, g, hw)] = [(ms, status, job, sl.get((graph, series, g), ""), mem,
                                             "max-rank" if mem is not None else "")]
    return rows


def job_of(p):
    return int(p.stem.rsplit("_", 1)[1])


best = {}
logs = sorted(list(LOGS.glob("sbatch_samebatch_*.log")) + list(LOGS.glob("samebatch80g_*.log")), key=job_of)
for path in logs:
    job = str(job_of(path))
    for key, cands in parse(path, job).items():
        if key[1] == "GT3D":
            ms, grid, j, n, mem, kind = min(cands)
            others = ", ".join(f"{c[1]} {c[0]:.1f}" for c in sorted(cands)[1:])
            best[key] = ("ok", ms, mem, kind, grid + (f" (also {others})" if others else ""), j, n)
        else:
            ms, status, j, n, mem, kind = cands[0]
            if status == "error" and key in best and best[key][0] in ("ok", "OOM"):
                continue
            best[key] = (status, ms, mem, kind, "seq_len % P != 0" if status == "error" and key[2] == 32 else "", j, n)


def fmt(v, spec):
    return "" if v is None else format(v, spec)


out = HERE / "data.csv"
with open(out, "w") as f:
    f.write("# " + __doc__.strip().replace("\n", "\n# ") + "\n")
    f.write("graph,series,gpus,hw,ms,mem_gb,mem_kind,status,seq_len,config,job\n")
    for (graph, series, g, hw), (status, ms, mem, kind, cfg, j, n) in sorted(best.items()):
        f.write(f"{graph},{series},{g},{hw},{fmt(ms, '.1f')},{fmt(mem, '.2f')},{kind},{status},{n},\"{cfg}\",{j}\n")
        print(f"{graph:14s} {series:24s} {g:3d} {hw}  {status:5s} {fmt(ms, '8.1f'):>8s} ms  {fmt(mem, '6.2f'):>6s} GB "
              f"{kind:8s}  sl {n}  {cfg}  [job {j}]")
print("wrote", out)
