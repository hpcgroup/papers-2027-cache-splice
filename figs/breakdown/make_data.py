"""Writes breakdown.csv: the white-box model's time terms (ms per epoch, per GPU) of the configuration the paper reports
at every GPU count, next to the measured epoch time.  The terms are the model's own (gt3d/dist/grid_select.py:
fused_layer_ms / attn_operator_ms("head") + head_gcn_ms / streamed_layer_ms / fixed_step_ms), regrouped into
    attn        attention kernels, forward + backward
    gemm        projections, FFN, pre-/post-MP, their gradients
    spmm        GCN SpMM (GCN + attention model only)
    ew          elementwise passes (norms, dropout, casts, combine)
    other       optimiser, CPU launch floor
    ar          the flat gradient all-reduce (reduce-scatter + all-gather over every rank)
    ag_col      all-gather over the column group (x_col / K|V slices); streamed layer: the EXPOSED part only
    rs_col      reduce-scatter over the column group (dx_col / dK|dV slices); streamed: exposed part only
    ag_row      all-gathers over the row group (Q side, and the backward gather of the partials' gradient)
    rs_row      reduce-scatters over the row group (the partials, and dx_row in the backward)
    a2a         Ulysses all-to-alls inside the head group (3D grids)
plus the coarse columns of the first figure, gather_col = ag_col and row = ag_row + rs_row.
sync = measured - model where positive (ranks waiting inside collectives; the model has no such term).
Panels: "scaling" (the configuration the model picks per GPU count), "layouts" (the three layouts at 64 GPUs) and "comm"
(the 2D and the 3D configuration per GPU count, both graphs: breakdown_comm.pdf, the epoch by collective kind).
Run with the gt3d Python environment (plexus_env); set GT3D to point to another checkout."""
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

GT3D = Path(os.environ.get("GT3D", Path(__file__).resolve().parents[3] / "gt3d"))
sys.path.insert(0, str(GT3D))
from gt3d.dist import grid_select as gs  # noqa: E402
from gt3d.dist.grid_select import (gemm_ms, ew_ms, coll_ms, sparse_coll_ms, sparse_share, attn_ms, spmm_ms,  # noqa: E402
                                   placement, nonempty_fraction, cpu_floor_ms, pipelined_rs_ms, HEAD_OPS, GPUS_PER_NODE as GPN)
from bench.validate_model import graph  # noqa: E402
from bench.model_validation_data import HEADS, MAIN_TABLE_1031  # noqa: E402

CATS = ["attn", "gemm", "spmm", "ew", "other", "ar", "ag_col", "rs_col", "ag_row", "rs_row", "a2a"]


def fixed_parts(P, g, m, p):
    """fixed_step_ms, split: pre-MP / output-head GEMMs -> gemm; elementwise, gradient all-reduce, optimiser -> other."""
    O, d, L = g.n / P, m.hidden, m.layers
    p["gemm"] += sum(gemm_ms(2 * O * g.f_in * d, d, m.tf32, w) for w in (0, 1, 2)) + gemm_ms(2 * O * d * m.classes * 3, d, m.tf32, 0)
    p["ew"] += ew_ms(O * d * 4 * 20 + O * g.f_in * 4, 20) + ew_ms(O * m.classes * 4 * 6, 10)
    n_params = L * 12 * d * d + g.f_in * d + d * m.classes
    lay = (min(P, GPN), max(1, P // GPN))
    p["ar"] += coll_ms("rs", lay, 2 * n_params * 4) + coll_ms("ag", lay, 1) * math.ceil(math.log2(max(P, 2)))
    p["other"] += 0.3 + ew_ms(n_params * 4 * 12, 10)


def tail_parts(O, d, m, p):
    p["ew"] += ew_ms(O * d * 4 * 14, 14) + ew_ms(O * d * 4 * 28, 28)
    p["gemm"] += sum(gemm_ms(8 * O * d * d, d, m.tf32, w) for w in (0, 1, 2))


def fused_parts(pd, po, g, m):
    """One fused 2D layer, forward + backward (mirror of fused_layer_ms)."""
    P = pd * po
    d, H, b = m.hidden, m.heads, m.wire_bytes
    dh = d // H
    R, C, O = g.n / pd, g.n / po, g.n / P
    ea, eg = g.e_attn / P, g.e_gcn / P
    pl = placement(pd, po, 1, GPN)
    sh_c, sh_r = sparse_share(g, pd), sparse_share(g, po)
    w = 2 * d + H
    p = defaultdict(float)
    fl = 2 * d * d * (3 * C + R) if m.with_gcn else 2 * d * d * (2 * C + R)
    # forward
    p["ag_col"] += sparse_coll_ms("ag", pl["col"], (C - O) * d * b, pd, sh_c, m)
    p["ag_row"] += sparse_coll_ms("ag", pl["row"], (R - O) * d * b, po, sh_r, m)
    p["ew"] += ew_ms((C + R) * d * (b + 4) + O * d * b * 2, 4)
    p["gemm"] += gemm_ms(fl, d, m.tf32, 0)
    if m.with_gcn:
        p["spmm"] += spmm_ms(eg, R, C, d, graph=g.name, grid=(pd, po))
    p["attn"] += attn_ms(ea, R, C, H, dh, "fwd", graph=g.name, grid=(pd, po))
    p["ew"] += ew_ms(R * w * 4 * 2 + R * w * (4 + b), 4) + ew_ms(O * w * (b + 4) + O * d * 4 * 2, 3)
    p["rs_row"] += sparse_coll_ms("rs", pl["row"], (R - O) * w * b, po, sh_r, m)
    tail_parts(O, d, m, p)
    # backward
    p["ew"] += ew_ms(O * w * (4 + b) + O * d * 4 * 3, 4) + ew_ms(R * w * (b + 4) + R * w * 4, 3)
    p["ag_row"] += sparse_coll_ms("ag", pl["row"], (R - O) * w * b, po, sh_r, m)
    p["attn"] += attn_ms(ea, R, C, H, dh, "bwd", graph=g.name, grid=(pd, po))
    if m.with_gcn:
        p["spmm"] += spmm_ms(eg, R, C, d, transpose=True, graph=g.name, grid=(pd, po))
    p["gemm"] += gemm_ms(fl, d, m.tf32, 1) + gemm_ms(fl, d, m.tf32, 2)
    p["ew"] += ew_ms(C * d * 4 * (6 if m.with_gcn else 4) + C * d * (4 + b), 4) + ew_ms(R * d * (4 + b), 2) \
        + ew_ms(O * d * (b + 4) * 2 + O * d * 4 * 4, 6)
    n_bs = getattr(m, "bwd_sub", 1)
    rs_col = sparse_coll_ms("rs", pl["col"], (C - O) * d * b / n_bs, pd, sh_c, m)
    if n_bs > 1:                                               # --bwd_sub: pipelined with the column pass (mirror of fused_layer_ms)
        attn_b = attn_ms(ea, R, C, H, dh, "bwd", graph=g.name, grid=(pd, po))
        rs_col = pipelined_rs_ms(rs_col, n_bs, 0.5 * attn_b + gemm_ms(fl, d, m.tf32, 1), 0.5 * attn_b + gemm_ms(fl, d, m.tf32, 2)) \
            + ew_ms(C * d * 4 * 2 * (2 if m.with_gcn else 1), 2 * n_bs)
    p["rs_col"] += rs_col
    p["rs_row"] += sparse_coll_ms("rs", pl["row"], (R - O) * d * b, po, sh_r, m)
    return p


def head_parts(pd, po, pi, g, m):
    """One head-grid layer, forward + backward (mirror of attn_operator_ms(family="head") + head_gcn_ms + the tail)."""
    P = pd * po * pi
    d, H, b = m.hidden, m.heads, m.wire_bytes
    dh = d // H
    pl = placement(pd, po, pi, GPN)
    sh_c, sh_r = sparse_share(g, pd), sparse_share(g, po)
    O = g.n / P
    rows_q, rows_kv = g.n / pd, g.n / po
    ds = d / pi
    Hs = max(H // pi, 1)
    ea = g.e_attn / (pd * po)
    w = ds + Hs
    a2a = (lambda nb: coll_ms("a2a", pl["head"], nb * (1 - 1 / pi), pi)) if pi > 1 else (lambda nb: 0.0)
    p = defaultdict(float)
    # forward
    p["gemm"] += gemm_ms(3 * 2 * O * d * d, d, m.tf32, 0)
    p["a2a"] += 3 * a2a(O * d * b)
    p["ew"] += ew_ms(3 * O * d * (b + 4) * 2, 9)
    p["ag_col"] += 2 * sparse_coll_ms("ag", pl["col"], (rows_kv - O * pi) * ds * b, pd, sh_c, m)
    p["ag_row"] += sparse_coll_ms("ag", pl["row"], (rows_q - O * pi) * ds * b, po, sh_r, m)
    p["ew"] += ew_ms((2 * rows_kv + rows_q) * ds * (b + 4), 3)
    p["attn"] += attn_ms(ea, rows_q, rows_kv, Hs, dh, "fwd", graph=g.name, grid=(pd, po))
    p["ew"] += ew_ms(rows_q * w * 4 * 2 + rows_q * w * (4 + b), 4)
    p["rs_row"] += sparse_coll_ms("rs", pl["row"], (rows_q - O * pi) * w * b, po, sh_r, m)
    p["ew"] += ew_ms(O * pi * w * (b + 4) + O * pi * ds * 8, 3)
    p["a2a"] += a2a(O * d * b)
    # backward
    p["a2a"] += a2a(O * d * b)
    p["ew"] += ew_ms(O * pi * w * (4 + b) + O * pi * ds * 8, 4)
    p["ag_row"] += sparse_coll_ms("ag", pl["row"], (rows_q - O * pi) * w * b, po, sh_r, m)
    p["ew"] += ew_ms(rows_q * w * (b + 4), 3)
    p["attn"] += attn_ms(ea, rows_q, rows_kv, Hs, dh, "bwd", graph=g.name, grid=(pd, po))
    p["rs_row"] += sparse_coll_ms("rs", pl["row"], (rows_q - O * pi) * ds * b, po, sh_r, m)
    n_bs = getattr(m, "bwd_sub", 1)
    rs_col = 2 * sparse_coll_ms("rs", pl["col"], (rows_kv - O * pi) * ds * b / n_bs, pd, sh_c, m)
    if n_bs > 1:                                               # --bwd_sub (mirror of attn_operator_ms, head family)
        attn_b = attn_ms(ea, rows_q, rows_kv, Hs, dh, "bwd", graph=g.name, grid=(pd, po))
        rs_col = pipelined_rs_ms(rs_col, n_bs, 0.5 * attn_b, 0.5 * attn_b) + ew_ms(rows_kv * ds * 4 * 2, 2 * n_bs)
    p["rs_col"] += rs_col
    p["ew"] += ew_ms((2 * rows_kv + rows_q) * ds * (4 + b), 3)
    p["a2a"] += 3 * a2a(O * d * b)
    p["ew"] += ew_ms(3 * O * d * (b + 4), 6)
    p["gemm"] += gemm_ms(3 * 2 * O * d * d, d, m.tf32, 1) + gemm_ms(3 * 2 * O * d * d, d, m.tf32, 2)
    if m.with_gcn:                                                   # head_gcn_ms
        eg = g.e_gcn / (pd * po)
        p["gemm"] += gemm_ms(2 * O * d * d, d, m.tf32, 0)
        p["a2a"] += a2a(O * d * b)
        p["ag_col"] += sparse_coll_ms("ag", pl["col"], (rows_kv - O * pi) * ds * b, pd, sh_c, m)
        p["ew"] += ew_ms(rows_kv * ds * (b + 4), 2)
        p["spmm"] += spmm_ms(eg, rows_q, rows_kv, ds, graph=g.name, grid=(pd, po))
        p["ew"] += ew_ms(rows_q * ds * 4 * 2, 2)
        p["rs_row"] += sparse_coll_ms("rs", pl["row"], (rows_q - O * pi) * ds * b, po, sh_r, m)
        p["a2a"] += a2a(O * d * b)
        p["ew"] += ew_ms(O * d * 4 * 6, 6)
        p["ew"] += ew_ms(O * d * 4 * 6, 6)
        p["a2a"] += a2a(O * d * b)
        p["ag_row"] += sparse_coll_ms("ag", pl["row"], (rows_q - O * pi) * ds * b, po, sh_r, m)
        p["spmm"] += spmm_ms(eg, rows_q, rows_kv, ds, transpose=True, graph=g.name, grid=(pd, po))
        p["ew"] += ew_ms(rows_kv * ds * (4 + b), 2)
        p["rs_col"] += sparse_coll_ms("rs", pl["col"], (rows_kv - O * pi) * ds * b, pd, sh_c, m)
        p["a2a"] += a2a(O * d * b)
        p["gemm"] += gemm_ms(2 * O * d * d, d, m.tf32, 1) + gemm_ms(2 * O * d * d, d, m.tf32, 2)
    tail_parts(O, d, m, p)
    return p


def streamed_parts(pd, po, n_sub, g, m):
    """One streamed layer, forward + backward, overlap on (mirror of streamed_layer_ms): the column collectives count
    only with their exposed part."""
    P = pd * po
    d, H, b = m.hidden, m.heads, m.wire_bytes
    dh = d // H
    R, C, O = g.n / pd, g.n / po, g.n / P
    cs = C / n_sub
    ea, eg = g.e_attn / P / n_sub, g.e_gcn / P / n_sub
    lam_a, lam_g = ea / R, eg / R
    pairs_a = nonempty_fraction(lam_a, g.kappa) * R
    pairs_g = nonempty_fraction(lam_g, g.kappa) * R
    pl = placement(pd, po, 1, GPN)
    sh_c, sh_r = sparse_share(g, pd), sparse_share(g, po)
    w = 2 * d + H
    p = defaultdict(float)
    # forward
    p["ag_row"] += sparse_coll_ms("ag", pl["row"], (R - O) * d * b, po, sh_r, m)
    p["ew"] += ew_ms(R * d * (b + 4), 2)
    p["gemm"] += gemm_ms(2 * R * d * d, d, m.tf32, 0)
    col_sparse = gs.plan_sparse(pl["col"], sh_c, m)
    cse = cs * sh_c if col_sparse else cs                                        # the plan's rows of a sub-block (2026-11-02)
    t_ag = sparse_coll_ms("ag", pl["col"], (cs - O / n_sub) * d * b, pd, sh_c, m)
    proj_ew, proj_gemm = gs.bulk_ms(cse * d * (b + 4), 2), gemm_ms(3 * 2 * cse * d * d, d, m.tf32, 0)
    spmm_f = spmm_ms(eg, pairs_g, cse, d, compact=True)
    spmm_ew = gs.bulk_ms(pairs_g * d * 4 * 3, 2)
    attn_f = attn_ms(ea, pairs_a, cse, H, dh, "fwd", regime="compact", graph=g.name)
    comp = proj_ew + proj_gemm + spmm_f + spmm_ew + attn_f
    p["ag_col"] += n_sub * max(gs.OVERLAP_LEAK * t_ag, t_ag - comp) + t_ag     # exposed, with the overlap leak
    p["gemm"] += n_sub * proj_gemm
    p["ew"] += n_sub * (proj_ew + spmm_ew)
    p["spmm"] += n_sub * spmm_f
    p["attn"] += n_sub * attn_f
    p["ew"] += ew_ms(R * w * (4 + b), 2) + ew_ms(O * w * (b + 4), 2)
    p["rs_row"] += sparse_coll_ms("rs", pl["row"], (R - O) * w * b, po, sh_r, m)
    p["ew"] += ew_ms(O * d * 4 * 14, 14)
    p["gemm"] += gemm_ms(8 * O * d * d, d, m.tf32, 0)
    # backward
    p["ew"] += ew_ms(O * d * 4 * 14, 14) + ew_ms(O * d * 4 * 28, 28)          # tail recompute + grads
    p["gemm"] += gemm_ms(8 * O * d * d, d, m.tf32, 0) + gemm_ms(8 * O * d * d, d, m.tf32, 1) + gemm_ms(8 * O * d * d, d, m.tf32, 2)
    p["ag_row"] += 3 * coll_ms("ag", pl["row"], (R - O) * (w / 3) * b, po)
    p["ew"] += ew_ms(R * w * (b + 4), 3)
    p["ag_row"] += sparse_coll_ms("ag", pl["row"], (R - O) * d * b, po, sh_r, m)      # rowside again (recompute)
    p["ew"] += ew_ms(R * d * (b + 4), 2)
    p["gemm"] += gemm_ms(2 * R * d * d, d, m.tf32, 0)
    attn_b = attn_ms(ea, pairs_a, cse, H, dh, "bwd", regime="compact", graph=g.name)
    spmm_b = spmm_ms(eg, cse, R, d, transpose=True, compact=True)
    gemm_b = gemm_ms(3 * 2 * cse * d * d, d, m.tf32, 1) + gemm_ms(3 * 2 * cse * d * d, d, m.tf32, 2)
    dx_ew = gs.bulk_ms(cse * d * 4 * 6 + cse * d * (4 + b), 6)
    t_rs = sparse_coll_ms("rs", pl["col"], (cs - O / n_sub) * d * b, pd, sh_c, m)
    comp_b = proj_ew + proj_gemm + attn_b + spmm_b + gemm_b + dx_ew
    exposed_b = n_sub * max(gs.OVERLAP_LEAK * (t_ag + t_rs), t_ag + t_rs - comp_b) + (t_ag + t_rs)
    f_ag = t_ag / (t_ag + t_rs) if t_ag + t_rs > 0 else 0.5
    p["ag_col"] += exposed_b * f_ag
    p["rs_col"] += exposed_b * (1 - f_ag)
    p["gemm"] += n_sub * (proj_gemm + gemm_b)
    p["ew"] += n_sub * (proj_ew + dx_ew)
    p["attn"] += n_sub * attn_b
    p["spmm"] += n_sub * spmm_b
    p["gemm"] += gemm_ms(2 * R * d * d, d, m.tf32, 1) + gemm_ms(2 * R * d * d, d, m.tf32, 2)
    p["ew"] += ew_ms(R * d * (4 + b) * 2, 3)
    p["rs_row"] += sparse_coll_ms("rs", pl["row"], (R - O) * d * b, po, sh_r, m)
    p["ew"] += ew_ms(O * d * 4 * 4, 4)
    return p


def breakdown(gname, hidden, family, grid, n_sub, with_gcn, sparse, bwd_sub=1):
    g = graph(gname)
    m = gs.Model(hidden, HEADS[hidden], 4, True, with_gcn=with_gcn, sparse_comm=sparse, bwd_sub=bwd_sub)
    pd, po, pi = grid
    P = pd * po * pi
    if family == "head":
        lp = head_parts(pd, po, pi, g, m)
        ref = gs.predict_head_ms(pd, po, pi, g, m)
        floor = cpu_floor_ms(m, HEAD_OPS)
    elif family == "fused":
        lp = fused_parts(pd, po, g, m)
        ref = gs.predict_fused_ms(pd, po, g, m)
        floor = cpu_floor_ms(m)
    else:
        lp = streamed_parts(pd, po, n_sub, g, m)
        o = gs.streamed_layer_ms(pd, po, n_sub, g, m, GPN, True)
        ref = max(m.layers * (o["fwd"] + o["bwd"]) + gs.fixed_step_ms(P, g, m, GPN), cpu_floor_ms(m) * (1 + 0.5 * n_sub / 8))
        floor = cpu_floor_ms(m) * (1 + 0.5 * n_sub / 8)
    p = {k: m.layers * lp.get(k, 0.0) for k in CATS}
    fixed_parts(P, g, m, p)
    gpu = sum(p.values())
    if floor > gpu:                                                     # launch-bound: the CPU floor is the epoch
        p["other"] += floor - gpu
    tot = sum(p.values())
    assert abs(tot - ref) <= 1e-6 * max(ref, 1.0) + 1e-6, (gname, hidden, family, grid, tot, ref)
    p["gather_col"], p["row"] = p["ag_col"], p["ag_row"] + p["rs_row"]  # the coarse columns of the first figure
    return p, tot


# panel, graph, hidden, gpus, family, (P_d, P_o, P_i), n_sub, with_gcn, sparse, measured ms, label
# measured: bench/model_validation_data.py HEAD_MODEL / ONE_D_MODEL (attention-only, 4-64 GPUs), gt3d/RESULTS.md 2026-10-23
# "256 GPUs" (products attention-only 32x8 --sparse_comm; papers100M streamed h64 16x4 n16 dense 64 GPUs, 64x4 n16 sparse 256)
CONFIGS = [
    # measured: the frozen chain of 2026-10-31 (compiled tail, pipelined backward reduce-scatter chosen by the model = the
    # bwd_sub below, sparse collectives under the plan's rule), jobs 60628818 / 60628819 / 60628820 / 60628821; the model has
    # no term for the compiled tail, so its elementwise bar overstates the measured epoch by ~10 %
    ("scaling", "products", 128, 4, "head", (1, 1, 4), None, False, False, 549.1, "(1,1,4)", 1),
    ("scaling", "products", 128, 16, "head", (2, 2, 4), None, False, False, 222.1, "(2,2,4)", 2),
    ("scaling", "products", 128, 64, "head", (4, 4, 4), None, False, False, 109.5, "(4,4,4)", 2),
    ("scaling", "products", 128, 256, "head", (8, 8, 4), None, False, False, 71.9, "(8,8,4)", 2),
    ("scaling", "reddit", 128, 4, "head", (1, 1, 4), None, False, False, 173.0, "(1,1,4)", 1),
    ("scaling", "reddit", 128, 16, "head", (2, 2, 4), None, False, False, 64.1, "(2,2,4)", 2),
    ("scaling", "reddit", 128, 64, "fused", (8, 8, 1), None, False, False, 34.7, "8x8", 1),
    ("scaling", "reddit", 128, 256, "fused", (32, 8, 1), None, False, True, 39.2, "32x8", 1),
    ("scaling", "papers", 64, 64, "streamed", (16, 4, 1), 16, True, True, 3496.0, "16x4 n16 sp.", 1),
    ("scaling", "papers", 64, 256, "streamed", (64, 4, 1), 16, True, True, 1515.5, "64x4 n16 sp.", 1),
    ("layouts", "products", 128, 64, "head", (4, 4, 4), None, False, False, 109.5, "3D (4,4,4)", 2),
    ("layouts", "products", 128, 64, "fused", (8, 8, 1), None, False, False, 114.2, "2D 8x8", 2),
    ("layouts", "products", 128, 64, "fused", (64, 1, 1), None, False, False, 229.6, "1D 64x1", 2),
    ("layouts", "reddit", 128, 64, "head", (4, 4, 4), None, False, False, 38.7, "3D (4,4,4)", 1),
    ("layouts", "reddit", 128, 64, "fused", (8, 8, 1), None, False, False, 34.7, "2D 8x8", 1),
    ("layouts", "reddit", 128, 64, "fused", (64, 1, 1), None, False, False, 72.3, "1D 64x1", 1),
]
# "comm": the 2D (best fused grid with P_o > 1) and the 3D (best head grid) configuration of every GPU count, both graphs,
# for the breakdown by collective kind (breakdown_comm.pdf): measured ms and the model's bwd_sub from MAIN_TABLE_1031
# (frozen chain of 2026-10-31).  The chain ran with --sparse_comm everywhere; the model's rule reproduces the runs' plan state
# except products 16x16 at 256 GPUs, which ran dense (column share 0.798, log 60628821), so that one is costed dense.
DENSE_RUNS = {("products", 256, (16, 16, 1))}
for (gname, hidden), byP in MAIN_TABLE_1031.items():
    if hidden != 128:
        continue
    for P, pts in sorted(byP.items()):
        for fam, pick in (("fused", min((r for r in pts if r[3] == "fused" and r[1] > 1), key=lambda r: r[4], default=None)),
                          ("head", min((r for r in pts if r[3] == "head"), key=lambda r: r[4], default=None))):
            if pick is None:
                continue
            pd_, po_, pi_, _, ms, nb = pick
            label = f"{pd_}x{po_}" if fam == "fused" else f"({pd_},{po_},{pi_})"
            CONFIGS.append(("comm", gname, hidden, P, fam, (pd_, po_, pi_), None, False, (gname, P, (pd_, po_, pi_)) not in DENSE_RUNS, ms, label, nb))

rows = []
for panel, gname, hidden, gpus, family, grid, n_sub, with_gcn, sparse, meas, label, nb in CONFIGS:
    p, tot = breakdown(gname, hidden, family, grid, n_sub, with_gcn, sparse, nb)
    sync = max(0.0, meas - tot) if meas is not None else 0.0
    rows.append([panel, gname, hidden, gpus, family, f"({grid[0]},{grid[1]},{grid[2]})", label, "" if meas is None else meas,
                 round(tot, 2), round(sync, 2)] + [round(p[k], 3) for k in CATS + ["gather_col", "row"]])
    print(f"{panel:8s} {gname:9s} h{hidden} {gpus:4d} {label:14s} model {tot:8.1f}  measured {meas if meas else float('nan'):8.1f}  "
          + " ".join(f"{k}={p[k]:.1f}" for k in CATS))

out = Path(__file__).resolve().parent / "breakdown.csv"
with open(out, "w", newline="") as f:
    f.write("# Generated by make_data.py: white-box model terms (ms per epoch) of the reported configuration per GPU count, and the\n"
            "# measured epoch time (frozen chain of 2026-10-31); sync_ms = measured - model where positive.  products / Reddit:\n"
            "# attention-only Exphormer, width 128; papers100M: GCN + attention, width 64, streamed layer (column collectives: exposed\n"
            "# part only).  The model has no term for the compiled tail (~10 % of the epoch), so the ticks sit below the bars.\n")
    w = csv.writer(f)
    w.writerow(["panel", "graph", "hidden", "gpus", "family", "grid", "label", "measured_ms", "model_ms", "sync_ms"] + CATS + ["gather_col", "row"])
    w.writerows(rows)
print("wrote", out, len(rows), "rows")
