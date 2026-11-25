"""Time per row of the three passes of the attention kernel on regular blocks (every row and column holds d edges),
from gt3d/bench/kernel_theory.json and kernel_theory_d16.json (bench/kernel_theory.py, one A100-40GB), next to the
analytic model max(L, bytes / B) with the latency L = t_c max(1, d / 16) growing smoothly with the 16-edge chunks a
program loads (no ceil, so the line has no step at 16 edges): data.csv.  The d_h = 16 configurations (2026-11-25,
gt3d/scripts/run_calib_d16.sh) are those of the paper's model (width 128, 8 heads); a row of an 8-head block runs one
program per group of 4 heads, and the bytes are summed over the groups."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "gt3d" / "bench"
SOURCES = [("kernel_theory.json", ("4x32", "2x32", "1x32")), ("kernel_theory_d16.json", ("8x16", "4x16", "2x16", "1x16"))]
CHUNK = 16
HPP = 4                                                            # heads per program (triton_kernel.HEADS_PER_PROGRAM)


def program_bytes(p, h, dh, d):
    """Bytes a program moves: per edge and per row / column, from the kernel's loads and stores."""
    w = 4 * h * dh
    if p == "fwd":
        return d * (2 * w + 8) + (2 * w + 8 * h + 20)
    if p == "row":
        return d * (2 * w + 8) + (3 * w + 12 * h + 20)
    return d * (2 * w + 12 * h + 8) + (4 * w + 20)


def row_bytes(p, h, dh, d):
    """Bytes of all programs of a row: one program per group of HPP heads."""
    return sum(program_bytes(p, min(HPP, h - g0), dh, d) for g0 in range(0, h, HPP))


lines = ["heads,d_head,pass,degree,measured_ns,model_ns,floor_ns,bandwidth_ns"]
n = 612352
for src, cfgs in SOURCES:
    th = json.load(open(BENCH / src))
    B = th["hbm_read_gbs"] * 1e9
    for cfg in cfgs:
        h, dh = map(int, cfg.split("x"))
        reg = th["regular"][cfg]
        for p in ("fwd", "row", "col"):
            t_c = reg[f"{n}_1"][p] * 1e6 / n
            for key, r in reg.items():
                if key == "compiled" or r["n"] != n or r["d"] == 0:
                    continue
                d = r["d"]
                meas = r[p] * 1e6 / n
                bw = row_bytes(p, h, dh, d) / B * 1e9
                fl = max(1.0, d / CHUNK) * t_c
                lines.append(f"{h},{dh},{p},{d},{meas:.3f},{max(fl, bw):.3f},{fl:.3f},{bw:.3f}")
    print(src, f"HBM read {B/1e9:.0f} GB/s")
(HERE / "data.csv").write_text("\n".join(lines) + "\n")
print("wrote", HERE / "data.csv", len(lines) - 1, "rows")
