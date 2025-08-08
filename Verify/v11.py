"""
Report-style Fast-Scalar MonSTER Triad (x, y, z), 12 dims/freq, DIM=768.

What you get in the terminal:
  1) Config summary (DIM, NUM_FREQ, base, top_delta, unit)
  2) Positions (s_q, s_k) and relative displacement Δs
  3) Frequency ladder samples (first N, last N)
  4) Angle/Rapidity samples for those freqs (φ, θx, θy, θz)
  5) Application summary (shapes, dtype)
  6) Per-frequency contribution breakdown (lhs & rhs; by X/Y/Z and totals)
  7) Sanity checks: RoPE identity, per-block Minkowski norm preservation

Math:
  - One 12-D block per frequency: [X4 | Y4 | Z4], each 4D order [t,x,y,z]
  - In each 4D: apply boost along axis (t<->axis), then rotate about axis (other two)
  - SAME inverse-frequency ladder for φ and θ; global unit = 1/top_delta
  - Exact identity: <L(s_q)q, L(s_k)k> = <q, L(s_k-s_q)k>

No 4×4 matrices are materialized—just cosh/sinh/cos/sin scalars per block.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

# -------------------------
# Pretty printing helpers
# -------------------------
def fmtf(x: float, prec=12, show_sign=True):
    sign = "+" if (show_sign and x >= 0) else ""
    return f"{sign}{x:.{prec}f}"

def head_tail(a: np.ndarray, n_head=4, n_tail=2):
    if a.ndim != 1 or a.size <= n_head + n_tail:
        return a
    return np.concatenate([a[:n_head], np.array([np.nan]), a[-n_tail:]])

# -------------------------
# Minkowski metric (+ - - -)
# -------------------------
ETA4 = np.diag([1.0, -1.0, -1.0, -1.0]).astype(np.float64)

def minkowski_dot4(u4: np.ndarray, v4: np.ndarray) -> float:
    return (u4 @ ETA4) @ v4

def minkowski_dot_big(u: np.ndarray, v: np.ndarray, dim: int) -> float:
    total = 0.0
    for i in range(0, dim, 4):
        total += minkowski_dot4(u[i:i+4], v[i:i+4])
    return total

def minkowski_dot_by_freq(u: np.ndarray, v: np.ndarray, dim: int, slice_size=12) -> np.ndarray:
    """Total per-frequency contribution (summing X/Y/Z 4D blocks)."""
    num_freq = dim // slice_size
    out = np.zeros(num_freq, dtype=np.float64)
    for j in range(num_freq):
        b = j * slice_size
        out[j]  = (u[b+0:b+4]   @ ETA4) @ v[b+0:b+4]   # X block
        out[j] += (u[b+4:b+8]   @ ETA4) @ v[b+4:b+8]   # Y block
        out[j] += (u[b+8:b+12]  @ ETA4) @ v[b+8:b+12]  # Z block
    return out

def minkowski_dot_by_freq_axis(u: np.ndarray, v: np.ndarray, dim: int, slice_size=12) -> np.ndarray:
    """Per-frequency, per-axis contributions. Shape: (F, 3) -> columns X,Y,Z."""
    num_freq = dim // slice_size
    out = np.zeros((num_freq, 3), dtype=np.float64)
    for j in range(num_freq):
        b = j * slice_size
        out[j, 0] = (u[b+0:b+4]   @ ETA4) @ v[b+0:b+4]   # X
        out[j, 1] = (u[b+4:b+8]   @ ETA4) @ v[b+4:b+8]   # Y
        out[j, 2] = (u[b+8:b+12]  @ ETA4) @ v[b+8:b+12]  # Z
    return out

# -----------------------------------------
# Closed-form 4D updates (no 4x4 matrices)
# -----------------------------------------
def apply_block_x(v4: np.ndarray, ch: float, sh: float, c: float, s: float) -> np.ndarray:
    t, x, y, z = v4
    # boost along x
    t1 = ch*t - sh*x
    x1 = -sh*t + ch*x
    # rotate y-z
    y2 = c*y - s*z
    z2 = s*y + c*z
    return np.array([t1, x1, y2, z2], dtype=np.float64)

def apply_block_y(v4: np.ndarray, ch: float, sh: float, c: float, s: float) -> np.ndarray:
    t, x, y, z = v4
    # boost along y
    t1 = ch*t - sh*y
    y1 = -sh*t + ch*y
    # rotate x-z
    x2 = c*x - s*z
    z2 = s*x + c*z
    return np.array([t1, x2, y1, z2], dtype=np.float64)

def apply_block_z(v4: np.ndarray, ch: float, sh: float, c: float, s: float) -> np.ndarray:
    t, x, y, z = v4
    # boost along z
    t1 = ch*t - sh*z
    z1 = -sh*t + ch*z
    # rotate x-y
    x2 = c*x - s*y
    y2 = s*x + c*y
    return np.array([t1, x2, y2, z1], dtype=np.float64)

# -----------------------------------------
# Fast-scalar table builder (per position)
# -----------------------------------------
DIM   = 768
SLICE = 12
assert DIM % SLICE == 0
NUM_FREQ = DIM // SLICE  # 64

class TriadMonSTERFast:
    """
    Scalar tables per position s = (t,x,y,z):
      φ_j  = (t * unit) * lam_j
      θx_j = (x * unit) * lam_j
      θy_j = (y * unit) * lam_j
      θz_j = (z * unit) * lam_j
    with lam_j = base^(-j / NUM_FREQ), unit = 1 / top_delta.
    Stores arrays of length NUM_FREQ for ch, sh, cx/sx, cy/sy, cz/sz.
    """
    def __init__(self, dim: int = DIM, base: float = 10000.0, top_delta: int = 1024):
        if dim % SLICE != 0:
            raise ValueError(f"dim must be divisible by {SLICE}, got {dim}.")
        self.dim      = dim
        self.num_freq = dim // SLICE
        self.base     = float(base)
        self.unit     = 1.0 / float(top_delta)

        j = np.arange(self.num_freq, dtype=np.float64)
        self.inv_freq = self.base ** (- j / self.num_freq)  # shape (F,)

        self._cache: Dict[Tuple[float, float, float, float, float, float], Dict[str, np.ndarray]] = {}

    def forward(self, s: np.ndarray) -> Dict[str, np.ndarray]:
        s = np.asarray(s, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError("position must be a 4D vector (t, x, y, z).")
        key = (s[0], s[1], s[2], s[3], self.unit, self.base)
        if key in self._cache:
            return self._cache[key]

        t, x, y, z = s
        u   = self.unit
        lam = self.inv_freq

        phi = (t * u) * lam       # (F,)
        thx = (x * u) * lam
        thy = (y * u) * lam
        thz = (z * u) * lam

        tables = {
            "lam": lam,           # keep for reporting
            "phi": phi, "thx": thx, "thy": thy, "thz": thz,  # angles (for reporting)
            "ch": np.cosh(phi), "sh": np.sinh(phi),
            "cx": np.cos(thx), "sx": np.sin(thx),
            "cy": np.cos(thy), "sy": np.sin(thy),
            "cz": np.cos(thz), "sz": np.sin(thz),
        }
        self._cache[key] = tables
        return tables

# -----------------------------------------
# Apply with scalar tables across the full embedding
# -----------------------------------------
def apply_monster_triad_fast(emb: np.ndarray, tables: Dict[str, np.ndarray], dim: int = DIM) -> np.ndarray:
    if emb.shape != (dim,):
        raise ValueError(f"embedding must be shape ({dim},), got {emb.shape}")

    ch = tables["ch"]; sh = tables["sh"]
    cx = tables["cx"]; sx = tables["sx"]
    cy = tables["cy"]; sy = tables["sy"]
    cz = tables["cz"]; sz = tables["sz"]

    out = np.empty_like(emb)
    for j in range(NUM_FREQ):
        b = j * SLICE
        out[b+0:b+4]   = apply_block_x(emb[b+0:b+4],   ch[j], sh[j], cx[j], sx[j])
        out[b+4:b+8]   = apply_block_y(emb[b+4:b+8],   ch[j], sh[j], cy[j], sy[j])
        out[b+8:b+12]  = apply_block_z(emb[b+8:b+12],  ch[j], sh[j], cz[j], sz[j])
    return out

# -----------------------------------------
# Reporting utilities
# -----------------------------------------
def report_header(title: str):
    print("\n" + "="*78)
    print(title)
    print("="*78)

def report_config(dim, num_freq, base, top_delta, unit):
    report_header("CONFIG")
    print(f"DIM                 : {dim}")
    print(f"NUM_FREQ (DIM/12)   : {num_freq}")
    print(f"base (inv_freq)     : {base}")
    print(f"top_delta (global)  : {top_delta}")
    print(f"unit = 1/top_delta  : {unit:.12f}")
    print("Frequency ladder lam_j = base^(-j/NUM_FREQ), j=0..NUM_FREQ-1")

def report_positions(s_q, s_k, dskq):
    report_header("POSITIONS")
    print(f"s_q  (t,x,y,z)      : {s_q}")
    print(f"s_k  (t,x,y,z)      : {s_k}")
    print(f"Δs = s_k - s_q      : {dskq}")

def report_ladder_and_angles(label, tables, n_head=5, n_tail=3):
    report_header(f"LADDER & ANGLES — {label}")
    lam = tables["lam"]
    phi = tables["phi"]; thx = tables["thx"]; thy = tables["thy"]; thz = tables["thz"]

    lam_ht  = head_tail(lam,  n_head, n_tail)
    phi_ht  = head_tail(phi,  n_head, n_tail)
    thx_ht  = head_tail(thx,  n_head, n_tail)
    thy_ht  = head_tail(thy,  n_head, n_tail)
    thz_ht  = head_tail(thz,  n_head, n_tail)

    def row(name, arr):
        print(f"{name:<6} :", " ".join("…" if np.isnan(v) else f"{v:+.6e}" for v in arr))

    row("lam", lam_ht)
    row("phi", phi_ht)
    row("thx", thx_ht)
    row("thy", thy_ht)
    row("thz", thz_ht)

def report_application_shapes(q, k, q_abs, k_abs):
    report_header("APPLICATION SUMMARY")
    print(f"q shape → {q.shape}, dtype={q.dtype}")
    print(f"k shape → {k.shape}, dtype={k.dtype}")
    print(f"L(s_q) q shape → {q_abs.shape}")
    print(f"L(s_k) k shape → {k_abs.shape}")

def report_contributions(label, u, v, dim, n_head=5, n_tail=3):
    report_header(f"PER-FREQUENCY CONTRIBUTIONS — {label}")
    by_freq = minkowski_dot_by_freq(u, v, dim)
    by_axis = minkowski_dot_by_freq_axis(u, v, dim)  # (F,3)
    total = by_freq.sum()
    # Show head/tail with a gap sentinel
    bf_ht = head_tail(by_freq, n_head, n_tail)
    print("Total (sum over all freqs) :", fmtf(total, 12))
    print("By freq (X+Y+Z)            :", " ".join("…" if np.isnan(x) else fmtf(x, 6) for x in bf_ht))
    # Also show the first and last few rows broken into X/Y/Z parts
    F = by_freq.size
    idxs = list(range(min(n_head, F))) + (["…"] if F > n_head + n_tail else []) + list(range(max(0, F - n_tail), F))
    print("\nBreakdown by axis per shown freq (X, Y, Z):")
    for i in idxs:
        if i == "…":
            print("   …")
        else:
            x, y, z = by_axis[i]
            print(f"  f={i:02d}: ({fmtf(x,6)}, {fmtf(y,6)}, {fmtf(z,6)})  sum={fmtf(x+y+z,6)}")
    return total, by_freq

def report_identity_and_norms(q, k, q_abs, k_abs, k_rel, dim):
    report_header("IDENTITY & NORM CHECKS")
    lhs = minkowski_dot_big(q_abs, k_abs, dim)
    rhs = minkowski_dot_big(q,     k_rel, dim)
    print(f"lhs = <L(s_q)q, L(s_k)k>   : {fmtf(lhs, 12)}")
    print(f"rhs = <q, L(Δs)k>         : {fmtf(rhs, 12)}")
    print(f"equal (allclose)?         : {np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)}")
    print(f"abs diff                  : {abs(lhs - rhs):.3e}")

    norms_before = np.array([minkowski_dot4(q[i:i+4],     q[i:i+4])     for i in range(0, dim, 4)])
    norms_after  = np.array([minkowski_dot4(q_abs[i:i+4], q_abs[i:i+4]) for i in range(0, dim, 4)])
    ok = np.allclose(norms_before, norms_after, rtol=1e-11, atol=1e-12)
    print(f"\nPer-4D Minkowski norms preserved? {ok}")
    print(f"max |Δ block-norm|         : {np.max(np.abs(norms_before - norms_after)):.3e}")

# -----------------------------------------
# Main demo
# -----------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # --- CONFIG ---
    BASE      = 10000.0   # classic RoPE base
    TOP_DELTA = 1024      # single global unit: unit = 1/TOP_DELTA
    UNIT      = 1.0 / TOP_DELTA

    monster = TriadMonSTERFast(dim=DIM, base=BASE, top_delta=TOP_DELTA)
    report_config(DIM, NUM_FREQ, BASE, TOP_DELTA, UNIT)

    # --- POSITIONS ---
    s_q  = np.array([ 700.0,  500.0, -300.0,  200.0], dtype=np.float64)  # (t,x,y,z)
    s_k  = np.array([ -40.0,  -20.0,   60.0,  -10.0], dtype=np.float64)
    dskq = s_k - s_q
    report_positions(s_q, s_k, dskq)

    # Build scalar tables
    T_q   = monster.forward(s_q)
    T_k   = monster.forward(s_k)
    T_rel = monster.forward(dskq)

    # --- SHOW LADDER/ANGLE SAMPLES ---
    report_ladder_and_angles("ABS(s_q)", T_q,  n_head=5, n_tail=3)
    report_ladder_and_angles("ABS(s_k)", T_k,  n_head=5, n_tail=3)
    report_ladder_and_angles("REL(Δs) ", T_rel, n_head=5, n_tail=3)

    # --- RANDOM EMBEDDINGS ---
    q = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)
    k = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)

    # Apply transforms
    q_abs = apply_monster_triad_fast(q, T_q, dim=DIM)
    k_abs = apply_monster_triad_fast(k, T_k, dim=DIM)
    k_rel = apply_monster_triad_fast(k, T_rel, dim=DIM)

    report_application_shapes(q, k, q_abs, k_abs)

    # --- PER-FREQ CONTRIBUTIONS ---
    lhs_total, lhs_by_freq = report_contributions("LHS  <L(s_q)q, L(s_k)k>", q_abs, k_abs, DIM)
    rhs_total, rhs_by_freq = report_contributions("RHS  <q, L(Δs)k>",        q,     k_rel, DIM)

    # Quick cross-check that totals match the earlier sums
    print("\nTotals cross-check:")
    print("  sum(lhs_by_freq) =", fmtf(lhs_by_freq.sum(), 12), "  vs lhs_total =", fmtf(lhs_total, 12))
    print("  sum(rhs_by_freq) =", fmtf(rhs_by_freq.sum(), 12), "  vs rhs_total =", fmtf(rhs_total, 12))

    # --- IDENTITY & NORM CHECKS ---
    report_identity_and_norms(q, k, q_abs, k_abs, k_rel, DIM)

    print("\nDone.")
