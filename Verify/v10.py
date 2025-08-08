"""
Fast-scalar MonSTER Triad (x, y, z), 12 dims per frequency, DIM = 768.

This file implements the same math as the matrix version but avoids materializing
4×4 matrices. We cache only scalars per (position, frequency):

  - boost scalars:   cosh(φ_j), sinh(φ_j)           (shared across x/y/z at freq j)
  - rotation scalars: cos(θ_{x,j}), sin(θ_{x,j})
                      cos(θ_{y,j}), sin(θ_{y,j})
                      cos(θ_{z,j}), sin(θ_{z,j})

Then we transform each 4-D subvector directly with closed-form updates.

Layout per frequency bucket j (12 real dims):
  [ X 4D | Y 4D | Z 4D ]
Each 4D block uses canonical coordinate order [t, x, y, z] and applies:
  - a boost along that axis (t <-> axis), then
  - a rotation about that axis (rotates the other two spatial coords).

Because both generators in a block are about the SAME axis, they commute.
Blocks act on disjoint coordinates; the full operator is block-diagonal.
Therefore the RoPE-style identity holds exactly:

  ⟨ L(s_q) q , L(s_k) k ⟩_ηbig  ==  ⟨ q , L(s_k − s_q) k ⟩_ηbig

Conventions:
- Vectors are *row* vectors (shape (4,) or (DIM,)).
- Minkowski metric signature is (+ − − −).
- Python '@' is matrix multiply; we don't use it here, just scalars.

Author: You :)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

# ============================================================================
# 1) Minkowski metric and helpers (row-vector convention)
# ============================================================================

ETA4 = np.diag([1.0, -1.0, -1.0, -1.0]).astype(np.float64)

def minkowski_dot4(u4: np.ndarray, v4: np.ndarray) -> float:
    """
    Minkowski inner product for 4-D *row* vectors.

    Math: ⟨u, v⟩_η = u_t v_t − u_x v_x − u_y v_y − u_z v_z
    Code: (u @ ETA4) @ v

    Args:
        u4: (4,) float64 row vector [t, x, y, z]
        v4: (4,) float64 row vector [t, x, y, z]
    Returns:
        float scalar inner product under (+ − − −)
    """
    return (u4 @ ETA4) @ v4

def minkowski_dot_big(u: np.ndarray, v: np.ndarray, dim: int) -> float:
    """
    Big/Multi-block Minkowski inner product.

    Our overall metric is block-diagonal with copies of η on each 4-D block.
    So the global dot is just the sum over 4-D chunks.

    Args:
        u, v: (dim,)
        dim : total embedding dimension (must be multiple of 4)
    """
    total = 0.0
    for i in range(0, dim, 4):
        total += minkowski_dot4(u[i:i+4], v[i:i+4])
    return total


# ============================================================================
# 2) Closed-form 4-D transforms per axis using only scalars
#    (No matrices; just a few FMAs.)
# ============================================================================

def apply_block_x(v4: np.ndarray, ch: float, sh: float, c: float, s: float) -> np.ndarray:
    """
    Apply: boost_x(φ) then rot_x(θ) to a 4-D *row* vector v4 = [t, x, y, z].

    - boost_x mixes (t, x):
        t1 =  ch*t − sh*x
        x1 = −sh*t + ch*x
        y1 =  y
        z1 =  z
    - rot_x rotates (y, z):
        y2 =  c*y1 − s*z1
        z2 =  s*y1 + c*z1

    Args:
        v4: (4,)
        ch: cosh(φ)
        sh: sinh(φ)
        c : cos(θ_x)
        s : sin(θ_x)
    Returns:
        (4,) transformed row vector
    """
    t, x, y, z = v4
    # boost along x
    t1 = ch*t - sh*x
    x1 = -sh*t + ch*x
    # rotate y-z
    y2 = c*y - s*z
    z2 = s*y + c*z
    return np.array([t1, x1, y2, z2], dtype=np.float64)

def apply_block_y(v4: np.ndarray, ch: float, sh: float, c: float, s: float) -> np.ndarray:
    """
    Apply: boost_y(φ), then rot_y(θ) on v4 = [t, x, y, z].

    - boost_y mixes (t, y):
        t1 =  ch*t − sh*y
        y1 = −sh*t + ch*y
        x1 =  x
        z1 =  z
    - rot_y rotates (x, z):
        x2 =  c*x1 − s*z1
        z2 =  s*x1 + c*z1
    """
    t, x, y, z = v4
    # boost along y
    t1 = ch*t - sh*y
    y1 = -sh*t + ch*y
    # rotate x-z
    x2 = c*x - s*z
    z2 = s*x + c*z
    return np.array([t1, x2, y1, z2], dtype=np.float64)

def apply_block_z(v4: np.ndarray, ch: float, sh: float, c: float, s: float) -> np.ndarray:
    """
    Apply: boost_z(φ), then rot_z(θ) on v4 = [t, x, y, z].

    - boost_z mixes (t, z):
        t1 =  ch*t − sh*z
        z1 = −sh*t + ch*z
        x1 =  x
        y1 =  y
    - rot_z rotates (x, y):
        x2 =  c*x1 − s*y1
        y2 =  s*x1 + c*y1
    """
    t, x, y, z = v4
    # boost along z
    t1 = ch*t - sh*z
    z1 = -sh*t + ch*z
    # rotate x-y
    x2 = c*x - s*y
    y2 = s*x + c*y
    return np.array([t1, x2, y2, z1], dtype=np.float64)


# ============================================================================
# 3) Fast-scalar cache builder (per position)
#    We store only the scalar tables per frequency bucket.
# ============================================================================

DIM   = 768         # total embedding size
SLICE = 12          # 12 dims per freq: [X4 | Y4 | Z4]
assert DIM % SLICE == 0
NUM_FREQ = DIM // SLICE  # 64

class TriadMonSTERFast:
    """
    Build and cache scalar tables for fast application.

    Frequencies:
      lam_j = base^{-j / NUM_FREQ}, for j = 0..NUM_FREQ-1

    Angles/Rapidities (SAME ladder for both):
      φ_j      = (t * unit) * lam_j
      θ_{x,j}  = (x * unit) * lam_j
      θ_{y,j}  = (y * unit) * lam_j
      θ_{z,j}  = (z * unit) * lam_j

    Where:
      unit = 1 / top_delta (one global unit; no per-axis normalization)

    We return a dict of NumPy arrays (length NUM_FREQ):
      {
        "ch": cosh(φ_j),
        "sh": sinh(φ_j),
        "cx": cos(θ_{x,j}), "sx": sin(θ_{x,j}),
        "cy": cos(θ_{y,j}), "sy": sin(θ_{y,j}),
        "cz": cos(θ_{z,j}), "sz": sin(θ_{z,j}),
      }
    """

    def __init__(self, dim: int = DIM, base: float = 10000.0, top_delta: int = 1024):
        if dim % SLICE != 0:
            raise ValueError(f"dim must be divisible by {SLICE}, got {dim}.")
        self.dim      = dim
        self.num_freq = dim // SLICE
        self.base     = float(base)
        self.unit     = 1.0 / float(top_delta)  # global unit (dimensionless per step)
        # geometric inverse-frequency ladder
        j = np.arange(self.num_freq, dtype=np.float64)
        self.inv_freq = self.base ** (- j / self.num_freq)
        self._cache: Dict[Tuple[float, float, float, float, float, float], Dict[str, np.ndarray]] = {}

    def forward(self, s: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Precompute scalar tables for a given absolute 4D position s.

        Args:
            s: (4,) np.ndarray = [t, x, y, z] in "steps"
        Returns:
            dict of arrays, each shape (NUM_FREQ,)
        """
        s = np.asarray(s, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError("position must be a 4D vector (t, x, y, z).")

        key = (s[0], s[1], s[2], s[3], self.unit, self.base)
        if key in self._cache:
            return self._cache[key]

        t, x, y, z = s
        u = self.unit
        lam = self.inv_freq  # shape (NUM_FREQ,)

        # Angles/rapidities per frequency (vectorized)
        phi  = (t * u) * lam                  # shape (F,)
        thx  = (x * u) * lam
        thy  = (y * u) * lam
        thz  = (z * u) * lam

        # Scalars we actually need
        tables = {
            "ch": np.cosh(phi),
            "sh": np.sinh(phi),
            "cx": np.cos(thx), "sx": np.sin(thx),
            "cy": np.cos(thy), "sy": np.sin(thy),
            "cz": np.cos(thz), "sz": np.sin(thz),
        }
        self._cache[key] = tables
        return tables


# ============================================================================
# 4) Fast-scalar application across the full embedding
# ============================================================================

def apply_monster_triad_fast(
    emb: np.ndarray,
    tables: Dict[str, np.ndarray],
    dim: int = DIM
) -> np.ndarray:
    """
    Apply triad transforms to a full embedding using only scalar tables.

    Args:
        emb   : (dim,) row vector to transform.
        tables: dict returned by TriadMonSTERFast.forward(...).
                Each entry is shape (NUM_FREQ,).
        dim   : total embedding dimension.

    Returns:
        out   : (dim,) transformed row vector.
    """
    if emb.shape != (dim,):
        raise ValueError(f"embedding must be shape ({dim},), got {emb.shape}")

    ch = tables["ch"]; sh = tables["sh"]
    cx = tables["cx"]; sx = tables["sx"]
    cy = tables["cy"]; sy = tables["sy"]
    cz = tables["cz"]; sz = tables["sz"]

    out = np.empty_like(emb)

    # Process each frequency bucket j (12 dims per bucket)
    # Bucket layout: [X4 | Y4 | Z4], each 4D is [t, x, y, z]
    for j in range(NUM_FREQ):
        b = j * SLICE

        # X block (indices b .. b+3)
        vX = emb[b + 0 : b + 4]
        out[b + 0 : b + 4] = apply_block_x(vX, ch[j], sh[j], cx[j], sx[j])

        # Y block (indices b+4 .. b+7)
        vY = emb[b + 4 : b + 8]
        out[b + 4 : b + 8] = apply_block_y(vY, ch[j], sh[j], cy[j], sy[j])

        # Z block (indices b+8 .. b+11)
        vZ = emb[b + 8 : b + 12]
        out[b + 8 : b + 12] = apply_block_z(vZ, ch[j], sh[j], cz[j], sz[j])

    return out


# ============================================================================
# 5) Demo / sanity checks
# ============================================================================

if __name__ == "__main__":
    np.random.seed(0)

    # --- Instantiate the fast-scalar MonSTER ---
    monster = TriadMonSTERFast(dim=DIM, base=10000.0, top_delta=1024)

    # Absolute 4D positions for query and key (in "steps")
    s_q  = np.array([ 700.0,  500.0, -300.0,  200.0], dtype=np.float64)  # (t,x,y,z)
    s_k  = np.array([ -40.0,  -20.0,   60.0,  -10.0], dtype=np.float64)
    dskq = s_k - s_q  # relative displacement

    # Precompute scalar tables (per position, per frequency)
    T_abs_q = monster.forward(s_q)   # dict of arrays for s_q
    T_abs_k = monster.forward(s_k)   # dict of arrays for s_k
    T_rel   = monster.forward(dskq)  # dict of arrays for s_k - s_q

    # Random embeddings (row vectors)
    q = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)
    k = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)

    # Apply absolute-position transforms to q and k
    q_abs = apply_monster_triad_fast(q, T_abs_q, dim=DIM)
    k_abs = apply_monster_triad_fast(k, T_abs_k, dim=DIM)

    # -------------------------------
    # WHAT ARE lhs AND rhs?
    # -------------------------------
    # lhs = ⟨ L(s_q) q , L(s_k) k ⟩_ηbig
    # rhs = ⟨ q , L(s_k − s_q) k ⟩_ηbig
    #
    # - lhs: transform both q and k using their absolute positions; then compute
    #        the big Minkowski inner product.
    # - rhs: transform only k using the *relative* displacement; then compute the
    #        big Minkowski inner product against the original q.
    #
    # If the construction is correct, lhs ≈ rhs (floating-point tolerance).
    lhs = minkowski_dot_big(q_abs, k_abs, dim=DIM)
    k_rel = apply_monster_triad_fast(k, T_rel, dim=DIM)
    rhs = minkowski_dot_big(q, k_rel, dim=DIM)

    print("RoPE-style identity holds? ", np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12))
    print(f"lhs: {lhs:+.12f}  rhs: {rhs:+.12f}")

    # Per-4D Minkowski norm preservation check:
    # Each 4-D block is an exact Lorentz isometry. After applying transforms,
    # ⟨v, v⟩_η should be preserved blockwise (up to tiny FP noise).
    norms_before = np.array([minkowski_dot4(q[i:i+4], q[i:i+4]) for i in range(0, DIM, 4)])
    norms_after  = np.array([minkowski_dot4(q_abs[i:i+4], q_abs[i:i+4]) for i in range(0, DIM, 4)])
    ok = np.allclose(norms_before, norms_after, rtol=1e-11, atol=1e-12)
    max_err = np.max(np.abs(norms_before - norms_after))

    print("Per-4D Minkowski norms preserved? ", ok, "| max abs err:", max_err)
    print("NUM_FREQ:", NUM_FREQ, " | DIM:", DIM, " | SLICE per freq:", SLICE)
