"""
MonSTER Triad (x, y, z) — Fully documented reference implementation.

This file implements a "Full Isotropic Absolute–Relative Fusion 4D MonSTER"
positional map with the triad layout:
  - Each frequency bucket contributes 12 real dims = 3 independent 4-D blocks:
      [ X 4D ] ⊕ [ Y 4D ] ⊕ [ Z 4D ]

Each 4-D block is a Lorentz isometry made by composing:
  - a boost along the corresponding axis (time <-> that axis),
  - a spatial rotation about that same axis.

Because the *generators are on the same axis inside a block*, they commute.
Because different axes act on *disjoint coordinates* (block-diagonal), the
whole map also commutes blockwise. This yields an *exact* RoPE-style identity:

  ⟨ L(s_q) q , L(s_k) k ⟩_ηbig  ==  ⟨ q , L(s_k − s_q) k ⟩_ηbig,

i.e., the attention "sees" only relative position (s_k - s_q).

We keep things simple:
  - DIM = 768
  - One frequency per 12-D block -> NUM_FREQ = 64
  - A *single* global unit u = 1 / top_delta to convert "steps" into a
    dimensionless angle/rapidity. No per-axis scaling, no "kappa" gains.
  - SAME geometric inverse-frequency ladder is used for both boosts (phi)
    and rotations (theta): lam_j = base^{-j / F}.

Conventions:
  - Vectors are treated as ROW vectors (shape (4,)).
  - The Minkowski metric uses signature (+ − − −).
  - Matrix multiplication uses Python's '@' operator.
  - Transpose is '.T'.
"""

import numpy as np

# ======================================================================
# 1) Minkowski metric (+ − − −) and two basic helpers
# ======================================================================

# ETA4 is the 4×4 diagonal "metric tensor" with signature (+ − − −).
# dtype=np.float64 ensures stable floating-point arithmetic.
ETA4 = np.diag([1.0, -1.0, -1.0, -1.0]).astype(np.float64)

def minkowski_dot4(u4: np.ndarray, v4: np.ndarray) -> float:
    """
    Compute the Minkowski inner product ⟨u, v⟩_η for two *row* 4-vectors.

    Mathematically:
      ⟨u, v⟩_η = u · (η v^T)  (if u is a column)
    With ROW vectors, we implement as: (u @ ETA4) @ v.

    Args:
      u4: shape (4,), row vector [t, x, y, z]
      v4: shape (4,), row vector [t, x, y, z]

    Returns:
      Scalar float: u_t v_t − u_x v_x − u_y v_y − u_z v_z

    Python/Numpy syntax notes:
      - '@' is matrix multiplication in Python 3 (PEP 465).
      - 'A @ B' does standard linear algebra multiply.
    """
    return (u4 @ ETA4) @ v4

def apply_lorentz_row(v4: np.ndarray, L4: np.ndarray) -> np.ndarray:
    """
    Apply a Lorentz transformation L to a *row* vector v.

    Column-vector form is: v' = L @ v
    Row-vector form (what we use): v' = v @ L^T

    Args:
      v4: shape (4,), row vector to transform
      L4: shape (4, 4), Lorentz matrix

    Returns:
      v4_transformed: shape (4,), row vector = v4 @ L4.T
    """
    return v4 @ L4.T


# ======================================================================
# 2) Primitive 4×4 Lorentz transforms for each cardinal axis
#    (boosts along axis; rotations about axis)
# ======================================================================

# ---- Boost along +X axis (mixes t <-> x) --------------------------------
def boost_x(phi: float) -> np.ndarray:
    """
    Return 4×4 Lorentz boost along +X with rapidity 'phi'.

    The nontrivial 2×2 block is:
      [ cosh φ  -sinh φ ]
      [ -sinh φ  cosh φ ]

    Everything else is identity on (y,z).
    """
    ch, sh = np.cosh(phi), np.sinh(phi)
    L = np.eye(4, dtype=np.float64)
    L[0,0], L[0,1] = ch, -sh  # time row: t' = ch*t - sh*x
    L[1,0], L[1,1] = -sh, ch  # x row:    x' = -sh*t + ch*x
    return L

# ---- Boost along +Y axis (mixes t <-> y) --------------------------------
def boost_y(phi: float) -> np.ndarray:
    ch, sh = np.cosh(phi), np.sinh(phi)
    L = np.eye(4, dtype=np.float64)
    L[0,0], L[0,2] = ch, -sh
    L[2,0], L[2,2] = -sh, ch
    return L

# ---- Boost along +Z axis (mixes t <-> z) --------------------------------
def boost_z(phi: float) -> np.ndarray:
    ch, sh = np.cosh(phi), np.sinh(phi)
    L = np.eye(4, dtype=np.float64)
    L[0,0], L[0,3] = ch, -sh
    L[3,0], L[3,3] = -sh, ch
    return L

# ---- Rotation about X axis (rotates y–z plane) --------------------------
def rot_x(theta: float) -> np.ndarray:
    """
    3D spatial rotation about +X by angle 'theta', embedded into 4D.

    The spatial sub-block rotates (y,z):
      [  c  -s ]
      [  s   c ]
    """
    c, s = np.cos(theta), np.sin(theta)
    L = np.eye(4, dtype=np.float64)
    L[2,2], L[2,3] = c, -s   # y'
    L[3,2], L[3,3] = s,  c   # z'
    return L

# ---- Rotation about Y axis (rotates x–z plane) --------------------------
def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    L = np.eye(4, dtype=np.float64)
    L[1,1], L[1,3] =  c, -s  # x'
    L[3,1], L[3,3] =  s,  c  # z'
    return L

# ---- Rotation about Z axis (rotates x–y plane) --------------------------
def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    L = np.eye(4, dtype=np.float64)
    L[1,1], L[1,2] =  c, -s  # x'
    L[2,1], L[2,2] =  s,  c  # y'
    return L


# ======================================================================
# 3) Per-axis 4×4 builders: compose rotation ∘ boost on the SAME axis
#    (order irrelevant because they commute on the same axis)
# ======================================================================

def Lx_from_pos(t: float, x: float, lam: float, unit: float) -> np.ndarray:
    """
    Build the X-axis 4×4 Lorentz map for absolute position (t, x), at frequency 'lam'.

    phi   = (t * unit) * lam   (rapidity for the boost along X)
    theta = (x * unit) * lam   (angle for the rotation about X)

    Returns:
      Lx = R_x(theta) @ B_x(phi)   (rot then boost; same-axis -> commute)
    """
    phi   = (t * unit) * lam
    theta = (x * unit) * lam
    return rot_x(theta) @ boost_x(phi)

def Ly_from_pos(t: float, y: float, lam: float, unit: float) -> np.ndarray:
    phi   = (t * unit) * lam
    theta = (y * unit) * lam
    return rot_y(theta) @ boost_y(phi)

def Lz_from_pos(t: float, z: float, lam: float, unit: float) -> np.ndarray:
    phi   = (t * unit) * lam
    theta = (z * unit) * lam
    return rot_z(theta) @ boost_z(phi)


# ======================================================================
# 4) Layout and module
# ======================================================================

# DIM is the total embedding dimension. We choose 768 so it’s divisible by 12.
DIM   = 768

# SLICE is the per-frequency footprint. Triad = X4 ⊕ Y4 ⊕ Z4 = 12 dims.
SLICE = 12

# Sanity: DIM must be a multiple of SLICE.
assert DIM % SLICE == 0, "DIM must be divisible by 12 for the triad layout."

# Number of frequency buckets = how many 12-D blocks fit into DIM.
NUM_FREQ = DIM // SLICE   # 64 for 768

class TriadMonSTER:
    """
    Precompute per-position, per-frequency Lorentz transforms for the triad layout.

    - SAME geometric inverse-frequency ladder is used for BOTH boosts (phi) and
      rotations (theta): lam_j = base^{-j / NUM_FREQ}, j = 0..NUM_FREQ-1.
    - A single global 'unit' converts raw "step" positions into dimensionless
      angles/rapidities: unit = 1 / top_delta.
    - Caches results by (position, unit, base) to avoid recomputation.

    Public API:
      forward(s: 4D) -> list[(Lx, Ly, Lz) for each frequency]
    """
    def __init__(self, dim: int = DIM, base: float = 10000.0, top_delta: int = 1024):
        """
        Args:
          dim: total embedding dimension (must be divisible by 12).
          base: geometric base for the inverse-frequency ladder (like standard RoPE).
          top_delta: ONE global "max meaningful delta" across t/x/y/z. The global unit
                     is unit = 1/top_delta (makes angles/rapidities dimensionless).
        """
        if dim % SLICE != 0:
            raise ValueError(f"dim must be divisible by {SLICE}. Got {dim}.")
        self.dim = dim
        self.num_freq = dim // SLICE
        self.base = float(base)

        # Global unit to convert "steps" into "radians-ish":
        # Example: with top_delta=1024, one full domain step maps to ~1/1024 of the
        # highest-frequency angle/rapidity. This keeps boosts from exploding.
        self.unit = 1.0 / float(top_delta)

        # Geometric inverse-frequency ladder: lam_j = base^{-j / F}
        # j increases -> frequency drops (coarser scale).
        j = np.arange(self.num_freq, dtype=np.float64)
        self.inv_freq = self.base ** (- j / self.num_freq)

        # Simple cache: dict with key = (*s, unit, base) to list[(Lx,Ly,Lz), ...]
        self._cache = {}

    def forward(self, s: np.ndarray) -> list:
        """
        Build the triad transforms for a single absolute 4D position s = (t, x, y, z).

        Args:
          s: np.ndarray shape (4,), absolute position (t, x, y, z) in "steps".

        Returns:
          tables: list of length NUM_FREQ; each element is a tuple (Lx, Ly, Lz),
                  where each L* is a 4×4 np.ndarray (float64).
        """
        s = np.asarray(s, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError("position must be a 4D vector (t, x, y, z).")

        # Cache key: position + unit + base. If any of those change, we recompute.
        key = (s[0], s[1], s[2], s[3], self.unit, self.base)
        if key in self._cache:
            return self._cache[key]

        t, x, y, z = s
        out = []
        for lam in self.inv_freq:
            # One frequency bucket -> three 4×4 blocks (X, Y, Z)
            Lx = Lx_from_pos(t, x, lam, self.unit)
            Ly = Ly_from_pos(t, y, lam, self.unit)
            Lz = Lz_from_pos(t, z, lam, self.unit)
            out.append((Lx, Ly, Lz))

        self._cache[key] = out
        return out


# ======================================================================
# 5) Apply the block-diagonal triad transforms across the full embedding
# ======================================================================

def apply_monster_triad(emb: np.ndarray, L_tables: list, dim: int = DIM) -> np.ndarray:
    """
    Apply the precomputed per-frequency triad transforms to a full embedding.

    Layout reminder per frequency:
      [ X 4D | Y 4D | Z 4D ]  -> total 12 dims per frequency.

    Args:
      emb: shape (dim,), the row-vector embedding to transform.
      L_tables: list of (Lx, Ly, Lz) per frequency, from TriadMonSTER.forward(...).
      dim: total embedding dimension (must match emb length).

    Returns:
      transformed: shape (dim,), the transformed embedding.
    """
    if emb.shape != (dim,):
        raise ValueError(f"embedding must be shape ({dim},), got {emb.shape}")
    if len(L_tables) * SLICE != dim:
        raise ValueError("L_tables length doesn't match embedding dim.")

    out = np.empty_like(emb)
    for j, (Lx, Ly, Lz) in enumerate(L_tables):
        b = j * SLICE  # base index into this frequency's 12D slice

        # X 4-D block
        out[b + 0 : b + 4] = apply_lorentz_row(emb[b + 0 : b + 4], Lx)

        # Y 4-D block
        out[b + 4 : b + 8] = apply_lorentz_row(emb[b + 4 : b + 8], Ly)

        # Z 4-D block
        out[b + 8 : b + 12] = apply_lorentz_row(emb[b + 8 : b + 12], Lz)

    return out


# ======================================================================
# 6) Big Minkowski inner product: sum over all 4D blocks
# ======================================================================

def minkowski_dot_big(u: np.ndarray, v: np.ndarray, dim: int = DIM) -> float:
    """
    Compute the "big" Minkowski inner product by summing over each 4-D block.

    Because our metric is block-diagonal (diag(η, η, η, ...)), the global score
    is simply the sum of per-4D Minkowski products.

    Args:
      u, v: shape (dim,), row vectors.
      dim:  total embedding dimension.

    Returns:
      Scalar float, the big Minkowski dot.
    """
    total = 0.0
    for i in range(0, dim, 4):
        total += minkowski_dot4(u[i:i+4], v[i:i+4])
    return total


# ======================================================================
# 7) Demo / sanity checks (you can run this file directly)
# ======================================================================

if __name__ == "__main__":
    np.random.seed(0)

    # Instantiate the triad module.
    # - base=10000.0 -> standard RoPE geometric ladder
    # - top_delta=1024 -> one global unit u = 1/1024, used by BOTH phi and theta
    triad = TriadMonSTER(dim=DIM, base=10000.0, top_delta=1024)

    # Absolute positions for query and key (4D: t, x, y, z) in *steps*.
    # You can pick any values within your expected max delta.
    s_q  = np.array([ 700.0,  500.0, -300.0,  200.0], dtype=np.float64)
    s_k  = np.array([ -40.0,  -20.0,   60.0,  -10.0], dtype=np.float64)

    # Relative displacement (this is what should *actually* matter in attention):
    dskq = s_k - s_q

    # Precompute per-frequency, per-axis matrices for each absolute position
    # and for the relative displacement.
    L_abs_q = triad.forward(s_q)   # list of (Lx, Ly, Lz) for s_q
    L_abs_k = triad.forward(s_k)   # list of (Lx, Ly, Lz) for s_k
    L_rel   = triad.forward(dskq)  # list of (Lx, Ly, Lz) for s_k - s_q

    # Random embeddings for q and k (row vectors of length DIM).
    q = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)
    k = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)

    # Apply absolute-position transforms to q and k.
    q_abs = apply_monster_triad(q, L_abs_q)
    k_abs = apply_monster_triad(k, L_abs_k)

    # -------------------------------
    # WHAT ARE lhs AND rhs?
    # -------------------------------
    # They are the two sides of the RoPE-style identity being tested.
    #
    #   lhs = ⟨ L(s_q) q , L(s_k) k ⟩_ηbig
    #   rhs = ⟨ q , L(s_k − s_q) k ⟩_ηbig
    #
    # - lhs: compute the big Minkowski dot *after* applying absolute transforms
    #        to both q and k.
    # - rhs: keep q untouched, but apply the transform built from the *relative*
    #        displacement (s_k − s_q) to k, then compute the big Minkowski dot.
    #
    # If your construction is correct, lhs and rhs should be *equal* (up to
    # floating point tolerance). That means the attention score depends only on
    # relative position, not on absolute positions separately.
    lhs = minkowski_dot_big(q_abs, k_abs)
    k_rel = apply_monster_triad(k, L_rel)
    rhs = minkowski_dot_big(q, k_rel)

    print("RoPE-style identity holds? ", np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12))
    print(f"lhs: {lhs:+.12f}  rhs: {rhs:+.12f}")

    # Per-4D Minkowski norm preservation:
    # Each 4D block is a Lorentz isometry, so ⟨v, v⟩_η should be preserved by L.
    norms_before = np.array([minkowski_dot4(q[i:i+4], q[i:i+4]) for i in range(0, DIM, 4)])
    norms_after  = np.array([minkowski_dot4(q_abs[i:i+4], q_abs[i:i+4]) for i in range(0, DIM, 4)])

    # The check below allows tiny numerical differences due to floating-point ops.
    ok = np.allclose(norms_before, norms_after, rtol=1e-11, atol=1e-12)
    max_err = np.max(np.abs(norms_before - norms_after))

    print("Per-4D Minkowski norms preserved? ", ok, "| max abs err:", max_err)
    print("NUM_FREQ:", NUM_FREQ, " | DIM:", DIM, " | SLICE per freq:", SLICE)

    # Notes:
    # - If you crank |t| and top frequency too high, phi can get large, making
    #   cosh/sinh enormous. That doesn't break the math, but FP errors grow.
    #   The single global unit (1/top_delta) keeps things sane without per-axis hacks.
    # - For speed in a real model: don't build 4×4 matrices each time; cache
    #   cos/sin/cosh/sinh per (pos,freq) and apply tiny 2×2 updates in-place.
