
# Vectorized NumPy version of the "Fast-scalar MonSTER Triad" with
# isotropic Fibonacci-sphere axes (no Python loops over frequencies)
from __future__ import annotations
import numpy as np

# =============================================================================
# 1) Minkowski helpers
# =============================================================================
ETA4 = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float64)

def minkowski_dot_big_vec(u: np.ndarray, v: np.ndarray) -> float:
    """
    Vectorized big Minkowski inner product for (dim,) row vectors,
    where the metric is block-diagonal with copies of ETA4 on each 4-D chunk.
    """
    U = u.reshape(-1, 4)
    V = v.reshape(-1, 4)
    return np.sum((U @ ETA4) * V)


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))

    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(1.0 - z * z)
    theta = i * phi

    out = np.empty((n, 3), dtype=np.float64)
    out[:, 0] = np.cos(theta) * r
    out[:, 1] = np.sin(theta) * r
    out[:, 2] = z
    return out


# =============================================================================
# 2) Vectorized scalar-table builder
# =============================================================================
SLICE = 12  # 12 dims per frequency: [A4 | B4 | C4], but each 4-D block gets its own Fibonacci axis.

class TriadMonSTERFastVec:
    """
    Vectorized cache of scalar tables for fast absolute/relative transforms.
    Frequencies: lam_j = base^{-j / F}, j=0..F-1, where F = dim // 12.

    Compared to the canonical triad version, we keep the same 12-D frequency
    bucket structure, but replace the repeated X/Y/Z axes with three distinct
    unit axes drawn from a Fibonacci sphere for each frequency bucket.

    For frequency j and local block b in {0,1,2}:

        phi_j       = (t * unit) * lam_j
        theta_{j,b} = ((axis_{j,b} · [x,y,z]) * unit) * lam_j

    Forward returns a dict with shapes:
        ch, sh:        (F,)          # cosh/sinh(phi)
        c_axes, s_axes:(F,3)         # cos/sin for the 3 axes in each bucket
        axis:          (F,3,3)       # the 3 unit spatial axes per bucket
    """
    def __init__(self, dim: int = 768, base: float = 10000.0, top_delta: int = 1024):
        if dim % SLICE != 0:
            raise ValueError(f"dim must be divisible by {SLICE}, got {dim}.")
        self.dim      = dim
        self.num_freq = dim // SLICE
        self.base     = float(base)
        self.unit     = 1.0 / float(top_delta)  # global unit (dimensionless per step)
        j = np.arange(self.num_freq, dtype=np.float64)
        self.inv_freq = self.base ** (- j / self.num_freq)

        # One new isotropic axis for each 4-D block in the embedding.
        axes = fibonacci_sphere(3 * self.num_freq).reshape(self.num_freq, 3, 3)
        # Re-normalize defensively in case of accumulated FP error.
        self.axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)

        self._cache = {}

    def forward(self, s: np.ndarray):
        s = np.asarray(s, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError("position must be a 4D vector (t, x, y, z).")
        key = (s[0], s[1], s[2], s[3], self.unit, self.base)
        if key in self._cache:
            return self._cache[key]

        t = s[0]
        spatial = s[1:]
        lam = self.inv_freq  # (F,)
        u = self.unit

        phi = (t * u) * lam

        # Project the spatial position onto each block's Fibonacci axis.
        proj = np.einsum("fab,b->fa", self.axes, spatial)  # (F,3)
        theta = (u * lam)[:, None] * proj                  # (F,3)

        ch = np.cosh(phi)      # (F,)
        sh = np.sinh(phi)      # (F,)
        c_axes = np.cos(theta) # (F,3)
        s_axes = np.sin(theta) # (F,3)

        out = {
            "ch": ch,
            "sh": sh,
            "c": c_axes,
            "s": s_axes,
            "axis": self.axes,
        }
        self._cache[key] = out
        return out


# =============================================================================
# 3) Vectorized apply (no loops over frequencies)
# =============================================================================
def apply_monster_triad_fast_vec(emb: np.ndarray, tables: dict, dim: int = 768) -> np.ndarray:
    """
    Apply the isotropic Fibonacci-axis triad transforms to a full embedding
    using only vectorized broadcasting.

    Args:
        emb   : (dim,) row vector.
        tables: dict with "ch","sh","c","s","axis" from TriadMonSTERFastVec.forward.
        dim   : total embedding dimension (multiple of 12).

    Returns:
        (dim,) transformed row vector.
    """
    if emb.shape != (dim,):
        raise ValueError(f"embedding must be shape ({dim},), got {emb.shape}")
    F = dim // SLICE

    # Reshape into (F, 3, 4): freq buckets × 3 local blocks × [t,x,y,z]
    V = emb.reshape(F, 3, 4).astype(np.float64, copy=False)
    out = V.copy()

    # Broadcasted scalars / axes
    ch = tables["ch"]             # (F,)
    sh = tables["sh"]             # (F,)
    c_axes = tables["c"]          # (F,3)
    s_axes = tables["s"]          # (F,3)
    axis = tables["axis"]         # (F,3,3)

    t = out[:, :, 0]              # (F,3)
    spatial = out[:, :, 1:]       # (F,3,3)

    # --------------------
    # Step 1: Boost along each block's own Fibonacci axis
    # --------------------
    proj = np.sum(spatial * axis, axis=-1)  # (F,3) = a · x

    t1 = ch[:, None] * t - sh[:, None] * proj
    spatial1 = spatial + (
        ((ch[:, None] - 1.0) * proj - sh[:, None] * t)[..., None] * axis
    )

    # --------------------
    # Step 2: Rotate the spatial part around the same axis
    # --------------------
    proj1 = np.sum(spatial1 * axis, axis=-1)  # (F,3) = a · x'
    cross = np.cross(axis, spatial1)          # (F,3,3) = a × x'

    spatial2 = (
        c_axes[..., None] * spatial1
        + s_axes[..., None] * cross
        + (1.0 - c_axes)[..., None] * proj1[..., None] * axis
    )

    out[:, :, 0] = t1
    out[:, :, 1:] = spatial2

    return out.reshape(dim,)


# =============================================================================
# 4) Demo / Sanity checks
# =============================================================================
if __name__ == "__main__":
    np.random.seed(0)
    DIM = 768
    monster = TriadMonSTERFastVec(dim=DIM, base=10000.0, top_delta=1024)

    # Absolute 4D positions (in "steps")
    s_q  = np.array([ 700.0,  500.0, -300.0,  200.0], dtype=np.float64)  # (t,x,y,z)
    s_k  = np.array([ -40.0,  -20.0,   60.0,  -10.0], dtype=np.float64)
    dskq = s_k - s_q

    # Tables
    T_abs_q = monster.forward(s_q)
    T_abs_k = monster.forward(s_k)
    T_rel   = monster.forward(dskq)

    # Random embeddings
    q = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)
    k = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)

    # Apply absolute maps
    q_abs = apply_monster_triad_fast_vec(q, T_abs_q, dim=DIM)
    k_abs = apply_monster_triad_fast_vec(k, T_abs_k, dim=DIM)

    # RoPE-style identity check
    lhs = minkowski_dot_big_vec(q_abs, k_abs)
    k_rel = apply_monster_triad_fast_vec(k, T_rel, dim=DIM)
    rhs = minkowski_dot_big_vec(q, k_rel)

    print("RoPE-style identity holds? ", np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12))
    print(f"lhs: {lhs:+.12f}  rhs: {rhs:+.12f}")

    # Per-4D Minkowski norm preservation
    Q_blocks = q.reshape(-1, 4)
    Q_abs_blocks = q_abs.reshape(-1, 4)
    norms_before = np.sum((Q_blocks @ ETA4) * Q_blocks, axis=1)
    norms_after  = np.sum((Q_abs_blocks @ ETA4) * Q_abs_blocks, axis=1)
    ok = np.allclose(norms_before, norms_after, rtol=1e-11, atol=1e-12)
    max_err = np.max(np.abs(norms_before - norms_after))
    print("Per-4D Minkowski norms preserved? ", ok, "| max abs err:", max_err)

    print("NUM_FREQ:", DIM // SLICE, " | DIM:", DIM, " | SLICE per freq:", SLICE)
    print("NUM_AXES:", 3 * (DIM // SLICE))
