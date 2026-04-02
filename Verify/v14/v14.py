# Vectorized NumPy version of the "Fast-scalar MonSTER Triad" (no Python loops over frequencies)
from __future__ import annotations
import numpy as np

# =============================================================================
# 1) Minkowski helpers
# =============================================================================
ETA4 = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float64)
L = 1.0 / 1024.0
QUERY_T_VALUES = (0.0, 256.0, 512.0, 1024.0, 2048.0, 3072.0, 4096.0)
QUERY_RADIAL_VALUES = QUERY_T_VALUES

def minkowski_dot_big_vec(u: np.ndarray, v: np.ndarray) -> float:
    """
    Vectorized big Minkowski inner product for (dim,) row vectors,
    where the metric is block-diagonal with copies of ETA4 on each 4-D chunk.
    """
    U = u.reshape(-1, 4)
    V = v.reshape(-1, 4)
    return np.sum((U @ ETA4) * V)


def minkowski_dot_rowwise_big_vec(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rowwise big Minkowski inner products for batches of row vectors.
    Returns a length-n array where entry i is the Minkowski dot between
    u[i] and v[i].
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("u and v must be rank-2 arrays of shape (num_vecs, dim).")
    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same shape, got {u.shape} and {v.shape}.")

    U = u.reshape(u.shape[0], -1, 4)
    V = v.reshape(v.shape[0], -1, 4)
    return np.einsum("nbi,ij,nbj->n", U, ETA4, V, optimize=True)


def minkowski_dot_pairwise_big_vec(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Pairwise big Minkowski inner products for batches of row vectors.
    Returns an (n_u, n_v) matrix where entry [i, j] is the Minkowski dot
    between u[i] and v[j].
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("u and v must be rank-2 arrays of shape (num_vecs, dim).")
    if u.shape[1] != v.shape[1]:
        raise ValueError(f"u and v must have the same dim, got {u.shape[1]} and {v.shape[1]}.")

    U = u.reshape(u.shape[0], -1, 4)
    V = v.reshape(v.shape[0], -1, 4)
    return np.einsum("nbi,ij,mbj->nm", U, ETA4, V, optimize=True)


# =============================================================================
# 2) Vectorized scalar-table builder
# =============================================================================
class TriadMonSTERFastVec:
    """
    Vectorized cache of scalar tables for fast absolute/relative transforms.
    Frequencies: lam_j = base^{-j / F}, j=0..F-1, where F = dim // 12.

    Angles/Rapidities:
        phi_j   = (L * t) * lam_j
        thx_j   = x * lam_j
        thy_j   = y * lam_j
        thz_j   = z * lam_j

    Forward returns a dict with shapes:
        ch, sh:     (F,)          # cosh/sinh(phi)
        c_axes, s_axes: (F,3)     # cos/sin for X,Y,Z axes respectively
    """
    def __init__(
        self,
        dim: int = 768,
        base: float = 10000.0,
    ):
        if dim % 12 != 0:
            raise ValueError(f"dim must be divisible by 12, got {dim}.")
        self.dim      = dim
        self.num_freq = dim // 12
        self.base     = float(base)
        self.L        = L
        j = np.arange(self.num_freq, dtype=np.float64)
        self.inv_freq = self.base ** (- j / self.num_freq)
        self._cache = {}

    def forward(self, s: np.ndarray):
        s = np.asarray(s, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError("position must be a 4D vector (t, x, y, z).")
        key = (s[0], s[1], s[2], s[3])
        if key in self._cache:
            return self._cache[key]

        t, x, y, z = s
        lam = self.inv_freq  # (F,)

        phi  = (self.L * t) * lam
        thx  = x * lam
        thy  = y * lam
        thz  = z * lam

        ch = np.cosh(phi)              # (F,)
        sh = np.sinh(phi)              # (F,)
        c_axes = np.stack((np.cos(thx), np.cos(thy), np.cos(thz)), axis=1)  # (F,3) -> X,Y,Z
        s_axes = np.stack((np.sin(thx), np.sin(thy), np.sin(thz)), axis=1)  # (F,3)

        out = {"ch": ch, "sh": sh, "c": c_axes, "s": s_axes}
        self._cache[key] = out
        return out


# =============================================================================
# 3) Vectorized apply (no loops over frequencies)
# =============================================================================
def apply_monster_triad_fast_vec(emb: np.ndarray, tables: dict, dim: int = 768) -> np.ndarray:
    """
    Apply triad transforms to a full embedding using only vectorized broadcasting.
    Args:
        emb   : (dim,) row vector.
        tables: dict with "ch","sh","c","s" from TriadMonSTERFastVec.forward.
        dim   : total embedding dimension (multiple of 12).
    Returns:
        (dim,) transformed row vector.
    """
    if emb.shape != (dim,):
        raise ValueError(f"embedding must be shape ({dim},), got {emb.shape}")
    F = dim // 12

    # Reshape into (F, 3, 4): freq buckets × {X,Y,Z} × [t,x,y,z]
    V = emb.reshape(F, 3, 4).astype(np.float64, copy=False)
    out = V.copy()

    # Broadcasted scalars
    ch = tables["ch"]          # (F,)
    sh = tables["sh"]          # (F,)
    c_axes = tables["c"]       # (F,3)
    s_axes = tables["s"]       # (F,3)

    # --------------------
    # Step 1: Boost along each axis' spatial component
    # --------------------
    # Indices for the spatial component aligned with axis: X->1, Y->2, Z->3
    comp_idx = np.array([1, 2, 3], dtype=np.int64)[None, :, None]  # (1,3,1)
    t = out[:, :, 0]                        # (F,3)
    x_axis = np.take_along_axis(out, comp_idx, axis=2)[..., 0]  # (F,3)

    t1 = ch[:, None] * t - sh[:, None] * x_axis
    x1 = -sh[:, None] * t + ch[:, None] * x_axis

    out[:, :, 0] = t1
    np.put_along_axis(out, comp_idx, x1[..., None], axis=2)

    # --------------------
    # Step 2: Rotate in the orthogonal spatial 2D planes
    # --------------------
    # For axis X: rotate (y,z) -> indices (2,3)
    # For axis Y: rotate (x,z) -> indices (1,3)
    # For axis Z: rotate (x,y) -> indices (1,2)
    pair_idx = np.array([[2, 3], [1, 3], [1, 2]], dtype=np.int64)[None, :, :]  # (1,3,2)

    pair_vals = np.take_along_axis(out, pair_idx, axis=2)  # (F,3,2)
    u = pair_vals[..., 0]  # first in the pair
    v = pair_vals[..., 1]  # second in the pair

    cu = c_axes  # (F,3)
    su = s_axes  # (F,3)

    u2 = cu * u - su * v
    v2 = su * u + cu * v

    rotated = np.stack((u2, v2), axis=-1)  # (F,3,2)
    np.put_along_axis(out, pair_idx, rotated, axis=2)

    return out.reshape(dim,)


def apply_monster_triad_fast_batch_vec(embs: np.ndarray, tables: dict, dim: int = 768) -> np.ndarray:
    """
    Batched version of apply_monster_triad_fast_vec for arrays of shape
    (num_vecs, dim).
    """
    embs = np.asarray(embs, dtype=np.float64)
    if embs.ndim != 2 or embs.shape[1] != dim:
        raise ValueError(f"embs must be shape (num_vecs, {dim}), got {embs.shape}")

    F = dim // 12
    out = embs.reshape(-1, F, 3, 4).copy()

    ch = tables["ch"][None, :, None]
    sh = tables["sh"][None, :, None]
    c_axes = tables["c"][None, :, :]
    s_axes = tables["s"][None, :, :]

    axis_idx = np.arange(3)
    spatial_idx = np.array([1, 2, 3], dtype=np.int64)
    t = out[..., 0]
    x_axis = out[:, :, axis_idx, spatial_idx]

    t1 = ch * t - sh * x_axis
    x1 = -sh * t + ch * x_axis

    out[..., 0] = t1
    out[:, :, axis_idx, spatial_idx] = x1

    first_pair_idx = np.array([2, 1, 1], dtype=np.int64)
    second_pair_idx = np.array([3, 3, 2], dtype=np.int64)
    u = out[:, :, axis_idx, first_pair_idx]
    v = out[:, :, axis_idx, second_pair_idx]

    u2 = c_axes * u - s_axes * v
    v2 = s_axes * u + c_axes * v

    out[:, :, axis_idx, first_pair_idx] = u2
    out[:, :, axis_idx, second_pair_idx] = v2

    return out.reshape(embs.shape[0], dim)


def absolute_minkowski_score(
    q: np.ndarray,
    k: np.ndarray,
    s_q: np.ndarray,
    s_k: np.ndarray,
    monster: "TriadMonSTERFastVec",
    dim: int = 768,
) -> float:
    q_abs = apply_monster_triad_fast_vec(q, monster.forward(s_q), dim=dim)
    k_abs = apply_monster_triad_fast_vec(k, monster.forward(s_k), dim=dim)
    return minkowski_dot_big_vec(q_abs, k_abs)


def relative_minkowski_score(
    q: np.ndarray,
    k: np.ndarray,
    s_q: np.ndarray,
    s_k: np.ndarray,
    monster: "TriadMonSTERFastVec",
    dim: int = 768,
) -> float:
    k_rel = apply_monster_triad_fast_vec(k, monster.forward(s_k - s_q), dim=dim)
    return minkowski_dot_big_vec(q, k_rel)


def sample_distinct_ordered_pairs(num_tokens: int, num_pairs: int, rng: np.random.Generator) -> np.ndarray:
    all_distinct_pairs = np.argwhere(~np.eye(num_tokens, dtype=bool))
    pair_rows = rng.choice(all_distinct_pairs.shape[0], size=num_pairs, replace=False)
    return all_distinct_pairs[pair_rows]


def sample_positions_on_radius_shell(
    radius: float,
    num_positions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if radius == 0.0:
        return np.zeros((num_positions, 3), dtype=np.float64)

    directions = rng.normal(size=(num_positions, 3)).astype(np.float64)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    directions /= norms
    return radius * directions


def pairwise_minkowski_by_query_time(
    token_vectors: np.ndarray,
    query_t_values: tuple[float, ...] = QUERY_T_VALUES,
    dim: int = 768,
    base: float = 10000.0,
) -> dict[float, np.ndarray]:
    """
    For each query time in query_t_values, compute the ordered pairwise
    Minkowski products between all distinct token pairs. The key time is fixed
    at 0, and all xyz coordinates remain 0 for both query and key.

    Returns a dict mapping q_t -> (num_tokens, num_tokens) score matrix with
    NaN on the diagonal to exclude self-pairs.
    """
    token_vectors = np.asarray(token_vectors, dtype=np.float64)
    if token_vectors.ndim != 2 or token_vectors.shape[1] != dim:
        raise ValueError(f"token_vectors must be shape (num_tokens, {dim}), got {token_vectors.shape}")

    monster = TriadMonSTERFastVec(dim=dim, base=base)
    key_pos = np.zeros(4, dtype=np.float64)
    key_tables = monster.forward(key_pos)
    key_abs = apply_monster_triad_fast_batch_vec(token_vectors, key_tables, dim=dim)

    results: dict[float, np.ndarray] = {}
    for q_t in query_t_values:
        query_pos = np.array([q_t, 0.0, 0.0, 0.0], dtype=np.float64)
        query_tables = monster.forward(query_pos)
        query_abs = apply_monster_triad_fast_batch_vec(token_vectors, query_tables, dim=dim)
        pairwise_scores = minkowski_dot_pairwise_big_vec(query_abs, key_abs)
        np.fill_diagonal(pairwise_scores, np.nan)
        results[float(q_t)] = pairwise_scores

    return results


def sampled_spatial_minkowski_by_radius(
    query_vectors: np.ndarray,
    key_vectors: np.ndarray,
    radial_values: tuple[float, ...] = QUERY_RADIAL_VALUES,
    dim: int = 768,
    base: float = 10000.0,
    rng: np.random.Generator | None = None,
) -> dict[float, np.ndarray]:
    """
    For each radius r in radial_values, sample one query position per token pair
    uniformly on the radius-r sphere, keep query/key time fixed at 0, keep the
    key at the spatial origin, and return the corresponding Minkowski products.
    """
    query_vectors = np.asarray(query_vectors, dtype=np.float64)
    key_vectors = np.asarray(key_vectors, dtype=np.float64)
    if query_vectors.ndim != 2 or query_vectors.shape[1] != dim:
        raise ValueError(f"query_vectors must be shape (num_pairs, {dim}), got {query_vectors.shape}")
    if key_vectors.shape != query_vectors.shape:
        raise ValueError(f"key_vectors must match query_vectors shape, got {key_vectors.shape}")

    if rng is None:
        rng = np.random.default_rng()

    monster = TriadMonSTERFastVec(dim=dim, base=base)
    key_pos = np.zeros(4, dtype=np.float64)
    key_abs = apply_monster_triad_fast_batch_vec(key_vectors, monster.forward(key_pos), dim=dim)

    num_pairs = query_vectors.shape[0]
    results: dict[float, np.ndarray] = {}
    for radius in radial_values:
        shell_positions = sample_positions_on_radius_shell(float(radius), num_pairs, rng)
        query_abs = np.empty_like(query_vectors)
        for i, xyz in enumerate(shell_positions):
            query_pos = np.array([0.0, xyz[0], xyz[1], xyz[2]], dtype=np.float64)
            query_abs[i] = apply_monster_triad_fast_vec(query_vectors[i], monster.forward(query_pos), dim=dim)
        results[float(radius)] = minkowski_dot_rowwise_big_vec(query_abs, key_abs)

    return results


# =============================================================================
# 4) Demo / Sanity checks
# =============================================================================
if __name__ == "__main__":
    DIM = 768
    NUM_TOKENS = 1000
    NUM_PAIRS = 1000
    rng = np.random.default_rng()
    token_vectors = rng.uniform(-0.6, 0.6, size=(NUM_TOKENS, DIM)).astype(np.float64)
    sampled_pairs = sample_distinct_ordered_pairs(NUM_TOKENS, NUM_PAIRS, rng)
    query_pair_idx = sampled_pairs[:, 0]
    key_pair_idx = sampled_pairs[:, 1]
    query_vectors = token_vectors[query_pair_idx]
    key_vectors = token_vectors[key_pair_idx]

    spatial_scores_by_radius = sampled_spatial_minkowski_by_radius(
        query_vectors,
        key_vectors,
        dim=DIM,
        rng=rng,
    )

    print(f"Generated {NUM_TOKENS} token/hidden vectors with dim={DIM}.")
    print(f"Averaging Minkowski products over {NUM_PAIRS} distinct ordered token pairs.")
    print("Query/key time is fixed at 0. Key xyz is fixed at the origin.")
    print("For each radius r, one query position per pair is sampled uniformly on the radius-r shell.")

    for radius, scores in spatial_scores_by_radius.items():
        print(f"r={radius:7.1f} | t_q=0.0 | t_k=0.0 | avg_product={scores.mean():+.12f}")
