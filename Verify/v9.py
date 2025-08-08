import numpy as np

# ============================================================
# MonSTER Triad (x,y,z) — 12 dims per frequency, DIM=768
# ============================================================

# 1) Minkowski metric (+ - - -) and helpers (row-vector convention)
ETA4 = np.diag([1.0, -1.0, -1.0, -1.0]).astype(np.float64)

def minkowski_dot4(u4, v4):
    """<u,v>_η for 4D row vectors."""
    return (u4 @ ETA4) @ v4

def apply_lorentz_row(v4, L4):
    """Row-vector action: v' = v · Lᵀ."""
    return v4 @ L4.T

# 2) Axis-specific 4×4 generators (boost along axis, rotation about axis)
def boost_x(phi):
    ch, sh = np.cosh(phi), np.sinh(phi)
    L = np.eye(4, dtype=np.float64)
    L[0,0], L[0,1] = ch, -sh
    L[1,0], L[1,1] = -sh, ch
    return L

def boost_y(phi):
    ch, sh = np.cosh(phi), np.sinh(phi)
    L = np.eye(4, dtype=np.float64)
    L[0,0], L[0,2] = ch, -sh
    L[2,0], L[2,2] = -sh, ch
    return L

def boost_z(phi):
    ch, sh = np.cosh(phi), np.sinh(phi)
    L = np.eye(4, dtype=np.float64)
    L[0,0], L[0,3] = ch, -sh
    L[3,0], L[3,3] = -sh, ch
    return L

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    L = np.eye(4, dtype=np.float64)
    # rotate (y,z)
    L[2,2], L[2,3] = c, -s
    L[3,2], L[3,3] = s,  c
    return L

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    L = np.eye(4, dtype=np.float64)
    # rotate (z,x)
    L[1,1], L[1,3] =  c, -s
    L[3,1], L[3,3] =  s,  c
    return L

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    L = np.eye(4, dtype=np.float64)
    # rotate (x,y)
    L[1,1], L[1,2] =  c, -s
    L[2,1], L[2,2] =  s,  c
    return L

def Lx_from_pos(t, x, omega):
    phi   = omega * t
    theta = omega * x
    # Same-axis generators commute -> order irrelevant
    return rot_x(theta) @ boost_x(phi)

def Ly_from_pos(t, y, omega):
    phi   = omega * t
    theta = omega * y
    return rot_y(theta) @ boost_y(phi)

def Lz_from_pos(t, z, omega):
    phi   = omega * t
    theta = omega * z
    return rot_z(theta) @ boost_z(phi)

# 3) The Triad MonSTER module: cache per-position, per-frequency transforms
DIM = 768                     # embedding dimension
SLICE = 12                    # 3×(4D)
assert DIM % SLICE == 0
NUM_FREQ = DIM // SLICE       # 64

class TriadMonSTER:
    """
    Precompute block-diagonal Lorentz rotors for x/y/z axes per frequency.
    Each frequency contributes 12 dims = [X 4D | Y 4D | Z 4D].
    """
    def __init__(self, dim=DIM, base=10000.0):
        if dim % SLICE != 0:
            raise ValueError(f"dim must be divisible by {SLICE}. Got {dim}.")
        self.dim = dim
        self.num_freq = dim // SLICE
        self.base = float(base)
        # One inverse-frequency per frequency bucket (shared across x/y/z)
        # Classic RoPE-style geometric schedule across frequencies:
        self.inv_freq = self.base ** ( - np.arange(self.num_freq, dtype=np.float64) / self.num_freq )
        self._cache = {}

    def forward(self, s):
        """
        For 4D absolute position s=(t,x,y,z), build a list of (Lx,Ly,Lz) 4×4 matrices
        per frequency. Returns a Python list of length NUM_FREQ, each item is a tuple
        (Lx, Ly, Lz).
        """
        s = np.asarray(s, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError("position must be a 4D vector (t, x, y, z).")
        key = tuple(s.tolist())
        if key in self._cache:
            return self._cache[key]

        t, x, y, z = s
        out = []
        for j in range(self.num_freq):
            om = self.inv_freq[j]
            Lx = Lx_from_pos(t, x, om)
            Ly = Ly_from_pos(t, y, om)
            Lz = Lz_from_pos(t, z, om)
            out.append((Lx, Ly, Lz))
        self._cache[key] = out
        return out

# 4) Apply block-diagonal triad transforms across the embedding
def apply_monster_triad(emb, L_tables, dim=DIM):
    """
    emb: (dim,) row vector
    L_tables: list of (Lx,Ly,Lz) per frequency from TriadMonSTER.forward(...)
    Returns transformed embedding of shape (dim,)
    """
    if emb.shape != (dim,):
        raise ValueError(f"embedding must be shape ({dim},), got {emb.shape}")
    if len(L_tables) * SLICE != dim:
        raise ValueError("L_tables length doesn't match embedding dim.")

    out = np.empty_like(emb)
    for j, (Lx, Ly, Lz) in enumerate(L_tables):
        base = j * SLICE
        # X slice
        bx = emb[base+0 : base+4]
        out[base+0 : base+4] = apply_lorentz_row(bx, Lx)
        # Y slice
        by = emb[base+4 : base+8]
        out[base+4 : base+8] = apply_lorentz_row(by, Ly)
        # Z slice
        bz = emb[base+8 : base+12]
        out[base+8 : base+12] = apply_lorentz_row(bz, Lz)
    return out

# 5) Big Minkowski inner product for triad layout (sum of per-4D blocks)
def minkowski_dot_big(u, v, dim=DIM):
    if u.shape != (dim,) or v.shape != (dim,):
        raise ValueError("inputs must be flat row vectors of length dim.")
    total = 0.0
    for i in range(0, dim, 4):
        total += minkowski_dot4(u[i:i+4], v[i:i+4])
    return total

# ============================================================
# Demo / tiny tests
# ============================================================
if __name__ == "__main__":
    np.random.seed(0)

    triad = TriadMonSTER(dim=DIM, base=10000.0)

    # Random 4D absolute positions
    s_q  = np.array([ 7.0,  5.0, -3.0,  2.0], dtype=np.float64)
    s_k  = np.array([-4.0, -2.0,  6.0, -1.0], dtype=np.float64)
    dskq = s_k - s_q          # relative displacement

    # Precompute per-position transforms
    L_abs_q = triad.forward(s_q)      # list of (Lx,Ly,Lz)
    L_abs_k = triad.forward(s_k)
    L_rel   = triad.forward(dskq)     # should give the "relative" operator

    # Random embeddings (row vectors)
    q = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)
    k = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)

    # Apply absolute transforms
    q_abs = apply_monster_triad(q, L_abs_q)
    k_abs = apply_monster_triad(k, L_abs_k)

    # Check RoPE-style identity:
    # < L(s_q) q , L(s_k) k >_ηbig  ==  < q , L(s_k - s_q) k >_ηbig
    lhs = minkowski_dot_big(q_abs, k_abs)
    k_rel = apply_monster_triad(k, L_rel)
    rhs = minkowski_dot_big(q, k_rel)

    print("RoPE-style identity holds? ", np.allclose(lhs, rhs, rtol=1e-10, atol=1e-12))
    print(f"lhs: {lhs:+.12f}  rhs: {rhs:+.12f}")

    # Minkowski norm preservation per 4D block (Lorentz isometry)
    # For each 4D slice, ||v||_η should be preserved by any Lx/Ly/Lz.
    norms_before = [minkowski_dot4(q[i:i+4], q[i:i+4]) for i in range(0, DIM, 4)]
    norms_after  = [minkowski_dot4(q_abs[i:i+4], q_abs[i:i+4]) for i in range(0, DIM, 4)]
    print("Per-4D Minkowski norms preserved? ", np.allclose(norms_before, norms_after, rtol=1e-10, atol=1e-12))

    # Quick shape sanity
    print("NUM_FREQ:", NUM_FREQ, " | DIM:", DIM, " | SLICE per freq:", SLICE)
