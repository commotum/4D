###############################################################################
#  RoPE-style Minkowski Rotor Demo (using Biquaternions)
#  ------------------------------------------------------
#  • One 10000-based “inverse-frequency ladder” (as in classic RoPE)
#  • A *single* fixed spatial axis  û = (1,1,1)/√3 shared by boost & rotation
#  • Scalars are linear:
#        φ(s) =  t · inv_freq[i]          (rapidity  → boost along û)
#        θ(s) = (û·r) · inv_freq[i]       (angle     → rotation around û)
#  • Because generators commute and the maps are linear:
#        L(s₂) · L(s₁)⁻¹  =  L(s₂ − s₁)
#    so the inner-product fusion <L(s_q)q , L(s_k)k> = <q , L(Δs)k> holds
###############################################################################

import numpy as np

# --------------------------------------------------------------------------- #
# 1.  Biquaternion class
# --------------------------------------------------------------------------- #
class Biquaternion:
    def __init__(self, w=0, x=0, y=0, z=0):
        self.w = complex(w)
        self.x = complex(x)
        self.y = complex(y)
        self.z = complex(z)

    def __add__(self, other):
        return Biquaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, other):
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Biquaternion(w, x, y, z)

    def quat_conj(self):
        return Biquaternion(self.w, -self.x, -self.y, -self.z)

    def complex_conj(self):
        return Biquaternion(self.w.conjugate(), self.x.conjugate(), self.y.conjugate(), self.z.conjugate())

# --------------------------------------------------------------------------- #
# 2.  Minkowski metric (+ – – –) and helpers
# --------------------------------------------------------------------------- #
ETA = np.diag([1.0, -1.0, -1.0, -1.0]).astype(np.float64)

def minkowski_dot(u, v):
    """<u,v>_η  for row vectors."""
    return (u @ ETA) @ v

def vector_to_bq(v):
    """Embed 4D vector as minquat."""
    return Biquaternion(v[0], 1j * v[1], 1j * v[2], 1j * v[3])

def bq_to_vector(q):
    """Extract 4D vector from minquat."""
    return np.array([q.w.real, q.x.imag, q.y.imag, q.z.imag], dtype=np.float64)

def apply_biquat(g, v):
    """Apply biquaternion Lorentz transformation to vector."""
    q = vector_to_bq(v)
    bar_g = g.quat_conj()
    bar_g_star = bar_g.complex_conj()
    q_trans = g * (q * bar_g_star)
    return bq_to_vector(q_trans)

# --------------------------------------------------------------------------- #
# 3.  RoPE-style inverse-frequency ladder
# --------------------------------------------------------------------------- #
DIM         = 4                         # embedding dim (== vector dim here)
inv_freq    = 1.0 / (10000 ** (np.arange(0, DIM, 4) / DIM))  # length 4
FREQ_INDEX  = 0                         # pick the lowest frequency for demo

u_hat = np.array([1.0, 1.0, 1.0])
u_hat /= np.linalg.norm(u_hat)          # (1,1,1)/√3

def g_from_position(s, idx=FREQ_INDEX):
    """
    Build g(s) = g_rot * g_boost
      φ =  t                 · inv_freq[idx]
      θ = (û·r)              · inv_freq[idx]
    """
    s = np.asarray(s, dtype=np.float64)
    t, r = s[0], s[1:]
    phi = t * inv_freq[idx]
    theta = (u_hat @ r) * inv_freq[idx]

    # g_boost (with sign to match standard -sh convention)
    ch = np.cosh(phi / 2)
    sh = np.sinh(phi / 2)
    g_boost = Biquaternion(ch, 0, 0, 0) + Biquaternion(0, -1j * sh * u_hat[0], -1j * sh * u_hat[1], -1j * sh * u_hat[2])

    # g_rot
    co = np.cos(theta / 2)
    si = np.sin(theta / 2)
    g_rot = Biquaternion(co, 0, 0, 0) + Biquaternion(0, si * u_hat[0], si * u_hat[1], si * u_hat[2])

    # generators commute ⇒ order irrelevant
    return g_rot * g_boost

# --------------------------------------------------------------------------- #
# 4.  Random experiment data (unchanged logic)
# --------------------------------------------------------------------------- #
def generate_experiment_data(
    dim: int = 4,
    pos_range: tuple = (-15, 15),
    vec_range: tuple = (-0.65, 0.65),
    seed: int = None
):
    if seed is not None:
        np.random.seed(seed)

    q     = np.random.uniform(*vec_range, size=dim)
    k     = np.random.uniform(*vec_range, size=dim)

    delta = np.random.randint(pos_range[0], pos_range[1] + 1, size=dim)

    pos1_q = np.random.randint(*pos_range, size=dim)
    pos2_q = np.random.randint(*pos_range, size=dim)

    pos1_k = pos1_q + delta
    pos2_k = pos2_q + delta
    return q, k, pos1_q, pos1_k, pos2_q, pos2_k, delta

q, k, pos1_q, pos1_k, pos2_q, pos2_k, delta = generate_experiment_data(seed=42)

# --------------------------------------------------------------------------- #
# 5.  Transform, compare inner products
# --------------------------------------------------------------------------- #
def inner_product_after_transform(q_vec, k_vec, sq, sk):
    g_q = g_from_position(sq)
    g_k = g_from_position(sk)
    q_t = apply_biquat(g_q, q_vec)
    k_t = apply_biquat(g_k, k_vec)
    return minkowski_dot(q_t, k_t)

dot1 = inner_product_after_transform(q, k, pos1_q, pos1_k)
dot2 = inner_product_after_transform(q, k, pos2_q, pos2_k)

# --------------------------------------------------------------------------- #
# 6.  Report
# --------------------------------------------------------------------------- #
def fmt(v, prec=2):
    return "[" + ", ".join(f"{x:+.{prec}f}" for x in v) + "]"

print("û axis:", fmt(u_hat, 4))
print("inv_freq ladder:", inv_freq)
print()

print("Pair-1 positions  s_q:", pos1_q, " s_k:", pos1_k)
print("Pair-2 positions  s_q:", pos2_q, " s_k:", pos2_k)
print("Common Δs:", delta, "\n")

print("Dot product (pair 1):", f"{dot1:+.9f}")
print("Dot product (pair 2):", f"{dot2:+.9f}")
print("Equal to machine tol?:", np.allclose(dot1, dot2))