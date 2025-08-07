###############################################################################
#  RoPE-style Minkowski Rotor Demo
#  --------------------------------
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
# 1.  Minkowski metric (+ – – –) and helpers
# --------------------------------------------------------------------------- #

# This line creates the Minkowski metric tensor, 
# commonly represented by η, the Greek letter eta.
η = np.diag([1.0, -1.0, -1.0, -1.0]).astype(np.float64)

# 1. Apply the metric to u: (u @ η) = u' (flipping spatial components)
# 2. Compute final dot product (u'⋅v) = (u_t*v_t) - (u_s⋅v_s).
def minkowski_dot(u, v):
    """<u,v>_η  for row vectors."""
    return (u @ η) @ v

# 1. Prepare L for row-vector action by transposing: L → Lᵀ.
# 2. Left-multiply by the vector to get the transformed vector: v' = v @ Lᵀ.
def apply_lorentz(vector, L):
    """Row-vector action  v' = v · Lᵀ."""
    return vector @ L.T

# --------------------------------------------------------------------------- #
# 2.  Generators for an arbitrary unit axis  û
# --------------------------------------------------------------------------- #
u_hat = np.array([1.0, 1.0, 1.0])
u_hat /= np.linalg.norm(u_hat)          # (1,1,1)/√3

def boost_u(phi, u=u_hat):
    """Lorentz boost mixing (t, û·r)."""
    ch, sh = np.cosh(phi), np.sinh(phi)
    L = np.eye(4)
    L[0,0] = ch
    L[0, 1:]  = -sh * u                 # time–space
    L[1:, 0]  = -sh * u
    L[1:, 1:] += (ch - 1.0) * np.outer(u, u)  # spatial block
    return L.astype(np.float64)

def rot_u(theta, u=u_hat):
    """Spatial rotation by θ about axis û (Rodrigues)."""
    c, s = np.cos(theta), np.sin(theta)
    ux, uy, uz = u
    K = np.array([[  0, -uz,  uy],
                  [ uz,   0, -ux],
                  [-uy,  ux,   0]])
    R3 = c * np.eye(3) + (1 - c) * np.outer(u, u) + s * K
    L = np.eye(4)
    L[1:, 1:] = R3
    return L.astype(np.float64)

# --------------------------------------------------------------------------- #
# 3.  RoPE-style inverse-frequency ladder
# --------------------------------------------------------------------------- #
 
DIM         = 512                         # embedding dim (== vector dim here)
NUM_BLOCKS  = DIM // 4

if DIM % NUM_BLOCKS != 0:
    raise ValueError(f"DIM ({DIM}) must be divisible by 4. Got DIM % NUM_BLOCKS={DIM%NUM_BLOCKS}")

inv_freq    = 1.0 / (10000 ** (np.arange(0, DIM, 4) / DIM))
FREQ_INDEX  = 0                         # pick the lowest frequency for demo

def L_from_position(s, idx=FREQ_INDEX):
    """
    Build L(s) = Rot_u(θ) · Boost_u(φ)
      φ =  t                 · inv_freq[idx]
      θ = (û·r)              · inv_freq[idx]
    """
    s = np.asarray(s, dtype=np.float64)
    t, r = s[0], s[1:]
    φ = t            * inv_freq[idx]
    θ = (u_hat @ r)  * inv_freq[idx]
    # generators commute ⇒ order irrelevant
    return rot_u(θ) @ boost_u(φ)

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
    L_q = L_from_position(sq)
    L_k = L_from_position(sk)
    q_t = apply_lorentz(q_vec, L_q)
    k_t = apply_lorentz(k_vec, L_k)
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
