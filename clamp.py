import numpy as np
import random
import itertools

# Round all NumPy array prints to 3 decimals
import numpy as np
import itertools
import random

np.set_printoptions(precision=3, suppress=True)

def generate_even_intervals(n):
    """n intervals → n+1 evenly spaced t in [-1,1]."""
    return np.linspace(-1.0, 1.0, n + 1)

def generate_signed_vectors(m):
    """
    m random 3-vectors in [0,16]^3, each with 0–3 coords flipped negative.
    Returns array of shape (m,3).
    """
    vecs = []
    for _ in range(m):
        v = np.random.randint(0, 17, size=3).astype(float)
        k = random.choice([0,1,2,3])
        # flip k randomly chosen coordinates
        for idx in np.random.choice(3, size=k, replace=False):
            v[idx] *= -1
        vecs.append(v)
    return np.stack(vecs)

def generate_four_vectors(n, m):
    """Combine (n+1) ts with m spatial vecs → list of 4-vectors (t,x,y,z)."""
    ts = generate_even_intervals(n)
    xs = generate_signed_vectors(m)
    return [np.array([t, x, y, z], float)
            for t, (x, y, z) in itertools.product(ts, xs)]

def generate_random_vectors(num, dim, std=None, seed=None):
    """Generate `num` vectors in R^dim with entries uniformly sampled from {1,2,3,4}."""
    if seed is not None:
        np.random.seed(seed)
    # draw random integers 1-4 inclusive
    return np.random.randint(1, 5, size=(num, dim)).astype(float)

def build_rotation_matrix(axis, theta):
    """Rodrigues’ 3×3 rotation about unit `axis` by angle `theta`."""
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    ct, st = np.cos(theta), np.sin(theta)
    oc = 1 - ct
    return np.array([
        [ct+ux*ux*oc,    ux*uy*oc-uz*st, ux*uz*oc+uy*st],
        [uy*ux*oc+uz*st, ct+uy*uy*oc,    uy*uz*oc-ux*st],
        [uz*ux*oc-uy*st, uz*uy*oc+ux*st, ct+uz*uz*oc   ]
    ])

def get_delta_monster(delta_t, delta_xyz, B, base_time, base_space, eps=1e-8):
    """
    Compute B block-wise 4×4 spacetime rotors (rotation+boost) for ΔP=(Δt,Δx).
    
    We drop the tanh clamp because |delta_t| ≤ 1 by construction
    (t is normalized in [-1,1]), and inv_freq ≤ 1 ⇒ φ_b=delta_t*inv_freq ≤ 1.
    Thus cosh(φ_b) ≤ cosh(1)≈1.54 and sinh(φ_b)≤sinh(1)≈1.18,
    so there is no risk of overflow in float32.

    Returns array of shape (B, 4, 4).
    """
    freqs = np.arange(B, dtype=float)
    inv_t = 1.0 / (base_time ** (freqs / B))
    inv_s = 1.0 / (base_space ** (freqs / B))

    # since |delta_t| ≤ 1 and inv_t ≤ 1, φ_b = delta_t * inv_t ≤ 1
    dt = delta_t * inv_t               # shape (B,)
    ds = delta_xyz[None, :] * inv_s[:, None]  # shape (B,3)

    # rotation: angle θ_b = ||ds_b||, axis = ds_b/θ_b
    theta = np.linalg.norm(ds, axis=1)
    axes = np.where(theta[:, None] < eps,
                    np.array([0, 0, 1.0]),
                    ds / theta[:, None])  # shape (B,3)

    # build blockwise spatial rotations
    R_rot = np.zeros((B, 4, 4))
    R_rot[:, 0, 0] = 1.0
    for b in range(B):
        R_rot[b, 1:, 1:] = build_rotation_matrix(axes[b], theta[b])

    # build blockwise boosts with φ_b = dt_b
    ch = np.cosh(dt)
    sh = np.sinh(dt)

    R_boost = np.zeros((B, 4, 4))
    R_boost[:, 0, 0] = ch
    R_boost[:, 0, 1:] = -axes * sh[:, None]
    R_boost[:, 1:, 0] = -axes * sh[:, None]
    I3 = np.eye(3)
    for b in range(B):
        R_boost[b, 1:, 1:] = I3 + (ch[b] - 1.0) * np.outer(axes[b], axes[b])

    # combined rotor: boost @ rotation
    return R_boost @ R_rot

def minkowski_dot(a, b):
    """Minkowski inner product η(a,b)=a0*b0 - ∑_{i=1}^3 ai*bi."""
    return a[0]*b[0] - np.dot(a[1:], b[1:])

# ——— Configuration (edit before run) —————
num_intervals     = 6       # number of time intervals
num_three_vectors = 10       # number of random spatial vectors
num_tokens        = 20       # how many tokens to sample
hidden_dim        = 512     # embedding dimensionality
std               = 0.006   # (unused)
base_time         = 10000.  # temporal frequency base
base_space        = 10000.  # spatial frequency base
B = hidden_dim // 4         # number of 4×4 blocks
# ————————————————————————————————————————

# generate token embeddings and 4-vectors
hidden_vecs = generate_random_vectors(num_tokens, hidden_dim, seed=42)
four_vs_all = generate_four_vectors(num_intervals, num_three_vectors)
token_4vs   = random.sample(four_vs_all, num_tokens)

# use the first token as reference
p4 = token_4vs[0]
h0 = hidden_vecs[0].reshape(B, 4)

for i in range(num_tokens):
    q4 = token_4vs[i]
    hi = hidden_vecs[i].reshape(B, 4)

    # compute spacetime rotor blocks for ΔP = q4 - p4
    dt = q4[0] - p4[0]
    dxyz = q4[1:] - p4[1:]
    Rb = get_delta_monster(dt, dxyz, B, base_time, base_space)

    # apply each 4×4 rotor to the 4-dim slice of the key
    hi_rot = np.einsum('bij,bj->bi', Rb, hi)

    # compute attention scores as full 512-d Minkowski dot
    orig_score = sum(minkowski_dot(h0[b], hi[b]) for b in range(B))
    rot_score  = sum(minkowski_dot(h0[b], hi_rot[b]) for b in range(B))

    # print results, rounded to 3 decimals
    print(f"\nToken #{i}:")
    print("  4-vector p:", np.round(p4, 3))
    print("  4-vector q:", np.round(q4, 3))
    print("  Minkowski⟨p,q⟩       =", round(minkowski_dot(p4, q4), 3))
    print("  Orig Mink attention  =", round(orig_score, 3))
    print("  Rot  Mink attention  =", round(rot_score, 3))
