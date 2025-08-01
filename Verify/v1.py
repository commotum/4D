# Original implementation of verify.py shows that the RoPE principle holds for
# both 2D and 4D vectors. Some factual and mathematical errors.


import numpy as np

# Input vectors
a = np.array([+0.09, +0.07, +0.05, +0.03], dtype=np.float32)
b = np.array([+0.06, +0.04, +0.02, +0.01], dtype=np.float32)
c = np.array([-0.01, -0.02, -0.03, -0.04], dtype=np.float32)
d = np.array([-0.05, -0.06, -0.07, -0.08], dtype=np.float32)

# Precomputed dot products (as provided)
token_pairs = [
    (a, a),
    (a, b),
    (a, c),
    (a, d),
    (b, b),
    (b, c),
    (b, d),
    (c, c),
    (c, d),
    (d, d)
]

dot_products = []

for i, j in token_pairs:
    dot = np.dot(i, j)
    dot_products.append(dot)


# Position Pairs

position_pairs = [
    (7, 2),
    (10, 5)
]

# Positions
m = 7
n = 2
i = 10
j = 5

# Theta values for each 2D group
theta = np.array([0.1, 0.2], dtype=np.float32)  # Different theta for each 2D pair

# Transformation function for a 2D vector at position n
def apply_rotation_2d(vec_2d, pos, theta):
    x, y = vec_2d
    cos_nt = np.cos(pos * theta)
    sin_nt = np.sin(pos * theta)
    return np.array([
        x * cos_nt - y * sin_nt,
        x * sin_nt + y * cos_nt
    ], dtype=np.float32)

# Apply rotation to a 4D vector (two 2D groups)
def apply_rotation_4d(vec, pos, theta):
    # Split into two 2D vectors
    vec1 = vec[:2]
    vec2 = vec[2:]
    # Apply rotation to each 2D group
    rotated1 = apply_rotation_2d(vec1, pos, theta[0])
    rotated2 = apply_rotation_2d(vec2, pos, theta[1])
    # Concatenate results
    return np.concatenate([rotated1, rotated2])

# Apply rotations to vectors at positions m, n, i, j
a_m = apply_rotation_4d(a, m, theta)  # q_m e^{im theta}
b_n = apply_rotation_4d(b, n, theta)  # k_n e^{in theta}
c_i = apply_rotation_4d(c, i, theta)  # q_i e^{i i theta}
d_j = apply_rotation_4d(d, j, theta)  # k_j e^{i j theta}

# Compute dot products after rotation
dot_mn = np.dot(a_m, b_n)  # <q_m e^{im theta}, k_n e^{in theta}>
dot_ij = np.dot(c_i, d_j)  # <q_i e^{i i theta}, k_j e^{i j theta}>

# Verify using complex number formulation for relative position
def complex_dot_product_4d(q, k, delta_pos, theta):
    # Split into two 2D vectors
    q1, q2 = q[:2], q[2:]
    k1, k2 = k[:2], k[2:]
    # Treat as complex numbers
    q1_complex = q1[0] + 1j * q1[1]
    q2_complex = q2[0] + 1j * q2[1]
    k1_complex = k1[0] + 1j * k1[1]
    k2_complex = k2[0] + 1j * k2[1]
    # Compute Re[q k^* e^{i(m-n)theta}] for each pair
    result1 = np.real(q1_complex * np.conj(k1_complex) * np.exp(1j * delta_pos * theta[0]))
    result2 = np.real(q2_complex * np.conj(k2_complex) * np.exp(1j * delta_pos * theta[1]))
    return result1 + result2

# Compute relative position dot products
delta_mn = m - n  # 7 - 2 = 5
delta_ij = i - j  # 10 - 5 = 5
complex_dot_mn = complex_dot_product_4d(a, b, delta_mn, theta)
complex_dot_ij = complex_dot_product_4d(c, d, delta_ij, theta)

# Print results to compare
print(f"Dot product <q_m e^(im theta), k_n e^(in theta)> (m=7, n=2): {dot_mn:.6f}")
print(f"Complex formulation Re[q_m k_n* e^(i(m-n)theta)]: {complex_dot_mn:.6f}")
print(f"Dot product <q_i e^(i i theta), k_j e^(i j theta)> (i=10, j=5): {dot_ij:.6f}")
print(f"Complex formulation Re[q_i k_j* e^(i(i-j)theta)]: {complex_dot_ij:.6f}")

# Check if relative position dot products are equal for same m-n
# Since m-n = 5 and i-j = 5, the results should be similar
print(f"\nComparing relative positions m-n = {delta_mn} and i-j = {delta_ij}:")
print(f"Dot product difference: {abs(dot_mn - dot_ij):.6f}")
print(f"Complex dot difference: {abs(complex_dot_mn - complex_dot_ij):.6f}")

print()
print("Lorentz Rotors:")


# Positions
# The two pairs of vectors, while different, share a unique difference/displacement vector
# 
STA_1 = np.array([25, 25, 15, 23], dtype=np.float32)
STA_2 = np.array([27, 20, 7, 15], dtype=np.float32)

STA_3 = np.array([28, 20, 9, 15], dtype=np.float32)
STA_4 = np.array([30, 15, 1, 7], dtype=np.float32)

import numpy as np

# ---------------------------
# Setup: Minkowski (+, -, -, -)
# ---------------------------
ETA = np.diag([1.0, -1.0, -1.0, -1.0])

def minkowski_dot(u, v):
    """<u, v>_eta for row vectors u, v (shape (4,))."""
    return float(u @ ETA @ v)

def apply_lorentz(vec, L):
    """Row-vector action v' = v @ L.T (consistent with our matrix convention)."""
    return vec @ L.T

def check_lorentz(L, tol=1e-10):
    return np.allclose(L.T @ ETA @ L, ETA, atol=tol)

# ---------------------------
# Commuting generators: boost_x and rot_x
# ---------------------------
def boost_x(rapidity):
    """
    Pure boost along +x with rapidity φ in (+---) signature:
      t' =  coshφ * t - sinhφ * x
      x' = -sinhφ * t + coshφ * x
    """
    φ = float(rapidity)
    ch, sh = np.cosh(φ), np.sinh(φ)
    L = np.eye(4)
    L[0, 0] =  ch; L[0, 1] = -sh
    L[1, 0] = -sh; L[1, 1] =  ch
    # y,z unchanged
    return L

def rot_x(theta):
    """
    Spatial rotation about x (i.e., in the yz-plane) by angle θ:
      y' =  cosθ * y - sinθ * z
      z' =  sinθ * y + cosθ * z
    Time is unchanged.
    """
    θ = float(theta)
    c, s = np.cos(θ), np.sin(θ)
    L = np.eye(4)
    L[2, 2] =  c; L[2, 3] = -s
    L[3, 2] =  s; L[3, 3] =  c
    return L

# Since boost_x and rot_x share axis x, they COMMUTE: L = rot_x(β) @ boost_x(α) = boost_x(α) @ rot_x(β)

# ---------------------------
# Position -> rotor mapping (linear maps to keep the "relative-only" property)
# ---------------------------
# You can tweak these weights; keep them modest to avoid huge rapidities/angles.
kappa = np.array([ 0.010,  0.005, -0.007,  0.003], dtype=np.float64)  # for rapidity α(s)
rho   = np.array([ 0.005, -0.004,  0.006, -0.003], dtype=np.float64)  # for angle    β(s)

def L_from_position(s):
    """Build L(s) = Rot_x(β(s)) * Boost_x(α(s)) with α(s)=kappa·s, β(s)=rho·s."""
    s = np.asarray(s, dtype=np.float64).reshape(4)
    α = float(kappa @ s)
    β = float(rho   @ s)
    # Because they commute, the order doesn't matter; pick one and stick with it.
    return rot_x(β) @ boost_x(α)

# ---------------------------
# Given data (your original embeddings and positions)
# ---------------------------
# Token embeddings
a = np.array([+0.09, +0.07, +0.05, +0.03], dtype=np.float64)
b = np.array([+0.06, +0.04, +0.02, +0.01], dtype=np.float64)
c = np.array([-0.01, -0.02, -0.03, -0.04], dtype=np.float64)
d = np.array([-0.05, -0.06, -0.07, -0.08], dtype=np.float64)

# Positions (two pairs with the same displacement)
STA_1 = np.array([25, 25, 15, 23], dtype=np.float64)
STA_2 = np.array([27, 20,  7, 15], dtype=np.float64)

STA_3 = np.array([28, 20,  9, 15], dtype=np.float64)
STA_4 = np.array([30, 15,  1,  7], dtype=np.float64)

# Displacements
Δ12 = STA_2 - STA_1
Δ34 = STA_4 - STA_3

# Sanity: the two displacements should match
print("Δ12:", Δ12)
print("Δ34:", Δ34)
print("Same displacement? ", np.all(Δ12 == Δ34))

# ---------------------------
# Build Lorentz transforms
# ---------------------------
L1 = L_from_position(STA_1)
L2 = L_from_position(STA_2)
L3 = L_from_position(STA_3)
L4 = L_from_position(STA_4)
LΔ = L_from_position(Δ12)  # = L_from_position(Δ34)

# Validate Lorentz property
assert check_lorentz(L1) and check_lorentz(L2) and check_lorentz(L3) and check_lorentz(L4) and check_lorentz(LΔ)

# Also check the key matrix identity: L(s1)^T η L(s2) = η L(s2 - s1)
lhs_12 = L1.T @ ETA @ L2
rhs_12 = ETA @ L_from_position(Δ12)
print("Matrix identity holds (pair 1)?", np.allclose(lhs_12, rhs_12, atol=1e-10))

lhs_34 = L3.T @ ETA @ L4
rhs_34 = ETA @ L_from_position(Δ34)
print("Matrix identity holds (pair 2)?", np.allclose(lhs_34, rhs_34, atol=1e-10))

# ---------------------------
# Apply transforms and compare inner products
# ---------------------------
# Absolute-position transforms
a_1 = apply_lorentz(a, L1)
b_2 = apply_lorentz(b, L2)
c_3 = apply_lorentz(c, L3)
d_4 = apply_lorentz(d, L4)

# Inner products after separate absolute transforms
dot12_abs = minkowski_dot(a_1, b_2)
dot34_abs = minkowski_dot(c_3, d_4)

# "Fused" relative transform predictions
# <L(s1)a, L(s2)b>_η = <a, L(s2-s1)b>_η when generators commute and mapping is linear
b_rel_12 = apply_lorentz(b, LΔ)    # Δ = STA_2 - STA_1
d_rel_34 = apply_lorentz(d, LΔ)    # same Δ

dot12_rel = minkowski_dot(a, b_rel_12)
dot34_rel = minkowski_dot(c, d_rel_34)

# Print and compare
print("\n--- Results ---")
print(f"<L(s1)a, L(s2)b>_η  (pair 1): {dot12_abs:.9f}")
print(f"<a, L(Δ)b>_η        (pair 1): {dot12_rel:.9f}")
print("Difference (pair 1):", abs(dot12_abs - dot12_rel))

print(f"\n<L(s3)c, L(s4)d>_η  (pair 2): {dot34_abs:.9f}")
print(f"<c, L(Δ)d>_η        (pair 2): {dot34_rel:.9f}")
print("Difference (pair 2):", abs(dot34_abs - dot34_rel))


"""
Terminal Output:

Dot product <q_m e^(im theta), k_n e^(in theta)> (m=7, n=2): 0.007527
Complex formulation Re[q_m k_n* e^(i(m-n)theta)]: 0.007527
Dot product <q_i e^(i i theta), k_j e^(i j theta)> (i=10, j=5): 0.003827
Complex formulation Re[q_i k_j* e^(i(i-j)theta)]: 0.003827

Comparing relative positions m-n = 5 and i-j = 5:
Dot product difference: 0.003700
Complex dot difference: 0.003700

Lorentz Rotors:
Δ12: [ 2. -5. -8. -8.]
Δ34: [ 2. -5. -8. -8.]
Same displacement?  True
Matrix identity holds (pair 1)? True
Matrix identity holds (pair 2)? True

--- Results ---
<L(s1)a, L(s2)b>_η  (pair 1): 0.001316573
<a, L(Δ)b>_η        (pair 1): 0.001316573
Difference (pair 1): 8.673617379884035e-19

<L(s3)c, L(s4)d>_η  (pair 2): -0.005991758
<c, L(Δ)d>_η        (pair 2): -0.005991758
Difference (pair 2): 0.0
"""