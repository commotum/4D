import numpy as np

# This script demonstrates that the principle of Rotary Positional Embeddings (RoPE)
# can be extended to a 4D Minkowski spacetime. The core idea is that the inner
# product between two vectors, each transformed according to its absolute position,
# depends only on the *relative displacement* between them.
#
# We achieve this by:
# 1. Defining a 4D Minkowski metric (+, -, -, -).
# 2. Creating Lorentz transformations from commuting generators (a boost and a
#    rotation sharing the same axis).
# 3. Defining a linear map from a 4D position vector `s` to the parameters of
#    the Lorentz transformation `L(s)`.
#
# The key mathematical property we verify is:
# <L(s_q)q, L(s_k)k>_η = <q, L(s_k - s_q)k>_η
#
# This means the "attention score" only sees the difference vector `Δs = s_k - s_q`,
# making the positions relative.

# --- 1. Setup: Spacetime & Transformations ---

# The Minkowski metric tensor with signature (+, -, -, -).
ETA = np.diag([1.0, -1.0, -1.0, -1.0]).astype(np.float64)

def minkowski_dot(u, v):
    """
    Calculates the Minkowski inner product <u, v>_η = u_t*v_t - u_x*v_x - ...
    For 1D numpy arrays u and v, the expression (u @ ETA) @ v correctly
    computes this sum.
    """
    return (u @ ETA) @ v

def apply_lorentz_transform(vector, transform_matrix):
    """
    Applies a Lorentz transformation L to a row vector v.
    The standard column vector transformation is v' = L @ v.
    For a row vector, this becomes v'_row = v_row @ L.T.
    """
    return vector @ transform_matrix.T

def boost_x(rapidity):
    """
    Generates a pure Lorentz boost along the +x axis.
    """
    phi = float(rapidity)
    ch, sh = np.cosh(phi), np.sinh(phi)
    L = np.eye(4, dtype=np.float64)
    L[0, 0], L[0, 1] = ch, -sh
    L[1, 0], L[1, 1] = -sh, ch
    return L

def rot_x(theta):
    """
    Generates a spatial rotation in the y-z plane (around the x-axis).
    """
    c, s = np.cos(theta), np.sin(theta)
    L = np.eye(4, dtype=np.float64)
    L[2, 2], L[2, 3] = c, -s
    L[3, 2], L[3, 3] = s, c
    return L

# --- 2. Setup: Position-to-Transformation Mapping ---

# We define a linear mapping from a 4D position vector `s` to the
# transformation parameters (rapidity `alpha` and angle `beta`).
# Tweak these weights to change the sensitivity to position.
KAPPA = np.array([ 0.010,  0.005, -0.007,  0.003], dtype=np.float64)  # Weights for rapidity
RHO   = np.array([ 0.005, -0.004,  0.006, -0.003], dtype=np.float64)  # Weights for angle

def get_transform_from_position(s):
    """
    Builds a composite Lorentz transformation L(s) from a position vector s.
    L(s) = Rot_x(β(s)) @ Boost_x(α(s)), where α(s)=KAPPA·s and β(s)=RHO·s.
    This works because the boost and rotation generators commute, ensuring that
    L(s1) @ L(s2) = L(s1 + s2).
    """
    s = np.asarray(s, dtype=np.float64)
    rapidity_alpha = KAPPA @ s
    angle_beta = RHO @ s
    
    # The order of multiplication does not matter since they commute.
    return rot_x(angle_beta) @ boost_x(rapidity_alpha)

# --- 3. Experiment: Relative Positions ---

# Define a single pair of query (q) and key (k) vectors.
# Using the same vectors for both tests isolates the effect of position.
q = np.array([+0.09, +0.07, +0.05, +0.03], dtype=np.float64)
k = np.array([+0.06, +0.04, +0.02, +0.01], dtype=np.float64)

# Define two pairs of positions. They have different absolute values
# but share the exact same displacement vector.
pos1_q = np.array([25, 25, 15, 23], dtype=np.float64)
pos1_k = np.array([27, 20,  7, 15], dtype=np.float64)
delta_1 = pos1_k - pos1_q
print(f"Position Pair 1: \n q_pos={pos1_q}, \n k_pos={pos1_k}")
print(f"Relative Displacement: {delta_1}\n")

pos2_q = np.array([28, 20,  9, 15], dtype=np.float64)
pos2_k = np.array([30, 15,  1,  7], dtype=np.float64)
delta_2 = pos2_k - pos2_q
print(f"Position Pair 2: q_pos={pos2_q}, k_pos={pos2_k}")
print(f"Relative Displacement: {delta_2}\n")

# Verify that the displacements are indeed identical.
assert np.all(delta_1 == delta_2), "Displacements do not match!"
print("Displacement vectors are identical, as required for the test.")

# --- 4. Calculations & Verification ---

print("\n--- Verification of RoPE Principle in 4D ---")

# Test 1: Apply transformations for the first pair of positions.
L1_q = get_transform_from_position(pos1_q)
L1_k = get_transform_from_position(pos1_k)
q_transformed_1 = apply_lorentz_transform(q, L1_q)
k_transformed_1 = apply_lorentz_transform(k, L1_k)
dot_product_1 = minkowski_dot(q_transformed_1, k_transformed_1)

# Test 2: Apply transformations for the second pair of positions.
L2_q = get_transform_from_position(pos2_q)
L2_k = get_transform_from_position(pos2_k)
q_transformed_2 = apply_lorentz_transform(q, L2_q)
k_transformed_2 = apply_lorentz_transform(k, L2_k)
dot_product_2 = minkowski_dot(q_transformed_2, k_transformed_2)

# The punchline: Because the relative displacement is the same for both pairs,
# the resulting Minkowski inner products should be identical.
print(f"Dot product for Pair 1: {dot_product_1:.9f}")
print(f"Dot product for Pair 2: {dot_product_2:.9f}\n")

print("Are the dot products equal?")
print(f"Answer: {np.allclose(dot_product_1, dot_product_2)}")
print(f"Difference: {abs(dot_product_1 - dot_product_2)}")

