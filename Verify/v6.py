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

# --- 3. Experiment Data ---

import numpy as np

def generate_experiment_data(
    dim: int = 4,
    pos_range: tuple = (0, 50),
    vec_range: tuple = (-0.65, 0.65),
    seed: int = None
):
    """
    Generates random experimental data for RoPE 4D Minkowski test.

    Args:
        dim: Dimension of the vectors and positions (default: 4).
        pos_range: Tuple (min, max) for random integer positions.
        vec_range: Tuple (min, max) for random float vector elements.
        seed: Optional random seed for reproducibility.

    Returns:
        q: np.ndarray, shape (dim,)
        k: np.ndarray, shape (dim,)
        pos1_q: np.ndarray, shape (dim,)
        pos1_k: np.ndarray, shape (dim,)
        pos2_q: np.ndarray, shape (dim,)
        pos2_k: np.ndarray, shape (dim,)
        delta: np.ndarray, shape (dim,)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate q and k with each element between vec_range[0] and vec_range[1]
    q = np.random.uniform(vec_range[0], vec_range[1], size=dim).astype(np.float64)
    k = np.random.uniform(vec_range[0], vec_range[1], size=dim).astype(np.float64)

    # Generate a random displacement vector (delta) of integers in pos_range
    delta = np.random.randint(pos_range[0], pos_range[1] + 1, size=dim).astype(np.float64)

    # Generate two random position vectors for pos1_q and pos2_q
    pos1_q = np.random.randint(pos_range[0], pos_range[1] + 1, size=dim).astype(np.float64)
    pos2_q = np.random.randint(pos_range[0], pos_range[1] + 1, size=dim).astype(np.float64)

    # Compute corresponding pos1_k and pos2_k so that both have the same displacement
    pos1_k = pos1_q + delta
    pos2_k = pos2_q + delta

    return q, k, pos1_q, pos1_k, pos2_q, pos2_k, delta

# Example usage:
q, k, pos1_q, pos1_k, pos2_q, pos2_k, delta_1 = generate_experiment_data()
delta_2 = pos2_k - pos2_q  # Should be identical to delta_1

# Verify that the displacements are indeed identical.
assert np.all(delta_1 == delta_2), "Displacements do not match!"

# --- 4. Calculations ---

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

# --- 5. Verification & Output ---

def format_vector(vec, precision=2):
    """Formats a numpy vector for clean printing with alignment."""
    # Check if all elements are effectively whole numbers
    if np.all(np.isclose(vec, np.round(vec))):
        return f"[{', '.join(f'{x:>4.0f}' for x in vec)}]"
    else:
        return f"[{', '.join(f'{x:>{precision+4}.{precision}f}' for x in vec)}]"

print("--- Experiment Setup ---")
print(f"Query Vector (q):      {format_vector(q)}")
print(f"Key Vector   (k):      {format_vector(k)}\n")

print("--- Position Pair 1 ---")
print(f"  q position (s1):     {format_vector(pos1_q)}")
print(f"  k position (s2):     {format_vector(pos1_k)}")
print(f"  Displacement (s2-s1):{format_vector(delta_1)}\n")

print("--- Position Pair 2 ---")
print(f"  q position (s3):     {format_vector(pos2_q)}")
print(f"  k position (s4):     {format_vector(pos2_k)}")
print(f"  Displacement (s4-s3):{format_vector(delta_2)}\n")

print("--- Verification of RoPE Principle in 4D ---")
print("The inner product should be identical for both pairs due to the same relative displacement.\n")

label_width = 25
val_width = 16
print(f"{'Dot Product (Pair 1):':<{label_width}} {dot_product_1:{val_width}.9f}")
print(f"{'Dot Product (Pair 2):':<{label_width}} {dot_product_2:{val_width}.9f}")
print("-" * (label_width + val_width + 1))
print(f"{'Products are equal?':<{label_width}} {str(np.allclose(dot_product_1, dot_product_2)):>{val_width}}")
print(f"{'Absolute Difference:':<{label_width}} {abs(dot_product_1 - dot_product_2):>{val_width}.2e}")
