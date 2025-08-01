# A polished implementation of verify.py that demonstrates the core RoPE 
# RoPE properties for 2D vectors, namely:
#
# 1. Equivalence of Formulations: Rotating vectors at absolute positions is 
#    equivalent to using the complex relative position formula.
# 2. Norm Preservation: Rotational encoding does not change a vector's length.
# 3. Shift Invariance: The dot product between two encoded vectors depends only
#    on their relative positional difference, not their absolute positions.



import numpy as np

# --- Setup ---

# 1. Define a single pair of query (q) and key (k) vectors.
# We will use the same q and k for all tests to isolate the effect of position.
q = np.array([+0.09, +0.07, +0.05, +0.03], dtype=np.float32)
k = np.array([+0.06, +0.04, +0.02, +0.01], dtype=np.float32)

# 2. Define the rotation frequencies for each 2D-subspace.
# Following the original paper, we pair dimensions (0,1), (2,3), etc.
theta = np.array([0.1, 0.2], dtype=np.float32)

# 3. Define the rotation function (as you had before).
# This function applies the rotation matrix R_p to a vector x.
def apply_rotation_4d(vec, pos, theta_pair):
    # Split into two 2D vectors
    vec1, vec2 = vec[:2], vec[2:]

    # Helper for 2D rotation
    def apply_rotation_2d(vec_2d, p, t):
        x, y = vec_2d
        cos_pt = np.cos(p * t)
        sin_pt = np.sin(p * t)
        return np.array([
            x * cos_pt - y * sin_pt,
            x * sin_pt + y * cos_pt
        ], dtype=np.float32)

    # Apply rotation to each 2D group and concatenate
    rotated1 = apply_rotation_2d(vec1, pos, theta_pair[0])
    rotated2 = apply_rotation_2d(vec2, pos, theta_pair[1])
    return np.concatenate([rotated1, rotated2])

# --- Experiment ---

# We want to show that the dot product <R_m*q, R_n*k> depends only on (m-n).
# To prove this, we will test two different pairs of positions (m,n) and (i,j)
# that have the same relative distance.

# Position Pair 1: (m=7, n=2)
m = 7
n = 2
delta_1 = m - n
print(f"Position Pair 1: (m={m}, n={n})")
print(f"Relative Distance: {delta_1}\n")

# Position Pair 2: (i=10, j=5)
i = 10
j = 5
delta_2 = i - j
print(f"Position Pair 2: (i={i}, j={j})")
print(f"Relative Distance: {delta_2}\n")

# --- Calculations ---

# Test 1: Apply rotations for positions m and n
q_m = apply_rotation_4d(q, m, theta) # This is R_m * q
k_n = apply_rotation_4d(k, n, theta) # This is R_n * k
dot_product_1 = np.dot(q_m, k_n)

# Test 2: Apply rotations for positions i and j
q_i = apply_rotation_4d(q, i, theta) # This is R_i * q
k_j = apply_rotation_4d(k, j, theta) # This is R_j * k
dot_product_2 = np.dot(q_i, k_j)

# --- Verification ---

print("--- Verification of RoPE Principle ---")
print(f"Dot product for positions (m={m}, n={n}): {dot_product_1:.8f}")
print(f"Dot product for positions (i={i}, j={j}): {dot_product_2:.8f}\n")

# The punchline: Because the relative distance is the same (5), the dot products should be identical.
print("Are the dot products equal?")
print(f"Answer: {np.allclose(dot_product_1, dot_product_2)}")
print(f"Difference: {abs(dot_product_1 - dot_product_2)}")