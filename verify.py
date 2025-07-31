import numpy as np

# Input vectors
a = np.array([+0.09, +0.07, +0.05, +0.03], dtype=np.float32)
b = np.array([+0.06, +0.04, +0.02, +0.01], dtype=np.float32)
c = np.array([-0.01, -0.02, -0.03, -0.04], dtype=np.float32)
d = np.array([-0.05, -0.06, -0.07, -0.08], dtype=np.float32)

# Precomputed dot products (as provided)
aa = np.dot(a, a)
ab = np.dot(a, b)
ac = np.dot(a, c)
ad = np.dot(a, d)
bb = np.dot(b, b)
bc = np.dot(b, c)
bd = np.dot(b, d)
cc = np.dot(c, c)
cd = np.dot(c, d)
dd = np.dot(d, d)

# Positions
m = 2
n = 7
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
delta_mn = m - n  # 2 - 7 = -5
delta_ij = i - j  # 10 - 5 = 5
complex_dot_mn = complex_dot_product_4d(a, b, delta_mn, theta)
complex_dot_ij = complex_dot_product_4d(c, d, delta_ij, theta)

# Print results to compare
print(f"Dot product <q_m e^(im theta), k_n e^(in theta)> (m=2, n=7): {dot_mn:.6f}")
print(f"Complex formulation Re[q_m k_n* e^(i(m-n)theta)]: {complex_dot_mn:.6f}")
print(f"Dot product <q_i e^(i i theta), k_j e^(i j theta)> (i=10, j=5): {dot_ij:.6f}")
print(f"Complex formulation Re[q_i k_j* e^(i(i-j)theta)]: {complex_dot_ij:.6f}")

# Check if relative position dot products are equal for same |m-n|
# Since |m-n| = |-5| = 5 and |i-j| = |5| = 5, results should be similar
print(f"\nComparing relative positions |m-n| = |{delta_mn}| and |i-j| = |{delta_ij}|:")
print(f"Dot product difference: {abs(dot_mn - dot_ij):.6f}")
print(f"Complex dot difference: {abs(complex_dot_mn - complex_dot_ij):.6f}")





# 1.  **Unit-Normalization (Lattice Units)**
# 2.  **Frequency Scaling** 
# 3.  **Compute Boost Rapidity**
# 


# Set the speed of light c = 1 for simplicity in the spacetime lattice.
# This means 1 temporal step (in seconds) corresponds to 1 spatial step (in light-seconds),
# where 1 light-second is equivalent to 299,792,458 meters.

import numpy as np  # Import NumPy for mathematical operations

# Define the spans for each dimension, analogous to sequence length in standard RoPE
# Temporal span: Max time steps
t_span = 1024 
# Spatial spans: Max positions for tokens in each spatial dimension (x, y, z)
x_span = 1024
y_span = 1024
z_span = 1024

# Determine the maximum span across all dimensions to normalize the unit step
top_delta = max(t_span, x_span, y_span, z_span)

# Define the full unit circle in radians
unit_circle = 2 * np.pi

# Calculate the unit step size by dividing the unit circle by the maximum span,
# similar to how angular steps are computed in RoPE for positional encoding
unit_step = unit_circle / top_delta


# Positions
STA_1 = np.array([25, 25, 15, 23], dtype=np.float32)
STA_2 = np.array([27, 20, 7, 15], dtype=np.float32)
STA_3 = np.array([28, 20, 9, 15], dtype=np.float32)
STA_4 = np.array([30, 15, 1, 7], dtype=np.float32)