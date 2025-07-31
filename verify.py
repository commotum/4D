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





"""
In physics, the speed of light $ c $ (approximately $ 3 \times 10^8 \, \text{m/s} $) 
relates time and space in spacetime calculations. Setting $ c = 1 $ in natural units 
means choosing units where time and space are measured such that the speed of light 
becomes a dimensionless constant equal to 1. For example:

If time is measured in seconds, then distance is measured in light-seconds (the 
distance light travels in one second, about $ 3 \times 10^8 \, \text{m} $).
Alternatively, if distance is measured in meters, time is measured in units of the 
time it takes light to travel one meter (about $ 3.33 \times 10^{-9} \, \text{s} $).
This makes the spacetime interval (or Minkowski norm) simpler, as $ c $ no longer 
appears explicitly in equations.

In the Minkowski metric, the spacetime interval between two events with separation 
$ \Delta p = (\Delta t, \Delta x, \Delta y, \Delta z) $ is typically written (in the 
$+,-,-,-$ signature used here):

$$\Delta s^2 = (c \Delta t)^2 - (\Delta x)^2 - (\Delta y)^2 - (\Delta z)^2$$

With $ c = 1 $, this becomes:
$$\Delta s^2 = (\Delta t)^2 - (\Delta x)^2 - (\Delta y)^2 - (\Delta z)^2$$
This means time and spatial coordinates are treated on equal footing, with no scaling factor $ c $.
Applying $ c = 1 $ to Positions $ p $
The positions $ p $ are 4D spacetime vectors, e.g., $ p = t \gamma_0 + x \gamma_1 + y \gamma_2 + z \gamma_3 $. Normalizing units so $ c = 1 $ means:

The time coordinate $ t $ is expressed in units where the speed of light is 1. For example:

If spatial coordinates $ x, y, z $ are in meters, then $ t $ is in units of $ \text{meter}/c $ (about $ 3.33 \times 10^{-9} \, \text{s} $).
If $ t $ is in seconds, then $ x, y, z $ are in light-seconds.
"""

"""
1. Normalize the units
2. Define the max number of positions for each dimension
3. unit circle/number of positions = 2 * pi / top_delta which is the step size
4. step_size * step index * spacetime unit = position

t = t index * 1/c * step size
x = x index * step size
y = y index * step size
z = z index * step size

Thought for 32sYes, your reasoning and proposed setup make complete sense, and I'll explain why step by step, addressing both the unit equivalence (with $c=1$) and the potential "blowup" issue in the Lorentz rotors (transformations). You're spot-on that dropping the explicit $1/c$ term (after normalizing to $c=1$) leads to time and space being treated in equivalent natural units, where a position value like 3.14 in the time coordinate corresponds to 3.14 seconds (if you choose seconds as the base unit), and the same 3.14 in a spatial coordinate (e.g., x) corresponds to 3.14 light-seconds (a physically large distance in meters). This asymmetry in physical scale is a direct consequence of relativity and natural units, but it's not a problem—it's a feature that respects the speed of light as the universal limit. I'll also cover why integers plugged directly into the rotors can cause blowups (especially for time) and how your max-to-2π scaling naturally mitigates it without needing extra terms.
1. Unit Normalization with $c=1$: Time in Seconds vs. Space in Light-Seconds
When you set $c=1$, you're using natural units where time and space are dimensionally the same, but the physical interpretation depends on your choice of base unit:

If you define the unit for time as seconds, then spatial units must be light-seconds (the distance light travels in one second) to make $c=1$ hold. This is because the Minkowski interval simplifies to $\Delta s^2 = (\Delta t)^2 - (\Delta x)^2 - (\Delta y)^2 - (\Delta z)^2$, with no $c$ factor, so $\Delta t$ and $\Delta x$ must have the same "dimension" (e.g., both in seconds or both in meters).
Conversely, if you define spatial units as meters, then time units become light-meters (the time it takes light to travel one meter, about $3.33 \times 10^{-9}$ seconds).

In your example, assuming seconds as the base for time (a common choice for temporal data), then:

A time position of 3.14 means 3.14 seconds.
A spatial position of 3.14 means 3.14 light-seconds.

Yes, 3.14 light-seconds is indeed a "huge number of meters":

1 light-second ≈ 299,792,458 meters (about 300,000 km, or roughly 7.5 times around Earth's equator).
3.14 light-seconds ≈ 3.14 × 299,792,458 ≈ 9.42 × 10^8 meters (942 million meters, or 942,000 km).
To put this in perspective:
Earth's diameter: ~12,742 km (tiny compared to 942,000 km).
Earth-Moon distance: ~384,000 km ≈ 1.28 light-seconds.
So 3.14 light-seconds ≈ 2.45 Earth-Moon distances.
1 AU (Earth-Sun distance): ~499 light-seconds ≈ 1.5 × 10^11 meters.
Thus, 3.14 light-seconds ≈ 0.0063 AU (small on astronomical scales but large for human or planetary ones).

This "huge" spatial scale makes physical sense because light is fast—covering vast distances in short times. Dropping the $1/c$ term is correct here because $c=1$ absorbs it: In pre-normalized physical units (where $c \approx 3 \times 10^8$ m/s), you'd scale time by multiplying by $c$ (e.g., $t_{normalized} = c \times t_{physical}$) to make it match spatial units in meters, but once normalized, the scaling is implicit in the unit choice. Your setup (position = index × step_size, with step_size = 2π / top_delta) ensures positions max out at ~2π across all dimensions, keeping values bounded and consistent.
If your data involves very different physical scales (e.g., small lab experiments where spatial distances are meters, not light-seconds), you could rescale the spatial coordinates by a factor (e.g., divide by a large number like 10^6) during preprocessing to make the effective units smaller. But for relativistic or large-scale applications (e.g., particle physics or cosmology), the light-second equivalence is ideal.


"""



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