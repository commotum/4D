# 1. Unit-Normalization (Lattice Units)
#      - Set the speed of light c=1 for simplicity, so temporal units 
#        (seconds) correspond directly to spatial units (light-seconds)
#      - Calculate the uniform unit step coefficient tau (τ): 
#        τ = 2π / sequence length
#      - Convert the input 4D position P = [t, x, y, z] 
#        to dimensionless lattice coordinates
#        Q = [τ*t, τ*x, τ*y, τ*z]

# 2. Frequency Scaling 
#      - ??

# 3. Compute Boost Rapidity
#      - ??


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