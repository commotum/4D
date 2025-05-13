# test.py
import jax.numpy as jnp
import numpy as np
from MonSTER import get_delta_monster # Assuming MonSTER.py contains get_delta_monster
from mink import minkowski_dot       # Assuming mink.py contains minkowski_dot

# Set random seed for reproducibility
np.random.seed(42)

# Generate two random 16d vectors (feature embeddings Q_feat, K_feat)
vec_dim = 16 
num_blocks = vec_dim // 4 # Each block is 4D
# Using small values for features, more typical for normalized embeddings
q_feat_np = np.random.uniform(-1.0, 1.0, vec_dim) 
k_feat_np = np.random.uniform(-1.0, 1.0, vec_dim)

q_feat_jax = jnp.array(q_feat_np)
k_feat_jax = jnp.array(k_feat_np)

print("Initial feature vector shapes:")
print(f"q_feat_jax shape: {q_feat_jax.shape}")
print(f"k_feat_jax shape: {k_feat_jax.shape}")

# Calculate original dot products for reference
original_std_dot = jnp.dot(q_feat_jax, k_feat_jax)
original_mink_dot = minkowski_dot(q_feat_jax, k_feat_jax) # From mink.py

print(f"\nOriginal Standard Dot: {original_std_dot:+.4f}")
print(f"Original Minkowski Dot: {original_mink_dot:+.4f}")

# Define a sample spacetime displacement (Δt, Δx, Δy, Δz)
# These are raw differences, e.g., from an ARC-AGI grid
delta_t = jnp.array(1.0)              # e.g., 1 time step difference
delta_coords = jnp.array([3.0, -4.0, 0.0]) # e.g., 3 pixels right, 4 pixels up

print("\nSpacetime displacement (ΔP):")
print(f"Δt: {delta_t}, Δcoords: {delta_coords}")

# Get the MonSTER relative transformation matrices
R_eff_blocks = get_delta_monster(
    delta_t_raw=delta_t,
    delta_coords_raw=delta_coords,
    num_blocks=num_blocks,
    base_time=10000.,
    base_space=10000.,
    time_rapidity_scale_C_t=2.0, # C_t hyperparameter for tanh
    dtype=q_feat_jax.dtype
)

print(f"\nShape of R_eff_blocks: {R_eff_blocks.shape}") # Expected (num_blocks, 4, 4)

# 1. Verify R_eff_blocks are Lorentz transformations
# Minkowski metric η as a matrix diag(1, -1, -1, -1)
eta_matrix = jnp.diag(jnp.array([1., -1., -1., -1.], dtype=q_feat_jax.dtype))
all_lorentz_verified = True
print("\nVerifying R_eff_b blocks are Lorentz transformations (R^T η R = η):")
for i in range(num_blocks):
    R_b = R_eff_blocks[i] # For a single batch/pair, R_eff_blocks might be (num_blocks, 4, 4)
                          # If delta_t/delta_coords had batch dims, R_eff_blocks would be (..., num_blocks, 4, 4)
                          # Assuming delta_t/delta_coords are simple scalars/vectors for this test
    
    # R_b should have shape (4,4) if no batch dims in delta_P
    if R_eff_blocks.ndim == 3: # (num_blocks, 4, 4)
        R_b_current = R_eff_blocks[i]
    elif R_eff_blocks.ndim > 3: # e.g. (batch_dims..., num_blocks, 4, 4)
        # For simplicity in this test, let's assume no extra batch dims in delta_P input
        # or handle how to pick one R_b from a batch if needed.
        # For now, this test is designed for scalar delta_t and vector delta_coords.
        print("Note: R_eff_blocks has batch dimensions. Verification will use the first element.")
        R_b_current = R_eff_blocks[tuple(0 for _ in range(R_eff_blocks.ndim - 3)) + (i, slice(None), slice(None))]


    check_matrix = R_b_current.T @ eta_matrix @ R_b_current
    if not jnp.allclose(check_matrix, eta_matrix, atol=1e-5):
        all_lorentz_verified = False
        print(f"Block {i} is NOT a Lorentz transformation!")
        # print("R_b.T @ eta @ R_b gives:\n", check_matrix) # Can be verbose
        # print("Expected eta:\n", eta_matrix)
        break
if all_lorentz_verified:
    print("All R_eff_b blocks are verified Lorentz transformations.")

# 2. Calculate the MonSTER-modulated Minkowski interaction
# This is Σ_b Q_b^T η R_eff_b K_b
monster_interaction_score = 0.0
q_feat_blocks_reshaped = q_feat_jax.reshape(num_blocks, 4)
k_feat_blocks_reshaped = k_feat_jax.reshape(num_blocks, 4)

for i in range(num_blocks):
    q_b = q_feat_blocks_reshaped[i]        # Shape (4,)
    k_b = k_feat_blocks_reshaped[i]        # Shape (4,)

    if R_eff_blocks.ndim == 3: # (num_blocks, 4, 4)
        R_eff_b = R_eff_blocks[i]
    else: # (batch_dims..., num_blocks, 4, 4) - take first batch element for test
        R_eff_b = R_eff_blocks[tuple(0 for _ in range(R_eff_blocks.ndim - 3)) + (i, slice(None), slice(None))]
            
    # term = q_b^T @ η @ R_eff_b @ k_b
    term = q_b @ eta_matrix @ R_eff_b @ k_b
    monster_interaction_score += term

print(f"\nMonSTER-Modulated Minkowski Interaction Score: {monster_interaction_score:+.4f}")
print(f"Difference from original Minkowski Dot: {abs(original_mink_dot - monster_interaction_score):.4f} (Note: This is expected to be different)")

# Format and print original vectors
def format_vector(vec):
    vec_np = np.array(vec) 
    return "[" + ", ".join(f"{x:+.2f}" for x in vec_np) + "]"

print("\nOriginal Feature Vectors:")
print(f"Q_feat: {format_vector(q_feat_np)}")
print(f"K_feat: {format_vector(k_feat_np)}")

print(f"\nThis test demonstrates generating R_eff_b from a ΔP and using it in a Minkowski-style interaction.")
print(f"The key verification is that R_eff_b matrices are valid Lorentz transformations.")
