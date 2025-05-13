"""
MonSTER: Minkowski Spacetime TransformERs (Relative Encoding)

MonSTER introduces a 4-dimensional, Minkowski-norm-respecting relative positional
encoding method designed for transformer attention, built upon the principles of
the real Clifford algebra Cl(1,3) with a (+, –, –, –) metric signature.
Unlike traditional Rotary Positional Embeddings (RoPE) that effectively encode
relative 1D sequence positions, or earlier MonSTER concepts focusing on absolute
positional transformations, this version of MonSTER computes a unique 4D Lorentz
transformation, $R_{eff}$, based directly on the *relative spacetime displacement*
$\Delta P = (\Delta t, \Delta x, \Delta y, \Delta z)$ between a query and a key element.
This $R_{eff}$ is constructed block-wise using different frequencies and then used to
modulate the attention score, typically within a Minkowski dot product, ensuring
that the same displacement $\Delta P$ always results in the same geometric
transformation $R_{eff}$ influencing the attention.

Key Steps for Computing $R_{eff,b}$ for each block $b$:
1. For a query at $P_Q$ and a key at $P_K$, compute the raw displacement
   $\Delta P_{raw} = P_K - P_Q = (\Delta t_{raw}, \Delta x_{raw}, \Delta y_{raw}, \Delta z_{raw})$.
   (For ARC-AGI, $\Delta x, \Delta y, \Delta z$ are raw pixel differences; $\Delta t$ is raw step difference).
2. For each embedding block $b$, compute scaled displacement components using
   separate inverse frequencies for time and space:
   $\Delta t_{scaled, b} = \Delta t_{raw} \cdot \text{invf}_{time, b}$
   $\vec{\Delta s}_{scaled, b} = (\Delta x_{raw} \cdot \text{invf}_{space, b}, \Delta y_{raw} \cdot \text{invf}_{space, b}, \Delta z_{raw} \cdot \text{invf}_{space, b})$
3. Boost Parameters: Calculate block-specific rapidity $\phi_b = C_t \cdot \tanh(\Delta t_{scaled, b} / C_t)$
   (e.g., $C_t=2.0$) to ensure stability.
4. Rotation Parameters: Calculate block-specific angle $\theta_b = |\vec{\Delta s}_{scaled, b}|$ and
   axis $\vec{u}_{rot, b} = \vec{\Delta s}_{scaled, b} / \theta_b$. If $\theta_b \approx 0$, $M_{rot,b}$ becomes
   Identity and $\vec{u}_{rot, b}$ is set to a default (e.g., global z-axis).
5. Build Transformations:
   a. Construct a 3x3 spatial rotation $R_{3,b}$ from $\theta_b, \vec{u}_{rot,b}$ (via Rodrigues' formula)
      and embed it into a 4x4 matrix $M_{rot,b}$.
   b. Construct a 4x4 Minkowski boost $M_{boost,b}$ using rapidity $\phi_b$ and spatial
      axis $\vec{u}_{boost,b} = \vec{u}_{rot,b}$ (using the same axis, or default if no spatial displacement).
6. Combine: Form the block-specific relative Lorentz transformation
   $R_{eff,b} = M_{boost,b} \cdot M_{rot,b}$ (convention: boost first, then rotation).
7. Application in Attention: The set of $R_{eff,b}$ matrices modulates attention scores.
   If query/key feature embeddings ($Q_{feat}, K_{feat}$) are split into 4-vector
   blocks $(Q_1, ..., Q_N)$ and $(K_1, ..., K_N)$, the geometric part of the
   attention score is $\sum_{b} Q_b^T \eta R_{eff,b} K_b$.
"""

import jax.numpy as jnp

# build_rotation_matrix remains a crucial utility
def build_rotation_matrix(axis, theta):
    """
    Rodrigues' formula for 3x3 rotation about 'axis' by angle 'theta'.
    Correctly handles broadcasting for batched axes and angles.

    Args:
        axis: (..., B, 3) unit vectors for B blocks.
        theta: (..., B) angles for B blocks.

    Returns:
        (..., B, 3, 3) rotation matrices.
    """
    theta_exp = theta[..., None]  # Shape: (..., B, 1)

    cos_t = jnp.cos(theta_exp)    # Shape: (..., B, 1)
    sin_t = jnp.sin(theta_exp)    # Shape: (..., B, 1)

    uuT = jnp.einsum('...bi,...bj->...bij', axis, axis)  # Shape: (..., B, 3, 3)

    zeros = jnp.zeros_like(axis[..., 0])  # Shape: (..., B)
    u_cross = jnp.stack([
        zeros, -axis[..., 2], axis[..., 1],
        axis[..., 2], zeros, -axis[..., 0],
        -axis[..., 1], axis[..., 0], zeros
    ], axis=-1).reshape((*axis.shape[:-2], axis.shape[-2], 3, 3)) # Shape: (..., B, 3, 3)

    I3 = jnp.eye(3, dtype=axis.dtype)  # Shape: (3, 3)

    # Expand cos_t and sin_t for broadcasting with I3, uuT, u_cross
    cos_t_exp_mat = cos_t[..., None]  # Shape: (..., B, 1, 1)
    sin_t_exp_mat = sin_t[..., None]  # Shape: (..., B, 1, 1)

    return (cos_t_exp_mat * I3 +
            (1 - cos_t_exp_mat) * uuT +
            sin_t_exp_mat * u_cross)

def compute_monster_relative_transform(
    delta_t_raw,
    delta_coords_raw,
    num_blocks: int,
    base_time: float = 10000.,
    base_space: float = 10000.,
    time_rapidity_scale: float = 2.0, # C_t for tanh saturation
    epsilon: float = 1e-8,
    dtype=jnp.float32
):
    """
    Computes block-diagonal relative spacetime Lorentz transformations R_eff,b
    based on raw spacetime displacements (delta_t_raw, delta_coords_raw).
    These R_eff,b matrices are designed to be used in an attention mechanism, e.g.,
    Score = sum_b Q_b^T @ eta @ R_eff_b @ K_b.

    Args:
        delta_t_raw: Raw temporal displacement. Shape can be (...)
        delta_coords_raw: Raw spatial displacement (dx, dy, dz). Shape (..., 3)
        num_blocks: Number of frequency blocks (B).
        base_time: Base for temporal inverse frequencies.
        base_space: Base for spatial inverse frequencies.
        time_rapidity_scale: Scaling factor C_t for tanh on scaled delta_t.
        epsilon: Small value for numerical stability.
        dtype: Data type for calculations.

    Returns:
        R_eff_blocks: Stack of 4x4 Lorentz transformation matrices.
                      Shape (..., num_blocks, 4, 4), where ... are leading dims
                      from delta_t_raw / delta_coords_raw.
    """
    delta_t_raw = jnp.asarray(delta_t_raw, dtype=dtype)
    delta_coords_raw = jnp.asarray(delta_coords_raw, dtype=dtype)

    # --- Inverse Frequencies ---
    # freqs shape: (num_blocks,)
    freqs = jnp.arange(num_blocks, dtype=dtype)
    inv_freq_time = 1.0 / (base_time ** (freqs / num_blocks))
    inv_freq_space = 1.0 / (base_space ** (freqs / num_blocks))

    # --- Scaled Displacements ---
    # delta_t_raw shape: (...)
    # delta_coords_raw shape: (..., 3)
    # inv_freq_time/space shape: (num_blocks,)
    # We want delta_t_scaled: (..., num_blocks)
    # We want delta_s_scaled: (..., num_blocks, 3)
    delta_t_scaled = jnp.einsum('...,b->...b', delta_t_raw, inv_freq_time)
    delta_s_scaled = jnp.einsum('...i,b->...bi', delta_coords_raw, inv_freq_space)

    # --- Spatial Rotation Part ---
    # theta_b shape: (..., num_blocks)
    theta_b = jnp.linalg.norm(delta_s_scaled, axis=-1, ord=2) # Magnitude of spatial displacement

    # Default axis (e.g., z-axis [0,0,1]) for zero spatial delta
    # This axis choice mainly matters if there's a boost but no spatial rotation.
    default_spatial_axis = jnp.array([0., 0., 1.], dtype=dtype)
    # Broadcast default_spatial_axis to shape of delta_s_scaled (..., num_blocks, 3)
    # Create shape like (1,1,...,1, num_blocks, 3) if delta_s_scaled has leading dims
    ones_for_broadcast = tuple([1] * (delta_s_scaled.ndim - 2)) # e.g., if delta_s is (N,M,B,3) -> (1,1)
    default_axis_bc = jnp.broadcast_to(default_spatial_axis.reshape(*ones_for_broadcast, 1, 3), delta_s_scaled.shape)

    # axis_u_rot_b shape: (..., num_blocks, 3)
    # Normalize, using default axis if magnitude is near zero
    is_zero_spatial_delta = theta_b < epsilon
    axis_u_rot_b = jnp.where(
        is_zero_spatial_delta[..., None], # Expand for broadcasting with 3-vector axis
        default_axis_bc,
        delta_s_scaled / jnp.maximum(theta_b[..., None], epsilon)
    )

    # R3_b shape: (..., num_blocks, 3, 3)
    R3_b = build_rotation_matrix(axis_u_rot_b, theta_b) # theta_b is (..., B)

    # Embed into M_rot_b: (..., num_blocks, 4, 4)
    # Prefix shape for M_rot_b, e.g. (..., num_blocks)
    pref_B_shape = R3_b.shape[:-2]
    M_rot_b = jnp.zeros((*pref_B_shape, 4, 4), dtype=dtype)
    M_rot_b = M_rot_b.at[..., 0, 0].set(1.0)
    M_rot_b = M_rot_b.at[..., 1:, 1:].set(R3_b)

    # --- Minkowski Boost Part ---
    # Rapidity phi_b shape: (..., num_blocks)
    phi_b_prescale = delta_t_scaled
    phi_b = time_rapidity_scale * jnp.tanh(phi_b_prescale / jnp.maximum(time_rapidity_scale, epsilon))

    ch_b = jnp.cosh(phi_b) # (..., num_blocks)
    sh_b = jnp.sinh(phi_b) # (..., num_blocks)

    # Boost axis is the same as rotation axis: (..., num_blocks, 3)
    axis_u_boost_b = axis_u_rot_b

    # M_boost_b shape: (..., num_blocks, 4, 4)
    M_boost_b = jnp.zeros((*pref_B_shape, 4, 4), dtype=dtype)
    M_boost_b = M_boost_b.at[..., 0, 0].set(ch_b)
    M_boost_b = M_boost_b.at[..., 0, 1:].set(-axis_u_boost_b * sh_b[..., None])
    M_boost_b = M_boost_b.at[..., 1:, 0].set(-axis_u_boost_b * sh_b[..., None])

    eye3 = jnp.eye(3, dtype=dtype)
    uuT_boost_b = jnp.einsum('...bi,...bj->...bij', axis_u_boost_b, axis_u_boost_b) # (..., num_blocks, 3, 3)
    ch_b_minus_1_exp = (ch_b - 1.0)[..., None, None] # (..., num_blocks, 1, 1)

    M_boost_b = M_boost_b.at[..., 1:, 1:].set(eye3 + ch_b_minus_1_exp * uuT_boost_b)

    # --- Combine: R_eff_b = M_boost_b @ M_rot_b ---
    # (Order: boost transformation applied first, then rotation to the result of boost)
    # This matches original MonSTER paper's R4 = M_boost @ M_rot.
    # R_eff_blocks shape: (..., num_blocks, 4, 4)
    R_eff_blocks = jnp.einsum("...bij,...bjk->...bik", M_boost_b, M_rot_b)

    return R_eff_blocks

# Example Usage (conceptual, actual use is inside attention layer)
# if __name__ == '__main__':
#     # Mock data for a batch of 2, sequence length 5, vs sequence length 7
#     # We'd compute delta_P for each pair, so delta_P could be (batch, seq_q, seq_k)
#     # For simplicity, let's make delta_P for one pair:
#     delta_t = jnp.array(1.0)  # e.g., t_key - t_query
#     delta_coords = jnp.array([3.0, -4.0, 0.0]) # e.g., coords_key - coords_query

#     # Typically, these would be arrays with leading batch/sequence dimensions
#     # For example, if we have Q_pos (N, D_pos) and K_pos (M, D_pos)
#     # delta_t = K_pos[:, None, 0] - Q_pos[None, :, 0] # (M, N)
#     # delta_coords = K_pos[:, None, 1:] - Q_pos[None, :, 1:] # (M, N, 3)

#     num_freq_blocks = 8 # Example, d_model_head // 4

#     # Assume these are precomputed based on num_freq_blocks
#     # For the function call, we pass num_blocks and the base values
    
#     R_eff_example_blocks = compute_monster_relative_transform(
#         delta_t_raw=delta_t,
#         delta_coords_raw=delta_coords,
#         num_blocks=num_freq_blocks,
#         base_time=10000.,
#         base_space=10000.,
#         time_rapidity_scale=2.0
#     )
#     print("Shape of R_eff_blocks:", R_eff_example_blocks.shape)
#     # Expected: (num_freq_blocks, 4, 4) if delta_t/coords are scalars/single vector
#     # Or (batch_dims..., num_freq_blocks, 4, 4)


# Commenting out the old clifford_rope function as it serves a different purpose (absolute PE)
# def clifford_rope(x, coords, times, base: float = 10000.):
#     """
#     Clifford RoPE (MonSTER): real Cl(1,3) rotors on 4D blocks of Q/K.
#     Uses standard convention for Lorentz boost matching (+,---) signature.

#     Args:
#         x     : (..., d) embeddings, d must be multiple of 4.
#         coords: (..., 3) spatial positions (x, y, z).
#         times : (...,)   temporal positions (t). Should broadcast with coords.
#         base  : Controls frequency range, similar to RoPE.

#     Returns:
#         (..., d) embeddings with positional information applied.
#     """
#     *pref, d = x.shape
#     if d % 4 != 0:
#         raise ValueError(f"Embedding dimension d ({d}) must be a multiple of 4.")

#     B = d // 4
#     xb = rearrange(x, "... (b k) -> ... b k", b=B, k=4)

#     freqs = jnp.arange(B, dtype=x.dtype)
#     invf = 1.0 / (base ** (freqs / B)) 

#     coords_expanded = jnp.broadcast_to(coords, (*pref, B, 3))
#     theta_vec = coords_expanded * invf[..., None] # invf needs broadcasting
#     theta = jnp.linalg.norm(theta_vec, axis=-1, keepdims=True)
#     axis_u = theta_vec / jnp.maximum(theta, 1e-8) 

#     R3 = build_rotation_matrix(axis_u, theta) # theta needs to be (...,B) or (...,B,1)
#     # If theta is (...,B,1), then build_rotation_matrix handles it.
#     # The original build_rotation_matrix expected theta (...,1) and axis (...,3)
#     # The updated one handles axis (...,B,3) and theta (...,B).
#     # So if theta_vec is (...,B,3) and invf is (...,B), then theta_vec * invf is not direct.
#     # Original: theta_vec = coords_expanded * invf_expanded_for_coords
#     # invf_exp = invf.reshape(*([1]*coords.ndim), B) # This was complex.
#     # My new einsum for scaled coords is cleaner.
    
#     # This whole section is part of the OLD absolute PE method
#     # M_rot = jnp.zeros((*R3.shape[:-2], 4, 4), dtype=x.dtype)
#     # M_rot = M_rot.at[..., 0, 0].set(1.0)
#     # M_rot = M_rot.at[..., 1:, 1:].set(R3)

#     # times_expanded = jnp.broadcast_to(times, (*pref, B))
#     # phi = times_expanded * invf
#     # ch = jnp.cosh(phi)
#     # sh = jnp.sinh(phi)

#     # M_boost = jnp.zeros((*axis_u.shape[:-1], 4, 4), dtype=x.dtype)
#     # M_boost = M_boost.at[..., 0, 0].set(ch)
#     # M_boost = M_boost.at[..., 0, 1:].set(-axis_u * sh[..., None]) 
#     # M_boost = M_boost.at[..., 1:, 0].set(-axis_u * sh[..., None]) 

#     # eye3 = jnp.eye(3, dtype=x.dtype)
#     # uuT_boost = jnp.einsum('...bi,...bj->...bij', axis_u, axis_u) # Corrected for B blocks
#     # ch_minus_1 = ch - 1
#     # M_boost = M_boost.at[..., 1:, 1:].set(eye3 + ch_minus_1[..., None, None] * uuT_boost)

#     # R4 = jnp.einsum("...bij,...bjk->...bik", M_boost, M_rot) # For block-wise
#     # yb = jnp.einsum("...bij,...bj->...bi", R4, xb) # Applying to each block
    
#     # return rearrange(yb, "... b k -> ... (b k)")