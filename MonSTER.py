"""
MonSTER: Minkowski Space Time Embedding Rotors

MonSTER computes 4-dimensional, Minkowski-metric-respecting relative positional
encodings for transformer attention, built upon the principles of the real Clifford
algebra Cl(1,3) with a (+, –, –, –) metric signature. This version calculates a
unique 4D Lorentz transformation, R_eff, based directly on the relative spacetime
displacement ΔP = (Δt, Δx, Δy, Δz) between query and key elements. This R_eff
is generated block-wise using different frequencies for multi-scale representation.
The resulting R_eff matrices modulate attention scores, typically within a Minkowski
dot product. This ensures that the same displacement ΔP consistently yields the
same geometric transformation influencing attention, crucial for tasks like ARC-AGI.

Key Steps for Computing R_eff_b per block b:
1. Compute raw displacement ΔP_raw = P_key - P_query = (Δt_raw, Δx_raw, Δy_raw, Δz_raw).
   (Input coordinates are raw pixel/step differences for ARC-AGI).
2. Calculate scaled displacements using separate inverse frequencies for time and space:
   Δt_scaled_b = Δt_raw * inv_freq_time_b
   Δs_scaled_b = (Δx_raw * inv_freq_space_b, ...)
3. Determine boost rapidity φ_b = C_t * tanh(Δt_scaled_b / C_t) for stability.
4. Determine spatial rotation angle θ_b = |Δs_scaled_b| and axis u_rot_b.
   If θ_b is near zero, rotation is identity, and u_rot_b defaults (e.g., global z-axis).
5. Build 4x4 block transformations:
   a. M_rot_b: From 3x3 spatial rotation (using θ_b, u_rot_b via Rodrigues' formula).
   b. M_boost_b: Minkowski boost (using φ_b and axis u_boost_b = u_rot_b).
6. Combine: R_eff_b = M_boost_b @ M_rot_b (operationally: rotation first, then boost).
   (Note: These specific rotation and boost operations commute here because their axes are identical.)
7. Modulate Attention: For query/key feature blocks Q_b, K_b, the geometric part
   of the attention score is Σ_b (Q_b^T ⋅ η ⋅ R_eff_b ⋅ K_b).
"""

import jax.numpy as jnp

def build_rotation_matrix(axis, theta):
    """
    Rodrigues' formula for 3x3 rotation about 'axis' by angle 'theta'.
    Handles broadcasting for batched axes (..., B, 3) and angles (..., B).
    """
    theta_exp = theta[..., None]  # Shape: (..., B, 1)
    cos_t = jnp.cos(theta_exp)
    sin_t = jnp.sin(theta_exp)

    uuT = jnp.einsum('...bi,...bj->...bij', axis, axis)

    zeros = jnp.zeros_like(axis[..., 0])
    u_cross = jnp.stack([
        zeros, -axis[..., 2], axis[..., 1],
        axis[..., 2], zeros, -axis[..., 0],
        -axis[..., 1], axis[..., 0], zeros
    ], axis=-1).reshape((*axis.shape[:-2], axis.shape[-2], 3, 3))

    I3 = jnp.eye(3, dtype=axis.dtype)
    cos_t_exp_mat = cos_t[..., None]
    sin_t_exp_mat = sin_t[..., None]

    return (cos_t_exp_mat * I3 +
            (1 - cos_t_exp_mat) * uuT +
            sin_t_exp_mat * u_cross)

def get_delta_monster(
    delta_t_raw,
    delta_coords_raw,
    num_blocks: int,
    base_time: float = 10000.,
    base_space: float = 10000.,
    time_rapidity_scale_C_t: float = 2.0,
    epsilon: float = 1e-8,
    dtype=jnp.float32
):
    """
    Computes block-diagonal relative spacetime Lorentz transformation matrices
    (R_eff_b) from raw spacetime displacements (Δt_raw, Δcoords_raw).
    Designed for ARC-AGI: uses raw pixel/step differences for deltas, tamps temporal
    rapidity with tanh, and uses distinct frequency bases for time and space.

    Args:
        delta_t_raw: Raw temporal displacement (Δt). Shape can be (...).
        delta_coords_raw: Raw spatial displacement (Δx, Δy, Δz). Shape (..., 3).
        num_blocks: Number of frequency blocks (B).
        base_time: Base for temporal inverse frequencies.
        base_space: Base for spatial inverse frequencies.
        time_rapidity_scale_C_t: Saturation scale C_t for tanh on temporal rapidity.
        epsilon: Small value for numerical stability.
        dtype: Data type for calculations.

    Returns:
        R_eff_blocks: Stack of 4x4 Lorentz transformation matrices.
                      Shape (..., num_blocks, 4, 4).
    """
    delta_t_raw = jnp.asarray(delta_t_raw, dtype=dtype)
    delta_coords_raw = jnp.asarray(delta_coords_raw, dtype=dtype)

    freqs = jnp.arange(num_blocks, dtype=dtype)
    inv_freq_time = 1.0 / (base_time ** (freqs / num_blocks))
    inv_freq_space = 1.0 / (base_space ** (freqs / num_blocks))

    delta_t_scaled = jnp.einsum('...,b->...b', delta_t_raw, inv_freq_time)
    delta_s_scaled = jnp.einsum('...i,b->...bi', delta_coords_raw, inv_freq_space)

    theta_b = jnp.linalg.norm(delta_s_scaled, axis=-1, ord=2)

    default_spatial_axis = jnp.array([0., 0., 1.], dtype=dtype)
    default_axis_bc = jnp.broadcast_to(default_spatial_axis, delta_s_scaled.shape)
    
    is_zero_spatial_delta = theta_b < epsilon
    axis_u_rot_b = jnp.where(
        is_zero_spatial_delta[..., None],
        default_axis_bc,
        delta_s_scaled / jnp.maximum(theta_b[..., None], epsilon)
    )

    R3_b = build_rotation_matrix(axis_u_rot_b, theta_b)
    
    pref_B_shape = R3_b.shape[:-2]
    M_rot_b = jnp.zeros((*pref_B_shape, 4, 4), dtype=dtype)
    M_rot_b = M_rot_b.at[..., 0, 0].set(1.0)
    M_rot_b = M_rot_b.at[..., 1:, 1:].set(R3_b)

    phi_b_prescale = delta_t_scaled
    safe_C_t = jnp.maximum(time_rapidity_scale_C_t, epsilon) # ensure C_t is not zero for division
    phi_b = safe_C_t * jnp.tanh(phi_b_prescale / safe_C_t)

    ch_b = jnp.cosh(phi_b)
    sh_b = jnp.sinh(phi_b)
    axis_u_boost_b = axis_u_rot_b

    M_boost_b = jnp.zeros((*pref_B_shape, 4, 4), dtype=dtype)
    M_boost_b = M_boost_b.at[..., 0, 0].set(ch_b)
    M_boost_b = M_boost_b.at[..., 0, 1:].set(-axis_u_boost_b * sh_b[..., None])
    M_boost_b = M_boost_b.at[..., 1:, 0].set(-axis_u_boost_b * sh_b[..., None])

    eye3 = jnp.eye(3, dtype=dtype)
    uuT_boost_b = jnp.einsum('...bi,...bj->...bij', axis_u_boost_b, axis_u_boost_b)
    ch_b_minus_1_exp = (ch_b - 1.0)[..., None, None]
    
    M_boost_b = M_boost_b.at[..., 1:, 1:].set(eye3 + ch_b_minus_1_exp * uuT_boost_b)

    # R_eff_b = M_boost_b @ M_rot_b.
    # Operationally, this applies M_rot_b first, then M_boost_b.
    # Since axis_u_boost_b == axis_u_rot_b, these operations commute,
    # so M_rot_b @ M_boost_b would yield the same R_eff_b.
    R_eff_blocks = jnp.einsum("...bij,...bjk->...bik", M_boost_b, M_rot_b)

    return R_eff_blocks