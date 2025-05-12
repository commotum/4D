"""
MonSTER: Minkowski Space Time Embedding Rotors:

MonSTER introduces a 4-dimensional, norm-preserving positional embedding method
designed for transformer attention, built upon the real Clifford algebra Cl(1,3)
with Minkowski signature (+, –, –, –). Unlike traditional Rotary Positional
Embeddings (RoPE), which encode absolute one-dimensional sequence positions
using block-diagonal 2×2 complex rotations on embedding dimension pairs, MonSTER
encodes absolute four-dimensional spacetime positions, or events (t, x, y, z),
using block-diagonal 4×4 spacetime rotors on embedding-dimension quartets.

1. Compute per-block inverse frequencies for spatial and temporal modulation.
2. Build a 3×3 spatial rotation via Rodrigues’ formula and embed it in a 4×4 block.
3. Assemble a 4×4 Minkowski boost along the same axis using cosh/sinh.
4. Multiply boost × rotation to get a single Cl(1,3) rotor R∈Spin(1,3).
5. Apply R to each 4-vector sub-block, preserving the Minkowski norm and encoding both
   time-like “before vs after” and spatial orientation in one real-valued transform.
"""

import jax.numpy as jnp
from einops import rearrange

def build_rotation_matrix(axis, theta):
    """
    Rodrigues' formula for 3×3 rotation about 'axis' by angle theta.
    axis: (...,3) unit vectors, theta: (...,1) angles.
    """
    if theta.shape[-1] != 1:
        theta = theta[..., None]

    cos_t = jnp.cos(theta)[...,0]
    sin_t = jnp.sin(theta)[...,0]
    ux, uy, uz = axis[...,0], axis[...,1], axis[...,2]

    uuT = jnp.einsum('...i,...j->...ij', axis, axis)

    zeros = jnp.zeros_like(ux)
    u_cross = jnp.stack([
        zeros,  -uz,   uy,
           uz, zeros,  -ux,
          -uy,   ux, zeros
    ], axis=-1).reshape((*axis.shape[:-1], 3, 3))

    I3 = jnp.eye(3, dtype=axis.dtype)

    cos_t_exp = cos_t[..., None, None]
    sin_t_exp = sin_t[..., None, None]

    return (cos_t_exp * I3 +
            (1 - cos_t_exp) * uuT +
            sin_t_exp * u_cross)

def clifford_rope(x, coords, times, base: float = 10000.):
    """
    Clifford RoPE (MonSTER): real Cl(1,3) rotors on 4D blocks of Q/K.
    Uses standard convention for Lorentz boost matching (+,---) signature.

    Args:
        x     : (..., d) embeddings, d must be multiple of 4.
        coords: (..., 3) spatial positions (x, y, z).
        times : (...,)   temporal positions (t). Should broadcast with coords.
        base  : Controls frequency range, similar to RoPE.

    Returns:
        (..., d) embeddings with positional information applied.
    """
    *pref, d = x.shape
    if d % 4 != 0:
        raise ValueError(f"Embedding dimension d ({d}) must be a multiple of 4.")

    B = d // 4
    xb = rearrange(x, "... (b k) -> ... b k", b=B, k=4)

    freqs = jnp.arange(B, dtype=x.dtype)
    invf = 1.0 / (base ** (freqs / B)) # Matches original intent more if B replaces 2*B

    # --- Spatial Rotation Part ---
    theta_vec = coords[..., None, :] * invf[None, :, None]
    theta = jnp.linalg.norm(theta_vec, axis=-1, keepdims=True)
    axis_u = theta_vec / jnp.maximum(theta, 1e-8) # Epsilon for stability

    R3 = build_rotation_matrix(axis_u, theta)
    M_rot = jnp.zeros((*R3.shape[:-2], 4, 4), dtype=x.dtype)
    M_rot = M_rot.at[..., 0, 0].set(1.0)
    M_rot = M_rot.at[..., 1:, 1:].set(R3)

    # --- Temporal Boost Part ---
    phi = times[..., None, None] * invf[None, :, None]
    ch = jnp.cosh(phi)
    sh = jnp.sinh(phi)

    M_boost = jnp.zeros((*axis_u.shape[:-1], 4, 4), dtype=x.dtype)
    M_boost = M_boost.at[..., 0, 0].set(ch[..., 0])
    M_boost = M_boost.at[..., 0, 1:].set(-axis_u * sh[..., 0, None]) # Sign fixed for (+,---)
    M_boost = M_boost.at[..., 1:, 0].set(-axis_u * sh[..., 0, None]) # Sign fixed for (+,---)

    eye3 = jnp.eye(3, dtype=x.dtype)
    uuT_boost = jnp.einsum('...i,...j->...ij', axis_u, axis_u)
    ch_minus_1 = (ch - 1)[..., None]
    M_boost = M_boost.at[..., 1:, 1:].set(eye3 + ch_minus_1 * uuT_boost)

    # --- Combine and Apply ---
    R4 = jnp.einsum("...ij,...jk->...ik", M_boost, M_rot) # Boost then Rotation
    yb = jnp.einsum("...ij,...bj->...bi", R4, xb)
    return rearrange(yb, "... b k -> ... (b k)")
