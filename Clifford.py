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
    # Ensure theta has a trailing dimension of 1 if it doesn't
    if theta.shape[-1] != 1:
        theta = theta[..., None]
        
    cos_t = jnp.cos(theta)[...,0]
    sin_t = jnp.sin(theta)[...,0]
    ux, uy, uz = axis[...,0], axis[...,1], axis[...,2]
    
    # Compute u outer product u^T
    uuT = jnp.einsum('...i,...j->...ij', axis, axis)
    
    # Compute cross product matrix K such that K v = u x v
    zeros = jnp.zeros_like(ux)
    u_cross = jnp.stack([
        zeros,  -uz,   uy,
           uz, zeros,  -ux,
          -uy,   ux, zeros
    ], axis=-1).reshape((*axis.shape[:-1], 3, 3))
    
    I3 = jnp.eye(3, dtype=axis.dtype) # Ensure dtype matches axis
    
    # Rodrigues' formula: R = I*cos(t) + (1-cos(t))*uuT + sin(t)*u_cross
    # Add dimensions to scalars for broadcasting
    cos_t = cos_t[..., None, None]
    sin_t = sin_t[..., None, None]
    
    return (cos_t * I3 +
            (1 - cos_t) * uuT +
            sin_t * u_cross)

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
    
    B = d // 4 # Number of 4-vector blocks

    # Reshape input into blocks of 4
    # Example: (..., 512) -> (..., 128, 4) if d=512
    xb = rearrange(x, "... (b k) -> ... b k", b=B, k=4)

    # Calculate inverse frequencies for each block dimension
    # Shape: (B,)
    freqs = jnp.arange(B, dtype=x.dtype) # Match dtype
    invf = 1.0 / (base ** (freqs / B)) # Note: RoPE often uses 2*freqs/d or similar
                                       # Here, using freqs/B implies a range over blocks. Adjust if needed.

    # --- Spatial Rotation Part ---
    
    # Scale spatial coordinates by frequencies and calculate angle/axis
    # invf needs shape (1, B, 1) to broadcast with coords (..., 1, 3)
    theta_vec = coords[..., None, :] * invf[None, :, None]  # Shape: (..., B, 3)
    
    # Calculate rotation angle (magnitude) per block
    theta = jnp.linalg.norm(theta_vec, axis=-1, keepdims=True)  # Shape: (..., B, 1)
    
    # Calculate unit axis, adding epsilon for numerical stability
    axis_u = theta_vec / jnp.maximum(theta, 1e-8) # Shape: (..., B, 3)

    # Build 3x3 rotation matrix using Rodrigues' formula
    R3 = build_rotation_matrix(axis_u, theta)  # Shape: (..., B, 3, 3)
    
    # Embed R3 into a 4x4 matrix (pure spatial rotation in Minkowski space)
    M_rot = jnp.zeros((*R3.shape[:-2], 4, 4), dtype=x.dtype)
    M_rot = M_rot.at[..., 0, 0].set(1.0)
    M_rot = M_rot.at[..., 1:, 1:].set(R3) # Shape: (..., B, 4, 4)

    # --- Temporal Boost Part ---
    
    # Calculate rapidity phi per block
    # Ensure times has shape (...,) -> (..., 1) -> (..., 1, 1) for broadcasting
    phi = times[..., None, None] * invf[None, :, None]  # Shape: (..., B, 1)
    
    ch = jnp.cosh(phi) # Shape: (..., B, 1)
    sh = jnp.sinh(phi) # Shape: (..., B, 1)

    # Assemble the 4x4 Lorentz Boost matrix (standard convention for +,---)
    M_boost = jnp.zeros((*axis_u.shape[:-1], 4, 4), dtype=x.dtype) # Shape: (..., B, 4, 4)
    
    # L_{00} = gamma = cosh(phi)
    M_boost = M_boost.at[..., 0, 0].set(ch[..., 0])
    
    # L_{0k} = -gamma * beta_k = -sinh(phi) * u_k
    # Note the negative sign aligns with standard (+,---) convention
    M_boost = M_boost.at[..., 0, 1:].set(-axis_u * sh[..., 0, None]) # Fixed sign
    
    # L_{k0} = -gamma * beta_k = -sinh(phi) * u_k
    # Note the negative sign aligns with standard (+,---) convention
    M_boost = M_boost.at[..., 1:, 0].set(-axis_u * sh[..., 0, None]) # Fixed sign
    
    # L_{ij} = delta_ij + (gamma - 1) * u_i * u_j
    eye3 = jnp.eye(3, dtype=x.dtype)
    # Need axis_u outer product: (..., B, 3, 1) * (..., B, 1, 3) -> (..., B, 3, 3)
    uuT_boost = jnp.einsum('...i,...j->...ij', axis_u, axis_u)
    # Need ch-1 with shape (..., B, 1, 1) for broadcasting
    ch_minus_1 = (ch - 1)[..., None]
    
    M_boost = M_boost.at[..., 1:, 1:].set(eye3 + ch_minus_1 * uuT_boost)

    # --- Combine and Apply ---
    
    # Multiply Boost * Rotation. Note: matrix multiplication applies right-to-left.
    # R4 @ vector = M_boost @ (M_rot @ vector) -> Rotation first, then Boost.
    R4 = jnp.einsum("...ij,...jk->...ik", M_boost, M_rot) # Shape: (..., B, 4, 4)
    
    # Apply the combined Lorentz transformation R4 to each 4-vector block xb
    # einsum performs batched matrix-vector multiplication for each block b
    # R4 shape: (..., B, 4, 4), xb shape: (..., B, 4)
    # Output yb shape: (..., B, 4)
    yb = jnp.einsum("...ij,...bj->...bi", R4, xb) 
    
    # Reshape back to original embedding dimension
    # Example: (..., 128, 4) -> (..., 512)
    return rearrange(yb, "... b k -> ... (b k)")
