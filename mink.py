"""
Minkowski inner product implementation for 4D vectors with signature (+, -, -, -).
This implementation aligns with the MonSTER (Minkowski Space Time Embedding Rotors)
convention for verifying invariance of transformations.
"""

import jax.numpy as jnp
from typing import Union, Tuple
from jax.typing import ArrayLike

def minkowski_dot(a: ArrayLike, b: ArrayLike) -> jnp.ndarray:
    """
    Compute the Minkowski inner product between two 4D vectors with signature (+, -, -, -).
    
    The Minkowski inner product is defined as:
    η(a, b) = a₀b₀ - a₁b₁ - a₂b₂ - a₃b₃
    where a₀ is the time component and a₁, a₂, a₃ are the space components.
    
    Args:
        a: First 4D vector or batch of vectors with shape (..., 4)
        b: Second 4D vector or batch of vectors with shape (..., 4)
        
    Returns:
        The Minkowski inner product with shape (...)
        
    Note:
        This implementation assumes the first component (index 0) is the time component
        and the remaining components (indices 1,2,3) are spatial components, matching
        the MonSTER convention.
    """
    # Ensure inputs are JAX arrays
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    
    # Extract time and space components
    a_time = a[..., 0]
    a_space = a[..., 1:]
    b_time = b[..., 0]
    b_space = b[..., 1:]
    
    # Compute Minkowski inner product: η(a,b) = a₀b₀ - a₁b₁ - a₂b₂ - a₃b₃
    time_term = a_time * b_time
    space_term = jnp.sum(a_space * b_space, axis=-1)
    
    return time_term - space_term

def minkowski_norm(v: ArrayLike) -> jnp.ndarray:
    """
    Compute the Minkowski norm (squared) of a 4D vector.
    
    The Minkowski norm is defined as:
    ||v||² = v₀² - v₁² - v₂² - v₃²
    
    Args:
        v: 4D vector or batch of vectors with shape (..., 4)
        
    Returns:
        The Minkowski norm (squared) with shape (...)
    """
    return minkowski_dot(v, v)
