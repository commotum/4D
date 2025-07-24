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


import jax.numpy as jnp
import numpy as np
import logging

logger = logging.getLogger(__name__)

def demo_compare_dot_products():
    """
    Generate two random 4-vectors with integer values, compute their standard dot product
    and Minkowski dot product, and log the results for comparison.
    """
    # Seed for reproducibility
    rng = np.random.default_rng(seed=42)
    a = rng.integers(low=-10, high=10, size=(4,))
    b = rng.integers(low=-10, high=10, size=(4,))

    logger.info(f"Random 4-vector a: {a}")
    logger.info(f"Random 4-vector b: {b}")

    # Standard dot product
    standard_dot = jnp.dot(a, b)
    logger.info(f"Standard dot product: {standard_dot}")

    # Minkowski dot product
    mink_dot = minkowski_dot(a, b)
    logger.info(f"Minkowski dot product: {mink_dot}")

    print("a =", a)
    print("b =", b)
    print("Standard dot product:", standard_dot)
    print("Minkowski dot product:", mink_dot)

# Optionally run the demo if this file is executed directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_compare_dot_products()


    def main() -> None:
        """
        Main function to configure logging and run the Minkowski dot product demo.
        """
        logging.basicConfig(level=logging.INFO)
        demo_compare_dot_products()
