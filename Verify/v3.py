# An extension of verify2.py that demonstrates the RoPE properties on randomly
# generated vectors.

import numpy as np
import random

# --- Setup ---

# 1. Define the rotation frequencies for each 2D-subspace.
# Following the original paper, we pair dimensions (0,1), (2,3), etc.
# These are kept constant to isolate the effect of position.
theta = np.array([0.1, 0.2], dtype=np.float32)

# 2. Define the rotation function.
# This function applies the rotation matrix R_p to a 4D vector vec.
def apply_rotation_4d(vec, pos, theta_pair):
    """Applies RoPE rotation to a 4D vector."""
    # Split into two 2D vectors for paired rotation
    vec1, vec2 = vec[:2], vec[2:]

    # Helper for 2D rotation
    def apply_rotation_2d(vec_2d, p, t):
        """Applies a 2D rotation based on position p and frequency t."""
        x, y = vec_2d
        cos_pt = np.cos(p * t)
        sin_pt = np.sin(p * t)
        return np.array([
            x * cos_pt - y * sin_pt,
            x * sin_pt + y * cos_pt
        ], dtype=np.float32)

    # Apply rotation to each 2D group and concatenate the results
    rotated1 = apply_rotation_2d(vec1, pos, theta_pair[0])
    rotated2 = apply_rotation_2d(vec2, pos, theta_pair[1])
    return np.concatenate([rotated1, rotated2])

def complex_relative_dot_product_4d(q, k, rel_pos, theta_pair):
    """
    Calculates the dot product using the relative position formula for 4D vectors.
    This is the "clever" formulation of RoPE that directly uses the relative distance.
    """
    # Split into two 2D vectors for paired rotation
    q1, q2 = q[:2], q[2:]
    k1, k2 = k[:2], k[2:]
    
    # Calculate dot product for each 2D pair using complex numbers
    def complex_relative_dot_product_2d(q_2d, k_2d, rel_pos, theta):
        """Calculates the dot product using the relative position formula for 2D vectors."""
        # Represent 2D vectors as complex numbers
        q_complex = q_2d[0] + 1j * q_2d[1]
        k_complex = k_2d[0] + 1j * k_2d[1]

        # Rotation in complex plane is multiplication by e^(i*theta)
        rotational_term = np.exp(1j * rel_pos * theta).astype(np.complex64)

        # The core RoPE formula
        # <R_m*q, R_n*k> = Re( (q * k_conjugate) * e^(i*(m-n)*theta) )
        dot_product_complex = q_complex * np.conj(k_complex) * rotational_term

        return np.real(dot_product_complex)
    
    # Sum the dot products from both 2D pairs
    dot1 = complex_relative_dot_product_2d(q1, k1, rel_pos, theta_pair[0])
    dot2 = complex_relative_dot_product_2d(q2, k2, rel_pos, theta_pair[1])
    
    return dot1 + dot2

def run_test(test_num):
    """
    Runs a single experiment to verify the RoPE principle.
    - Generates random q and k vectors.
    - Generates three position pairs with an identical relative distance.
    - Calculates and compares the dot products.
    """
    print(f"--- Test Run #{test_num} ---")

    # --- Experiment Setup ---

    # 1. Generate a single pair of random query (q) and key (k) vectors.
    # The values are uniformly distributed between -0.6 and 0.6.
    q = (0.6 - (-0.6)) * np.random.rand(4) - 0.6
    k = (0.6 - (-0.6)) * np.random.rand(4) - 0.6
    print(f"Random Query Vector (q): [{q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f} {q[3]:+.3f}]")
    print(f"Random Key Vector   (k): [{k[0]:+.3f} {k[1]:+.3f} {k[2]:+.3f} {k[3]:+.3f}]\n")


    # 2. We want to show that the dot product <R_m*q, R_n*k> depends only on (m-n).
    # To prove this, we will test three different pairs of positions
    # that have the same relative distance (delta).

    # Generate a random relative distance for this test run.
    delta = random.randint(3, 10)
    print(f"Using a constant relative distance (delta) of: {delta}\n")

    # Position Pair 1
    n1 = random.randint(1, 15)
    m1 = n1 + delta
    
    # Position Pair 2
    n2 = random.randint(1, 15)
    m2 = n2 + delta

    # Position Pair 3
    n3 = random.randint(1, 15)
    m3 = n3 + delta

    position_pairs = [(m1, n1), (m2, n2), (m3, n3)]
    dot_products = []

    # --- Calculations ---

    for i, (m, n) in enumerate(position_pairs):
        print(f"Position Pair {i+1}: (m={m}, n={n})")
        
        # Apply rotations for the current pair of positions
        q_rotated = apply_rotation_4d(q, m, theta) # This is R_m * q
        k_rotated = apply_rotation_4d(k, n, theta) # This is R_n * k
        
        # Calculate and store the dot product
        dot_product = np.dot(q_rotated, k_rotated)
        dot_products.append(dot_product)
        print(f"Dot product for pair {i+1}: {dot_product:.8f}\n")

    # Calculate the direct dot product using the relative position formula
    direct_dot_product = complex_relative_dot_product_4d(q, k, delta, theta)
    print(f"Position Delta:  (d={delta})")
    print(f"Dot product for delta:  {direct_dot_product:.8f}\n")

    # --- Verification ---
    
    # The punchline: Because the relative distance is the same for all pairs,
    # the dot products should be identical.
    dot1, dot2, dot3 = dot_products
    
    print("--- Verification of RoPE Principle for this Run ---")
    print("Are all three dot products equal?")
    # np.allclose is used to check for equality with a small tolerance for floating point errors.
    are_equal = np.allclose([dot1, dot2], dot3)
    print(f"Answer: {are_equal}")
    if not are_equal:
        print(f"Difference (1 vs 2): {abs(dot1 - dot2)}")
        print(f"Difference (1 vs 3): {abs(dot1 - dot3)}")
        print(f"Difference (2 vs 3): {abs(dot2 - dot3)}")
    
    print("\nAre all three dot products equal to the direct calculation?")
    all_equal_to_direct = np.allclose([dot1, dot2, dot3], direct_dot_product)
    print(f"Answer: {all_equal_to_direct}")
    if not all_equal_to_direct:
        print(f"Difference (1 vs direct): {abs(dot1 - direct_dot_product)}")
        print(f"Difference (2 vs direct): {abs(dot2 - direct_dot_product)}")
        print(f"Difference (3 vs direct): {abs(dot3 - direct_dot_product)}")
    
    print("\n" + "="*40 + "\n")


# --- Main Execution ---
# Run the experiment multiple times to show it holds true for
# different random vectors and different position pairs.
if __name__ == "__main__":
    number_of_tests = 5
    for i in range(1, number_of_tests + 1):
        run_test(i)

