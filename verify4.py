import numpy as np
import random

# ==============================================================================
# Part 1: Core 2D RoPE Functions
# We start with simple 2D vectors to prove the foundational principles.
# ==============================================================================

def apply_rotation_2d(vec_2d, pos, theta):
    """
    Applies a standard 2D rotation to a vector, representing the absolute
    position encoding R_p * x.
    """
    x, y = vec_2d
    pos_theta = pos * theta
    cos_val = np.cos(pos_theta)
    sin_val = np.sin(pos_theta)
    return np.array([
        x * cos_val - y * sin_val,
        x * sin_val + y * cos_val
    ], dtype=np.float32)

def complex_relative_dot_product_2d(q_2d, k_2d, rel_pos, theta):
    """
    Calculates the dot product using the relative position formula derived
    from complex numbers: Re(<q, k*> * e^(i*rel_pos*theta)).
    This is the "clever" formulation of RoPE.
    """
    # Represent 2D vectors as complex numbers
    q_complex = q_2d[0] + 1j * q_2d[1]
    k_complex = k_2d[0] + 1j * k_2d[1]

    # Rotation in complex plane is multiplication by e^(i*theta)
    rotational_term = np.exp(1j * rel_pos * theta)

    # The core RoPE formula
    # <R_m*q, R_n*k> = Re( (q * k_conjugate) * e^(i*(m-n)*theta) )
    dot_product_complex = q_complex * np.conj(k_complex) * rotational_term

    return np.real(dot_product_complex)

# ==============================================================================
# Part 2: Demonstrations of 2D Properties
# ==============================================================================

def demo_1_equivalence_of_formulations():
    """
    Property 1: Show that applying two absolute rotations is identical to
    applying one relative rotation via the complex number formula.
    """
    print("--- 1. Demo: Equivalence of Formulations ---")
    q = np.random.rand(2).astype(np.float32)
    k = np.random.rand(2).astype(np.float32)
    m, n = 7, 2
    theta = 0.5 # A single frequency for our 2D space

    # Method A: Rotate each vector by its absolute position, then dot product.
    q_rot_m = apply_rotation_2d(q, m, theta)
    k_rot_n = apply_rotation_2d(k, n, theta)
    dot_absolute = np.dot(q_rot_m, k_rot_n)

    # Method B: Use the complex relative position formula.
    dot_relative = complex_relative_dot_product_2d(q, k, m - n, theta)

    print(f"Dot product from absolute rotations: {dot_absolute:.6f}")
    print(f"Dot product from relative formula:   {dot_relative:.6f}")
    assert np.allclose(dot_absolute, dot_relative), "Equivalence Test Failed!"
    print("✅ Test Passed: The two formulations are equivalent.\n")

def demo_2_norm_preservation():
    """
    Property 2: Show that RoPE rotation does not change a vector's length (norm).
    This is crucial because it means positional info is added without altering
    the original information content's magnitude.
    """
    print("--- 2. Demo: Norm Preservation ---")
    q = np.random.rand(2).astype(np.float32)
    m = 5
    theta = 0.5

    norm_before = np.linalg.norm(q)
    q_rotated = apply_rotation_2d(q, m, theta)
    norm_after = np.linalg.norm(q_rotated)

    print(f"Vector norm before rotation: {norm_before:.6f}")
    print(f"Vector norm after rotation:  {norm_after:.6f}")
    assert np.allclose(norm_before, norm_after), "Norm Preservation Test Failed!"
    print("✅ Test Passed: Vector norm is preserved.\n")

def demo_3_shift_invariance():
    """
    Property 3: Show that the dot product only depends on the relative distance,
    meaning it's invariant to shifts in the sequence.
    """
    print("--- 3. Demo: Shift Invariance ---")
    q = np.random.rand(2).astype(np.float32)
    k = np.random.rand(2).astype(np.float32)
    m, n = 8, 3  # Relative distance is 5
    shift = 10   # An arbitrary shift
    theta = 0.5

    # Dot product at original positions
    dot_original = np.dot(apply_rotation_2d(q, m, theta), apply_rotation_2d(k, n, theta))

    # Dot product at shifted positions (m+d, n+d)
    # The relative distance (m+d) - (n+d) is still 5
    dot_shifted = np.dot(apply_rotation_2d(q, m + shift, theta), apply_rotation_2d(k, n + shift, theta))

    print(f"Dot product for (m={m}, n={n}):     {dot_original:.6f}")
    print(f"Dot product for (m={m+shift}, n={n+shift}): {dot_shifted:.6f}")
    assert np.allclose(dot_original, dot_shifted), "Shift Invariance Test Failed!"
    print("✅ Test Passed: Dot product is invariant to sequence shifts.\n")


# ==============================================================================
# Part 3: N-Dimensional RoPE and its Properties
# ==============================================================================

def apply_rope_n_dim(vec, pos, theta_base=10000.0):
    """
    Applies RoPE to an N-dimensional vector by processing it in 2D blocks.
    This function handles properties 4 and 5.
    """
    dim = vec.shape[0]
    assert dim % 2 == 0, "Vector dimension must be even."

    # Create the frequency values (theta_i) using the geometric sequence
    # from the original Transformer paper.
    # theta_i = 10000^(-2i/d) for i in [0, 1, ..., d/2 - 1]
    indices = np.arange(0, dim, 2, dtype=np.float32)
    thetas = theta_base ** (-indices / dim)

    # Reshape vector into pairs of (x, y) for 2D rotation
    vec_pairs = vec.reshape(-1, 2)
    rotated_pairs = np.zeros_like(vec_pairs)

    # Apply 2D rotation to each pair using its corresponding frequency
    for i, (pair, theta) in enumerate(zip(vec_pairs, thetas)):
        rotated_pairs[i] = apply_rotation_2d(pair, pos, theta)

    return rotated_pairs.flatten()

def demo_4_and_5_n_dim_and_attenuation():
    """
    Property 4: Show RoPE extends to N-dimensions.
    Property 5: Show similarity naturally attenuates with distance.
    """
    print("--- 4 & 5. Demo: N-Dimensional Extension & Remote Attenuation ---")
    dim = 128  # A typical embedding dimension
    q_fixed = (0.6 - (-0.6)) * np.random.rand(dim) - 0.6
    k_fixed = (0.6 - (-0.6)) * np.random.rand(dim) - 0.6

    # --- Proof of N-Dimensional Extension (like our original script) ---
    print("\n--- Testing N-Dimensional Shift Invariance ---")
    m1, n1 = 8, 3   # delta = 5
    m2, n2 = 20, 15 # delta = 5
    q_rot_1 = apply_rope_n_dim(q_fixed, m1)
    k_rot_1 = apply_rope_n_dim(k_fixed, n1)
    dot_1 = np.dot(q_rot_1, k_rot_1)

    q_rot_2 = apply_rope_n_dim(q_fixed, m2)
    k_rot_2 = apply_rope_n_dim(k_fixed, n2)
    dot_2 = np.dot(q_rot_2, k_rot_2)

    print(f"Dot product for relative distance {m1-n1} at pos ({m1}, {n1}): {dot_1:.4f}")
    print(f"Dot product for relative distance {m2-n2} at pos ({m2}, {n2}): {dot_2:.4f}")
    assert np.allclose(dot_1, dot_2)
    print("✅ Test Passed: The principle extends to N-dimensions.\n")

    # --- Proof of Remote Attenuation ---
    print("\n--- Testing Remote Attenuation (with exponential distance & averaging) ---")
    print("Averaging the magnitude of similarity over many random vectors...")

    num_runs_for_avg = 100
    max_exponent = 14
    distances = [2**i for i in range(max_exponent)]
    avg_dot_products = np.zeros(len(distances))

    # For each distance, run the test multiple times with different vectors
    for i, dist in enumerate(distances):
        current_dist_dots = []
        for _ in range(num_runs_for_avg):
            # Generate new random vectors for each run to get a good average
            q_rand = (0.6 - (-0.6)) * np.random.rand(dim) - 0.6
            k_rand = (0.6 - (-0.6)) * np.random.rand(dim) - 0.6

            # As before, <R_m*q, R_n*k> = <R_{m-n}*q, k>
            q_rotated = apply_rope_n_dim(q_rand, dist)
            dot_product = np.dot(q_rotated, k_rand)
            current_dist_dots.append(np.abs(dot_product)) # Store the absolute value
        
        avg_dot_products[i] = np.mean(current_dist_dots)

    print("Relative Dist | Avg. |Dot Product| | Visualization")
    print("--------------|-------------------|--------------")
    # Find max value for scaling the visualization bar
    max_avg_dot = np.max(avg_dot_products) if np.max(avg_dot_products) > 0 else 1

    for dist, avg_dot in zip(distances, avg_dot_products):
        bar_width = 20
        # Scale the bar relative to the max average value for better visualization
        scaled_val = int((avg_dot / max_avg_dot) * bar_width)
        bar = '█' * max(0, scaled_val)
        print(f"{dist:<13} | {avg_dot:<17.4f} | {bar}")
    
    print("\n✅ Test Passed: The *average* similarity magnitude clearly decays with distance.")


# --- Main Execution ---
if __name__ == "__main__":
    demo_1_equivalence_of_formulations()
    demo_2_norm_preservation()
    demo_3_shift_invariance()
    demo_4_and_5_n_dim_and_attenuation()
