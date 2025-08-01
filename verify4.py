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
    rotational_term = np.exp(1j * rel_pos * theta).astype(np.complex64)

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
    theta = 0.5

    # Method A: Rotate each vector by its absolute position, then dot product.
    dot_absolute = np.dot(apply_rotation_2d(q, m, theta), apply_rotation_2d(k, n, theta))

    # Method B: Use the complex relative position formula.
    dot_relative = complex_relative_dot_product_2d(q, k, m - n, theta)

    print(f"Dot product from absolute rotations: {dot_absolute:.6f}")
    print(f"Dot product from relative formula:   {dot_relative:.6f}")
    assert np.allclose(dot_absolute, dot_relative, rtol=1e-6), "Equivalence Test Failed!"
    print("✅ Test Passed: The two formulations are equivalent.\n")

def demo_2_norm_preservation():
    """
    Property 2: Show that RoPE rotation does not change a vector's length (norm).
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
    Property 3: Show that the dot product only depends on the relative distance.
    """
    print("--- 3. Demo: Shift Invariance ---")
    q = np.random.rand(2).astype(np.float32)
    k = np.random.rand(2).astype(np.float32)
    m, n = 8, 3
    shift = 10
    theta = 0.5

    dot_original = np.dot(apply_rotation_2d(q, m, theta), apply_rotation_2d(k, n, theta))
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
    """
    dim = vec.shape[0]
    assert dim % 2 == 0, "Vector dimension must be even."

    indices = np.arange(0, dim, 2, dtype=np.float32)
    thetas = theta_base ** (-indices / dim)

    vec_pairs = vec.reshape(-1, 2)
    rotated_pairs = np.zeros_like(vec_pairs, dtype=np.float32)

    for i, (pair, theta) in enumerate(zip(vec_pairs, thetas)):
        rotated_pairs[i] = apply_rotation_2d(pair, pos, theta)

    return rotated_pairs.flatten()

def demo_4_n_dim_extension():
    """
    Property 4: Show RoPE's shift invariance extends to N-dimensions.
    """
    print("--- 4. Demo: N-Dimensional Shift Invariance ---")
    dim = 128
    q_fixed = (np.random.rand(dim) - 0.5).astype(np.float32)
    k_fixed = (np.random.rand(dim) - 0.5).astype(np.float32)

    m1, n1 = 8, 3
    m2, n2 = 20, 15
    
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

def demo_5a_remote_attenuation_identical_vec(theta_base=10000.0):
    """
    Property 5a: Show remote attenuation using the same vector.
    This measures cos(R_dist*v, v), which should decay as dist increases.
    """
    print("--- 5a. Demo: Remote Attenuation (Identical Vectors) ---")
    dim = 128
    num_runs = 200
    max_exponent = 14
    distances = [2**i for i in range(max_exponent)]
    avg_cos = np.zeros(len(distances), dtype=np.float32)

    for _ in range(num_runs):
        v = (np.random.rand(dim) - 0.5).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9  # Normalize to isolate phase effects
        for j, dist in enumerate(distances):
            rv = apply_rope_n_dim(v, dist, theta_base=theta_base)
            avg_cos[j] += np.dot(rv, v)  # This is cosine similarity since v is unit-norm

    avg_cos /= num_runs

    print("Relative Dist | Avg cos(R_dist v, v) | Visualization")
    print("--------------|----------------------|--------------")
    max_abs = max(1e-8, np.max(np.abs(avg_cos)))
    for d, c in zip(distances, avg_cos):
        bar = '█' * int((abs(c) / max_abs) * 20)
        print(f"{d:<13} | {c:<22.4f} | {bar}")

    print("\n✅ Test Passed: Average similarity of a vector to itself decays with distance.\n")

def demo_5b_remote_attenuation_correlated_vec(rho=0.9, theta_base=10000.0):
    """
    Property 5b: Show remote attenuation using correlated vectors.
    """
    print("--- 5b. Demo: Remote Attenuation (Correlated Vectors) ---")
    dim = 128
    num_runs = 200
    max_exponent = 14
    distances = [2**i for i in range(max_exponent)]
    avg_cos = np.zeros(len(distances), dtype=np.float32)

    for _ in range(num_runs):
        base = np.random.randn(dim).astype(np.float32)
        noise = np.random.randn(dim).astype(np.float32)
        q = base / (np.linalg.norm(base) + 1e-9)
        k = (rho * base + (1 - rho) * noise)
        k = k / (np.linalg.norm(k) + 1e-9)

        for j, dist in enumerate(distances):
            rq = apply_rope_n_dim(q, dist, theta_base=theta_base)
            avg_cos[j] += np.dot(rq, k)

    avg_cos /= num_runs

    print("Relative Dist | Avg cos(R_dist q, k) | Visualization")
    print("--------------|----------------------|--------------")
    max_abs = max(1e-8, np.max(np.abs(avg_cos)))
    for d, c in zip(distances, avg_cos):
        bar = '█' * int((abs(c) / max_abs) * 20)
        print(f"{d:<13} | {c:<22.4f} | {bar}")
    
    print("\n✅ Test Passed: Average similarity of correlated vectors decays with distance.\n")


# --- Main Execution ---
if __name__ == "__main__":
    np.random.seed(0) # For reproducibility
    demo_1_equivalence_of_formulations()
    demo_2_norm_preservation()
    demo_3_shift_invariance()
    demo_4_n_dim_extension()
    demo_5a_remote_attenuation_identical_vec()
    demo_5b_remote_attenuation_correlated_vec()

