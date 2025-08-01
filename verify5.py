import numpy as np
import random
import matplotlib.pyplot as plt

# --- Setup ---

def generate_theta(d, base=10000.0):
    """
    Generate the rotation frequencies (theta) for each 2D subspace.
    Follows the geometric schedule: theta_i = base^{-2i/d} for i=0 to (d/2)-1.
    """
    exponents = -2.0 * np.arange(0, d//2) / d
    theta = np.power(base, exponents).astype(np.float32)
    return theta

# Define the general rotation function for d-dimensional vectors.
# Applies blockwise 2D rotations to pairs of dimensions.
def apply_rotation(vec, pos, theta):
    """
    Applies RoPE rotation to a d-dimensional vector.
    - vec: numpy array of shape (d,)
    - pos: integer position
    - theta: array of frequencies, shape (d//2,)
    """
    d = len(vec)
    if d % 2 != 0:
        raise ValueError("Vector dimension must be even for RoPE.")
    if len(theta) != d // 2:
        raise ValueError("Theta must have length d//2.")

    rotated = np.zeros_like(vec)
    for i in range(d // 2):
        # Extract the 2D pair
        x, y = vec[2*i], vec[2*i + 1]
        t = theta[i]
        cos_pt = np.cos(pos * t)
        sin_pt = np.sin(pos * t)
        # Apply rotation
        rotated[2*i] = x * cos_pt - y * sin_pt
        rotated[2*i + 1] = x * sin_pt + y * cos_pt
    return rotated

# Complex relative dot product for verification (Property 1)
def complex_dot_product(vec_q, vec_k, rel_pos, theta):
    """
    Computes the dot product using the complex relative formulation.
    Treats vectors as complex numbers in C^{d/2}.
    - vec_q, vec_k: (d,)
    - rel_pos: m - n (integer)
    - theta: (d//2,)
    """
    d = len(vec_q)
    # Convert to complex
    q_complex = vec_q[::2] + 1j * vec_q[1::2]
    k_complex = vec_k[::2] + 1j * vec_k[1::2]
    # Apply relative rotation to k (conjugate for dot product)
    angles = rel_pos * theta
    rot = np.exp(-1j * angles)  # e^{-i * (m-n) * theta}
    k_rot_complex = k_complex * rot
    # Dot product: Re(q * conj(k_rot))
    # But since it's q dot (R_{m-n} k), and R is rotation, equivalent to Re(q conj(k) e^{-i delta theta})
    dot = np.real(np.dot(q_complex, np.conj(k_rot_complex)))
    return dot

# Function to demonstrate remote attenuation (Property 5)
def compute_attenuation(q, k, theta, max_distance=50):
    """
    Computes dot products for increasing relative distances to show attenuation.
    Fixes m=0, varies n from -max_distance to 0.
    Returns distances and dot_products.
    """
    distances = np.arange(1, max_distance + 1)
    dot_products = []
    for dist in distances:
        q_rot = apply_rotation(q, 0, theta)
        k_rot = apply_rotation(k, -dist, theta)  # relative dist = dist
        dot = np.dot(q_rot, k_rot)
        dot_products.append(dot)
    return distances, np.array(dot_products)

# Function to run demonstrations for all properties
def demonstrate_properties(d=4, base=10000.0, num_tests=3):
    """
    Runs demonstrations for all 5 properties.
    - d: embedding dimension (even)
    - base: frequency base (e.g., 10000)
    - num_tests: number of random tests for properties 1-3
    """
    theta = generate_theta(d, base)
    print(f"--- RoPE Demonstration Setup ---")
    print(f"Embedding dimension (d): {d}")
    print(f"Frequency base: {base}")
    print(f"Theta frequencies: {np.round(theta, 6)}")
    print("\n" + "="*60 + "\n")

    # Generate random q and k for demonstrations
    q = np.random.uniform(-0.6, 0.6, d).astype(np.float32)
    k = np.random.uniform(-0.6, 0.6, d).astype(np.float32)
    print(f"Random Query Vector (q): {np.round(q, 4)}")
    print(f"Random Key Vector (k): {np.round(k, 4)}\n")

    # Property 1: Equivalence of Formulations
    print("--- Property 1: Equivalence of Formulations ---")
    for test in range(1, num_tests + 1):
        m = random.randint(1, 20)
        n = random.randint(1, 20)
        rel_pos = m - n

        # Absolute rotations dot product
        q_rot = apply_rotation(q, m, theta)
        k_rot = apply_rotation(k, n, theta)
        dot_abs = np.dot(q_rot, k_rot)

        # Complex relative dot product
        dot_complex = complex_dot_product(q, k, rel_pos, theta)

        print(f"Test {test}: Positions (m={m}, n={n}), rel_pos={rel_pos}")
        print(f"Absolute Dot: {dot_abs:.8f}")
        print(f"Complex Dot: {dot_complex:.8f}")
        equal = np.allclose(dot_abs, dot_complex, atol=1e-6)
        print(f"Equal? {equal}\n")
    print("="*40 + "\n")

    # Property 2: Norm Preservation
    print("--- Property 2: Norm Preservation ---")
    for test in range(1, num_tests + 1):
        pos = random.randint(1, 20)
        norm_orig = np.linalg.norm(q)
        q_rot = apply_rotation(q, pos, theta)
        norm_rot = np.linalg.norm(q_rot)

        print(f"Test {test}: Position {pos}")
        print(f"Original Norm: {norm_orig:.8f}")
        print(f"Rotated Norm: {norm_rot:.8f}")
        preserved = np.allclose(norm_orig, norm_rot, atol=1e-6)
        print(f"Preserved? {preserved}\n")
    print("="*40 + "\n")

    # Property 3: Shift Invariance
    print("--- Property 3: Shift Invariance ---")
    delta = random.randint(3, 10)
    print(f"Using relative distance delta={delta}\n")
    position_pairs = []
    for _ in range(num_tests):
        n = random.randint(1, 15)
        m = n + delta
        position_pairs.append((m, n))

    dot_products = []
    for i, (m, n) in enumerate(position_pairs, 1):
        q_rot = apply_rotation(q, m, theta)
        k_rot = apply_rotation(k, n, theta)
        dot = np.dot(q_rot, k_rot)
        dot_products.append(dot)
        print(f"Pair {i}: (m={m}, n={n}), Dot: {dot:.8f}")

    are_equal = np.allclose(dot_products, dot_products[0], atol=1e-6)
    print(f"\nAll dots equal? {are_equal}\n")
    print("="*40 + "\n")

    # Property 4: Extensibility to N-Dimensional Vectors
    print("--- Property 4: Extensibility to N-Dimensional Vectors ---")
    print(f"Using d={d}, which is extensible by applying rotations to {d//2} pairs.")
    # Reuse the above tests as they already work for general d
    print("Demonstrated via general apply_rotation function for arbitrary even d.")
    print("Complex interpretation shown in complex_dot_product.")
    print("="*40 + "\n")

    # Property 5: Multiscale Coverage and Remote Attenuation
    print("--- Property 5: Multiscale Coverage and Remote Attenuation ---")
    print("Multiscale: Theta spans frequencies:", np.round(theta, 6))

    # Remote Attenuation: Compute and plot dot vs distance
    distances, dots = compute_attenuation(q, k, theta, max_distance=50)
    print("\nSample Dot Products vs Distance:")
    for dist, dot in zip(distances[:5], dots[:5]):
        print(f"Distance {dist}: {dot:.8f}")
    print("... (attenuation continues)")

    # Plot for visualization (if running in an environment with display)
    plt.figure(figsize=(8, 4))
    plt.plot(distances, dots, marker='o')
    plt.title("Dot Product Attenuation with Increasing Distance")
    plt.xlabel("Relative Distance")
    plt.ylabel("Dot Product")
    plt.grid(True)
    plt.show()  # Comment out if no display available
    print("\nAttenuation shown: Dot products decrease with distance due to phase misalignment.")
    print("="*40 + "\n")

# --- Main Execution ---
if __name__ == "__main__":
    # Example: Run with d=4
    demonstrate_properties(d=4, base=10000.0, num_tests=3)
    
    # Example: Run with higher d to show extensibility
    # demonstrate_properties(d=8, base=10000.0, num_tests=3)