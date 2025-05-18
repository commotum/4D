import numpy as np
import random
import itertools

def generate_even_intervals(n):
    """Generate n even intervals between -1 and 1, returning n+1 values."""
    return np.linspace(-1.0, 1.0, n + 1)

def generate_signed_vectors4():
    """
    Generate 4 random 3-vectors with integer values 0..16,
    randomly flipping 0 to 3 of their coords to negative.
    Returns only the signed vectors.
    """
    signed_vecs = []
    for _ in range(4):
        v = [random.randint(0, 16) for _ in range(3)]
        k = random.choice([0, 1, 2, 3])
        neg_indices = set(random.sample(range(3), k))
        signed = [(-v[i] if i in neg_indices else v[i]) for i in range(3)]
        signed_vecs.append(signed)
    return signed_vecs

# Parameters
n = 4

# Generate inputs
intervals = generate_even_intervals(n)        # 5 values: [-1. , -0.5, 0. , 0.5, 1. ]
spatial_vecs = generate_signed_vectors4()     # 4 vectors, e.g. [[8,11,-1], ...]

# Build all 20 four-vectors
four_vectors = [
    (t, x, y, z)
    for t, (x, y, z) in itertools.product(intervals, spatial_vecs)
]

# Print them
for vec in four_vectors:
    print(vec)

