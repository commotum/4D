import numpy as np

# --------------------------------------------------------------------------- #
# 1.  Minkowski metric (+ – – –) and helpers
# --------------------------------------------------------------------------- #

# This line creates the Minkowski metric tensor, 
# commonly represented by η, the Greek letter eta.
η = np.diag([1.0, -1.0, -1.0, -1.0]).astype(np.float64)

# 1. Apply the metric to u: (u @ η) = u' (flips spatial components)
# 2. Compute final dot product (u'⋅v) = (u_t*v_t) - (u_s⋅v_s).
def minkowski_dot(u, v):
    """<u,v>_η  for row vectors."""
    return (u @ η) @ v

# 1. Prepare L for row-vector action by transposing: L → Lᵀ.
# 2. Left-multiply by the vector to get the transformed vector: v' = v @ Lᵀ.
def apply_lorentz(vector, L):
    """Row-vector action  v' = v · Lᵀ."""
    return vector @ L.T

# --------------------------------------------------------------------------- #
# 2.  Generators for an arbitrary unit axis  û
# --------------------------------------------------------------------------- #
u_hat = np.array([1.0, 1.0, 1.0])
u_hat /= np.linalg.norm(u_hat)          # (1,1,1)/√3

def boost_u(phi, u=u_hat):
    """Lorentz boost mixing (t, û·r)."""
    ch, sh = np.cosh(phi), np.sinh(phi)
    L = np.eye(4)
    L[0,0] = ch
    L[0, 1:]  = -sh * u                 # time–space
    L[1:, 0]  = -sh * u
    L[1:, 1:] += (ch - 1.0) * np.outer(u, u)  # spatial block
    return L.astype(np.float64)

def rot_u(theta, u=u_hat):
    """Spatial rotation by θ about axis û (Rodrigues)."""
    c, s = np.cos(theta), np.sin(theta)
    ux, uy, uz = u
    K = np.array([[  0, -uz,  uy],
                  [ uz,   0, -ux],
                  [-uy,  ux,   0]])
    R3 = c * np.eye(3) + (1 - c) * np.outer(u, u) + s * K
    L = np.eye(4)
    L[1:, 1:] = R3
    return L.astype(np.float64)

# --------------------------------------------------------------------------- #
# 3.  Minkowski Rotary Embedding Module (Precomputation)
# --------------------------------------------------------------------------- #
 
DIM         = 512                         # embedding dim (== vector dim here)
NUM_BLOCKS  = DIM // 4

if DIM % 4 != 0:
    raise ValueError(f"DIM ({DIM}) must be divisible by 4.")

class MinkowskiRotaryEmbedding:
    """
    Module to precompute and cache the Minkowski transformation matrices.
    This is analogous to the RotaryEmbedding module that precomputes cos/sin tables.
    """
    def __init__(self, dim, base=10000):
        self.dim = dim
        self.num_blocks = dim // 4
        self.base = base
        # Precompute the inverse frequencies, one for each 4D block
        self.inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 4) / self.dim))
        # Cache for storing precomputed transformation matrices for given positions
        self._cache = {}

    def _L_from_position(self, s, idx):
        """Helper to build L(s) for a specific frequency index."""
        s = np.asarray(s, dtype=np.float64)
        t, r = s[0], s[1:]
        
        freq = self.inv_freq[idx]
        φ = t * freq
        θ = (u_hat @ r) * freq
        
        return rot_u(θ) @ boost_u(φ)

    def forward(self, position_vector):
        """
        Takes a 4D position vector and returns a list of 4x4 transformation matrices.
        
        Args:
            position_vector (np.ndarray): The 4D position vector 's'.
            
        Returns:
            list[np.ndarray]: A list of NUM_BLOCKS (e.g., 128) 4x4 matrices.
        """
        # Use a tuple representation of the position vector as a hashable cache key
        pos_key = tuple(position_vector)
        
        if pos_key in self._cache:
            return self._cache[pos_key]

        # If not cached, compute the list of transformation matrices
        L_matrices = [self._L_from_position(position_vector, i) for i in range(self.num_blocks)]
        
        self._cache[pos_key] = L_matrices
        return L_matrices

# --------------------------------------------------------------------------- #
# 4.  Function to Apply the Precomputed Transformations
# --------------------------------------------------------------------------- #

def apply_rope_minkowski(embedding_vector, precomputed_Ls):
    """
    Applies the precomputed Minkowski transformations to a full embedding vector.
    This is analogous to the apply_rotary_pos_emb function.
    
    Args:
        embedding_vector (np.ndarray): The high-dimensional vector (e.g., 512-dim).
        precomputed_Ls (list[np.ndarray]): The list of precomputed 4x4 matrices from the module.
        
    Returns:
        np.ndarray: The transformed high-dimensional vector.
    """
    if embedding_vector.shape[0] != DIM:
        raise ValueError(f"Input embedding vector must have dimension {DIM}")

    transformed_vector = np.zeros_like(embedding_vector)

    for i in range(NUM_BLOCKS):
        start, end = i * 4, (i + 1) * 4
        block = embedding_vector[start:end]
        
        # Apply the i-th precomputed transformation to the i-th block
        transformed_block = apply_lorentz(block, precomputed_Ls[i])
        
        transformed_vector[start:end] = transformed_block
        
    return transformed_vector

# --------------------------------------------------------------------------- #
# 5.  Example Usage
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    # 1. Instantiate the embedding module once in your model's __init__
    minkowski_rope = MinkowskiRotaryEmbedding(dim=DIM)

    # 2. In your model's forward pass, get the position vector for a token
    #    This could be from a grid, a sequence index, etc.
    position_s = np.array([10, 5, -3, 8]) # Example 4D position

    # 3. Precompute the list of 4x4 transformation matrices for this position
    #    This is analogous to getting the cos/sin tables.
    L_matrices_for_s = minkowski_rope.forward(position_s)
    
    print(f"Generated {len(L_matrices_for_s)} transformation matrices for position {position_s}.")
    print(f"Shape of the first matrix: {L_matrices_for_s[0].shape}")

    # 4. Get your query or key embedding vector
    q_embedding = np.random.randn(DIM)

    # 5. Apply the positional encoding
    q_rotated = apply_rope_minkowski(q_embedding, L_matrices_for_s)

    print(f"\nOriginal embedding (first 8 elements): {q_embedding[:8]}")
    print(f"Rotated embedding (first 8 elements):  {q_rotated[:8]}")

