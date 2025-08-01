# RoPE Demo - Proof

To show that a RoPE implementation works in all cases, you would need to demonstrate the following:

### 1. Equivalence of Formulations

    - Show that applying separate rotations to absolute positions yields the same 
      result as the complex relative position formula. For any 2D vectors q and k 
      at positions m and n:

    - np.dot(apply_rotation_2d(q, m), apply_rotation_2d(k, n)) 
      must equal complex_dot_product_2d(q, k, m-n).

### 2. Norm Preservation

    - Show that the rotation doesn't change the vector's length.
    - The magnitude of a vector should remain constant after applying 
      the positional encoding.

    - np.linalg.norm(q) must equal np.linalg.norm(apply_rotation_2d(q, m)).

### 3. Shift Invariance

    - Show that the dot product is the same for any pair of tokens that have
      the same relative distance, regardless of where they are in the sequence.
      
    - For any shift d:
      np.dot(apply_rotation_2d(q, m), apply_rotation_2d(k, n)) must equal 
      np.dot(apply_rotation_2d(q, m+d), apply_rotation_2d(k, n+d)).

### 4. Multiscale Coverage and Remote Attenuation

    - Show that by applying pairwise 2D rotations using the same geometric 
      frequency schedule () from the original sinusoidal positional encoding 
      retains:

    1. Multiscale positional coverage: The geometric progression of frequencies ensures coverage across multiple distance scales.

Remote attenuation effects: Phase misalignment across frequency bands naturally induces positional similarity decay with increasing relative distance.

### 4. Extensibility to N-Dimensional Vectors

    - 
    
    Show that RoPE can be applied blockwise using the frequency function to pairs of dimensions of the token embedding vector
      by simply instead of viewing $\mathbf{q}=\left(q_1, q_2, q_3, q_4, \ldots, q_d\right)$
      as a $d$ -dimensional real vector we view it as $\mathbf{q}=\left(q_1+i q_2, q_3+i q_4, \ldots q_{d-1}+i q_d\right) \in \mathbb{C}^{d / 2}$.



### 5. Remote Attenuation

    - Show that it can be applied per block, or whatever
    - Show that RoPE's per-band rotations enhance the sinusoidal PE's 
      natural distance decay by inducing phase misalignment, causing 
      destructive interference in the dot product.
    - 


Su's core contribution is demonstrating that applying a specific rotational transformation (absolute positioning) results in an inner product that depends only on the relative position.
* He presents the key formula: $\langle \boldsymbol{q}_m e^{im\theta}, \boldsymbol{k}_n e^{in\theta} \rangle = \text{Re}[\boldsymbol{q}_m \boldsymbol{k}_n^* e^{i(m-n)\theta}]$.



The frequency `θ` used in Rotary Position Embedding (RoPE) is not calculated dynamically; instead, it's a fixed value based on the scheme from the original Sinusoidal position encoding used in the "Attention is All You Need" paper.

***

### Frequency Formula

The value for each frequency `θ_i` is decided by a predefined formula:
$$\theta_i = 10000^{-2i/d}$$
Where:
* `i` is the index for each 2D pair of dimensions, ranging from `0` to `d/2 - 1`.
* `d` is the total dimension of the embedding vector (e.g., 768).

This means that for a high-dimensional vector, the frequency is applied in pairs of two dimensions, and each pair gets a different, geometrically decreasing frequency.

***

### Rationale for the Choice

This specific formula is chosen for two main reasons:

1.  **Remote Attenuation**: This choice of frequencies provides a desirable property where the inner product result between two tokens tends to decay as their relative distance increases. This is beneficial for models to weigh closer tokens more heavily.
2.  **Empirical Stability**: The author of RoPE, Jianlin Su, experimented with making the `θ` values trainable parameters. He found that they were not significantly updated during training, which led to the decision to simply keep them fixed according to the original formula.