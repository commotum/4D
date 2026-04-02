To extend Rotary Position Embedding (RoPE) to 4-dimensional spacetime, we work with 4-vectors equipped with the Minkowski metric
$$
\eta=\operatorname{diag}(-1,+1,+1,+1),
$$
so positions are written as $s=(t,x,y,z)$ and each transformed embedding block lives in the same signature. The objective remains RoPE’s core property: encode absolute positions so that the similarity between transformed queries and keys depends only on their relative displacement. In the implementation here, that relative structure is realized with Lorentz-compatible block transforms rather than ordinary Euclidean 2D rotations, and the preserved quadratic form is
$$
s^\top \eta s=-t^2+x^2+y^2+z^2.
$$

As in standard RoPE, frequencies are distributed across the embedding, but here the embedding is partitioned into 12-dimensional frequency buckets, each made of three separate 4D blocks aligned with the $x$, $y$, and $z$ spatial axes. For each frequency
$$
\lambda_j=\mathrm{base}^{-j/F},
$$
the code defines one temporal rapidity
$$
\phi_j=(Lt)\lambda_j
$$
and three spatial angles
$$
\theta_j^x=x\lambda_j,\qquad \theta_j^y=y\lambda_j,\qquad \theta_j^z=z\lambda_j.
$$
Each 4D block then applies two steps: first a boost mixing time with its aligned spatial coordinate, and second a rotation in the orthogonal spatial 2-plane.

The three axis-aligned 4D blocks apply identical boost–rotation structure, differing only in which spatial coordinate is paired with time and which orthogonal spatial plane is rotated:

$$
M_x(\phi,\theta_x)=
\begin{bmatrix}
\cosh\phi & -\sinh\phi & 0 & 0 \\
-\sinh\phi & \cosh\phi & 0 & 0 \\
0 & 0 & \cos\theta_x & -\sin\theta_x \\
0 & 0 & \sin\theta_x & \cos\theta_x
\end{bmatrix}
$$

$$
M_y(\phi,\theta_y)=
\begin{bmatrix}
\cosh\phi & 0 & -\sinh\phi & 0 \\
0 & \cos\theta_y & 0 & -\sin\theta_y \\
-\sinh\phi & 0 & \cosh\phi & 0 \\
0 & \sin\theta_y & 0 & \cos\theta_y
\end{bmatrix}
$$

$$
M_z(\phi,\theta_z)=
\begin{bmatrix}
\cosh\phi & 0 & 0 & -\sinh\phi \\
0 & \cos\theta_z & -\sin\theta_z & 0 \\
0 & \sin\theta_z & \cos\theta_z & 0 \\
-\sinh\phi & 0 & 0 & \cosh\phi
\end{bmatrix}
$$

These are the explicit matrices implemented by the code’s broadcasted update rules, rather than a generic STA rotor construction. The resulting map preserves the Minkowski form on each 4D block under $\eta$, and the full embedding uses a block-diagonal repetition of this metric in the inner product. That gives the intended RoPE-style identity: applying absolute transforms to $q$ and $k$ yields the same Minkowski inner product as leaving $q$ unchanged and applying the relative transform built from $s_k-s_q$ to $k$.