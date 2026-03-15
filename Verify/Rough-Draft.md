Despite its central role in transformer models, positional encoding remains unsettled beyond one-dimensional sequence tasks. 

While Rotary Positional Encoding (RoPE) has become the canonical choice for autoregressive $1 \text{D}$ modeling, no comparable consensus exists in $2 \text{D}$, $3 \text{D}$, or $4 \text{D}$ settings. 

Instead, a proliferation of positional schemes is available, including rasterized sequences, axial spaces, learned embeddings, and their variants. 



This should be unsurprising, as success in any given single-task setting for a sufficiently scaled transformer is relatively independent of positional encoding. 

A proliferation of positional schemes including axial spaces, rasterized sequences, learned embeddings, and their variants are hand-selected, designed, and engineered, for specific domains, tasks, and benchmarks prior to deployment.

the Fortunately, or not, a sufficiently large transformer 


From a proliferation of positional schemes—including axial spaces, rasterized sequences, learned embeddings, and their variants—practitioners typically select a base design and further adapt and engineer it for the domain, task, and benchmark at hand.

In part this is caused by the transformer's power. 
Instead, a proliferation of positional schemes exists, including axial spaces, rasterized sequences, learned embeddings, and their variants.

Typically, a major design is selected, and then retrofitted and further engineered for specific domains, tasks, and benchmarks prior to deployment.

naturally incurring major human capital costs. 



Higher-dimensional transformers instead rely on a proliferation of learned embeddings, rasterized sequences, and hand-engineered positional schemes tailored to specific domains, tasks, or benchmarks. 


We introduce a unified positional encoding method for transformers operating over arbitrary-dimensional inputs that scores attention using the Minkowski inner product under a $(−,+,+,+)$ metric. The method applies without task-specific redesign across 1D, 2D, 3D, and 4D problems and serves as a drop-in replacement for existing encodings. Across evaluations, it matches RoPE on 1D tasks and Axial RoPE on specialized 2D, 3D, and 4D benchmarks without adjustment or tuning. By allowing models to operate directly on data in its native dimensional form, our approach reduces positional-engineering overhead and provides a simpler foundation for scalable multimodal modeling.



---
Despite its central role in transformer models, positional encoding remains unsettled beyond one-dimensional sequence tasks. While Rotary Positional Encoding (RoPE) has become the canonical choice for autoregressive $1 \text{D}$ modeling, a proliferation of positional schemes including axial spaces, rasterized sequences, and learned embeddings, is used in $2 \text{D}$, $3 \text{D}$, or $4 \text{D}$ settings. 

In part, this is because RoPE was hand designed and engineered for 1D tasks, while are hand-selected, designed, and engineered, for specific domains, tasks, and benchmarks



no comparable consensus exists in $2 \text{D}$, $3 \text{D}$, or $4 \text{D}$ settings. 

A proliferation of  prior to deployment.



---

Despite its central role in transformer models, positional encoding remains unsettled beyond one-dimensional sequence tasks. While Rotary Positional Encoding (RoPE) has become the canonical choice for autoregressive $1$D modeling, no comparable consensus exists in $2$D, $3$D, or $4$D settings. Higher-dimensional transformers instead rely on a proliferation of learned embeddings, rasterized sequences, and hand-engineered positional schemes tailored to specific domains, tasks, or benchmarks. We introduce a unified positional encoding method for transformers operating over arbitrary-dimensional inputs that scores attention using the Minkowski inner product under a $(-,+,+,+)$ metric. The method applies without task-specific redesign across $1$D, $2$D, $3$D, and $4$D problems and serves as a drop-in replacement for existing encodings. Across evaluations, it matches RoPE on $1$D tasks and Axial RoPE on specialized $2$D, $3$D, and $4$D benchmarks without adjustment or tuning. By allowing models to operate directly on data in its native dimensional form, our approach reduces positional-engineering overhead and provides a simpler foundation for scalable multimodal modeling.

---

Positional encoding remains one of the least unified components of transformer architectures beyond one-dimensional sequence modeling. Although RoPE has emerged as the canonical solution for autoregressive $1$D tasks, higher-dimensional settings remain fragmented across learned embeddings, rasterized sequences, and a wide range of hand-engineered methods designed for particular domains, tasks, and benchmarks. We present a unified positional encoding method for transformers over arbitrary-dimensional inputs based on attention scores computed with the Minkowski inner product under a $(-,+,+,+)$ metric. The method transfers directly across $1$D, $2$D, $3$D, and $4$D settings with no task-specific modifications and can be used as a drop-in replacement for existing encodings. It matches RoPE in $1$D and Axial RoPE in $2$D, $3$D, and $4$D benchmarks without tuning, while removing the need to manually redesign positional structure for each new modality. This yields a more general and practical foundation for multimodal transformers operating over text, spatial data, and spacetime.

---


No, what I'm saying is, what if the context/problem is:

1. No canonical approach above 1D

2. Tons of methods/approaches/hacks for positional encoding in various dimensions have been developed for the transformer architecture, yet many researchers still use learned embeddings or a rasterized sequences.

Then what if the gap/limitation is:

1. many of these methods are hand crafted, hand engineered, designed solutions/patches either for specific domains or specific tasks or benchmarks, and may not transfer/generalize well for different ones

2. This is because a generalized method would require serious investment, and SOTA can be achieved with a sufficiently sized transformer even if your positional encoding isn't great. In fact, you don't even need your positional encoding if you have padding! (it's a real study, trust me)

Contribution/Method

1. a single positional encoding method that can be applied to any dimension of any task in any domain that uses the minkowski inner product with a (-,+,+,+) metric to score attention

Key Results

1. Matches RoPEs performance in 1D tasks, and Axial RoPE's performance in 2D, 3D, and 4D specific tasks, with no adjustment or tuning necessary. Drop in replacement.

Impact

1. Likely to lead to better multimodal models with coherent inner models of the world

2. Saves engineers time spent translating tasks into positional structures the model understands. Anyone can now describe the task in its natural state.


---









$$
\left(R_m q\right)^{\top}\left(R_n k\right)=q^{\top} R_{n-m} k = \left(R_{m-n} q\right)^{\top} k
$$

The intuition is:

- rotate both by absolute positions $m, n$,
- or keep $q$ fixed and rotate $k$ by $n-m$,
- or keep $k$ fixed and rotate $q$ by $m-n$, 

and you get the same dot product.

$$
R_m^{\top}=R_m^{-1}=R_{-m}
$$
$$
R_a R_b=R_{a+b}
$$


# RoPE Demo — Proof Obligations

To validate that a RoPE implementation faithfully realizes the RoPE construction, verify the exact algebraic identities that define the method. The first four items are core correctness properties. The fifth is a property of the standard frequency schedule used in practice.

---

### 1. Relative-Position Dependence from Absolute Rotations

For a single 2D block, identify $q=(q_0,q_1)$ and $k=(k_0,k_1)$ with the complex numbers  
$$
z_q = q_0 + i q_1, \qquad z_k = k_0 + i k_1.
$$

Let $R_m(\theta)$ denote rotation by angle $m\theta$. A correct RoPE implementation should satisfy

$$
\langle R_m(\theta) q,\; R_n(\theta) k\rangle
=
\operatorname{Re}\!\left[z_q z_k^{*} e^{i(m-n)\theta}\right]
=
q^\top R_{n-m}(\theta) k.
$$

This is the defining RoPE identity: the transform is applied using the absolute positions $m$ and $n$, but the resulting attention score depends on position only through the relative offset $m-n$.

```python
np.allclose(
    np.dot(rotate_2d(q, m, theta), rotate_2d(k, n, theta)),
    rope_score_2d(q, k, m - n, theta)
)
```

---

### 2. Orthogonality and Norm Preservation

Each 2D rotary block is an orthogonal matrix:

$$
R_m(\theta)^\top R_m(\theta)=I.
$$

Therefore,

$$
\|R_m(\theta)q\|_2=\|q\|_2.
$$

RoPE changes only the phase/orientation of each 2D pair; it does not change vector length.

```python
np.allclose(
    np.linalg.norm(q),
    np.linalg.norm(rotate_2d(q, m, theta))
)
```

---

### 3. Joint-Shift Invariance

For fixed content vectors $q$ and $k$, the rotary attention score is invariant under a common positional shift $d$:

$$
\langle R_m q,\; R_n k\rangle
=
\langle R_{m+d} q,\; R_{n+d} k\rangle.
$$

Equivalently,

$$
R_{m+d}^\top R_{n+d}
=
R_m^\top R_n
=
R_{n-m}.
$$

This is the precise sense in which RoPE encodes relative rather than absolute position inside the attention logit.

```python
np.allclose(
    np.dot(rotate_2d(q, m, theta), rotate_2d(k, n, theta)),
    np.dot(rotate_2d(q, m + d, theta), rotate_2d(k, n + d, theta))
)
```

---

### 4. Higher-Dimensional Block-Diagonal Construction

For even embedding dimension $d$, RoPE acts on coordinate pairs. Equivalently, view

$$
q =
(q_0+i q_1,\; q_2+i q_3,\; \dots,\; q_{d-2}+i q_{d-1})
\in \mathbb{C}^{d/2}.
$$

The full rotary operator at position $m$ is

$$
R_m
=
\operatorname{diag}\!\big(
R(m\theta_0),\;
R(m\theta_1),\;
\dots,\;
R(m\theta_{d/2-1})
\big),
$$

where each $R(m\theta_i)$ is a $2\times2$ rotation block.

A correct implementation should match this block-diagonal operator, even if it uses the more efficient paired-coordinate formula instead of explicitly materializing the matrix.

---

### 5. Standard Frequency Schedule: Multiscale Coverage and Long-Range Phase Cancellation

Using the standard RoPE angular frequencies

$$
\theta_i = 10000^{-2i/d}
$$

assigns different coordinate pairs different rotation rates, so the representation spans multiple positional scales.

Under this multi-frequency schedule, larger relative offsets tend to produce more phase cancellation across coordinate pairs, which is the source of the “remote attenuation” discussion in RoPE.

More precisely, the claim is not that every individual attention score decays monotonically with distance. The claim is that, under the standard geometric frequency schedule, the multi-frequency rotary sum tends to exhibit greater cancellation at larger offsets.

---

# MonSTERs Demo — Proof Obligations

For the case $d = 4$, we consider two embedding vectors $\boldsymbol{X}_q$ and $\boldsymbol{X}_k$ corresponding to a query and a key, located at spacetime positions $M^\mu$ and $N^\mu$, respectively:

$$
M^\mu = [t_m, x_m, y_m, z_m], \quad
N^\mu = [t_n, x_n, y_n, z_n].
$$

According to Eq. (1), their position-encoded counterparts are

$$
\begin{aligned}
\boldsymbol{q}_m &= f_q(\boldsymbol{X}_q, m), \\
\boldsymbol{k}_n &= f_k(\boldsymbol{X}_k, n),
\end{aligned}
$$

where the subscripts $m$ and $n$ indicate that the embeddings are encoded using positional information associated with the spacetime locations $M^\mu$ and $N^\mu$, respectively.

---

Let the Minkowski inner product be defined by

$$
\langle q, k \rangle_\eta = q^\top \eta k,
$$

where

$$
\eta=\left[\begin{array}{cccc}
-1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{array}\right].
$$

Assume there exists a function $g$ that defines the inner product between vectors produced by $f_{\{q,k\}}$:

$$
\boldsymbol{q}_m^{\top} \boldsymbol{k}_n=\left\langle f_q\left(\boldsymbol{x}_m, m\right), f_k\left(\boldsymbol{x}_n, n\right)\right\rangle=g\left(\boldsymbol{x}_m, \boldsymbol{x}_n, n-m\right),
$$

$$
\langle \boldsymbol{q}_M, \boldsymbol{k}_N \rangle_\eta
=
\left\langle f_q(\boldsymbol{X}_q, M), f_k(\boldsymbol{X}_k, N) \right\rangle_\eta
=
g\!\left(\boldsymbol{X}_q, \boldsymbol{X}_k, (N-M)^\top \eta (N-M)\right).
$$

We further require the following initial condition to hold:

$$
\begin{aligned}
\boldsymbol{q} &= f_q(\boldsymbol{X}_q, 0), \\
\boldsymbol{k} &= f_k(\boldsymbol{X}_k, 0).
\end{aligned}
$$




























Let the Minkowski metric be defined as

$$
\eta=\left[\begin{array}{cccc}
-1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{array}\right].
$$

For spacetime coordinates $M, N \in \mathbb{R}^4$, the Minkowski inner product is defined by

$$
\langle M, N \rangle_\eta = M^\top \eta N.
$$

Assume there exists a function $g$ that defines the inner product between vectors produced by $f_{\{q,k\}}$:

$$
\boldsymbol{q}_M^\top \boldsymbol{k}_N
=
\left\langle f_q(\boldsymbol{X}_q, M), f_k(\boldsymbol{X}_k, N) \right\rangle
=
g\!\left(\boldsymbol{X}_q, \boldsymbol{X}_k, M^\top \eta N\right).
$$

We further require the following initial condition to hold:

$$
\begin{aligned}
\boldsymbol{q} &= f_q(\boldsymbol{X}_q, 0), \\
\boldsymbol{k} &= f_k(\boldsymbol{X}_k, 0).
\end{aligned}
$$




---

Assume that there exists a function $g$ that defines the inner product between vectors produced by $f_{\{q, k\}}$ :

$$
\boldsymbol{q}_m^{\top} \boldsymbol{k}_n=\left\langle f_q\left(\boldsymbol{x}_m, m\right), f_k\left(\boldsymbol{x}_n, n\right)\right\rangle=g\left(\boldsymbol{x}_m, \boldsymbol{x}_n, n-m\right),
$$

we further require below initial condition to be satisfied:

$$
\begin{aligned}
\boldsymbol{q} & =f_q\left(\boldsymbol{x}_q, 0\right), \\
\boldsymbol{k} & =f_k\left(\boldsymbol{x}_k, 0\right),
\end{aligned}
$$



Assume there exists a function $g$ that defines the Minkowski product between vectors produced by $f_{\{q,k\}}$:

$$
\boldsymbol{q}_M^{\top}\boldsymbol{k}_N
=
\left\langle
f_q(\boldsymbol{X}_q, M),
f_k(\boldsymbol{X}_k, N)
\right\rangle
=
g(\boldsymbol{X}_q,\boldsymbol{X}_k, N-M),
$$

where $M,N \in \mathbb{R}^4$ denote spacetime coordinates.

We further require the following initial condition to hold:

$$
\begin{aligned}
\boldsymbol{q} &= f_q(\boldsymbol{X}_q, 0), \\
\boldsymbol{k} &= f_k(\boldsymbol{X}_k, 0),
\end{aligned}
$$



For a 4-vector $v^\mu = [t, x, y, z]$, its covector (dual) $v_\mu$ is obtained by lowering the index with the Minkowski metric:

$$
v_\mu = \eta_{\mu\nu} v^\nu
$$

Using the $(-,+,+,+)$ convention, the Minkowski metric signature is defined as

$$
\eta=\left[\begin{array}{cccc}
-1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{array}\right]
$$


$$
M=[t_m, x_m, y_m, z_m], \quad N=[t_n, x_n, y_n, z_n]
$$



$$
\eta_{\mu\nu} =
\begin{pmatrix}
-1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
= \mathrm{diag}(-1,+1,+1,+1)
$$

If

$$
v^\mu = [t, x, y, z],
$$

then

$$
v_\mu = [-t, x, y, z].
$$











For a 4-vector $v^\mu=[t, x, y, z]$, its covector, or dual, $v_\mu$, can be found by lowering the index with the Minkowski metric:

$$
\eta = (-,+,+,+)
$$
Therefore,

$$
v_\mu=\eta v^\mu
$$




the dual usually means the corresponding covector $v_\mu$, found by lowering the index with the metric:

$$
v_\mu=g_{\mu \nu} v^\nu
$$


So it depends on the metric.
In special relativity with the common metric signature $(+,-,-,-)$,

$$
g_{\mu \nu}=\left(\begin{array}{cccc}
1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & -1
\end{array}\right)
$$

then

$$
[t, x, y, z] \rightarrow[t,-x,-y,-z]
$$


If instead you use the $(-,+,+,+)$ convention, then

$$
[t, x, y, z] \rightarrow[-t, x, y, z]
$$


So the answer is:
- with signature $(+,-,-,-):[t,-x,-y,-z]$
- with signature $(-,+,+,+):[-t, x, y, z]$

If by "dual" you meant Hodge dual, that is something different.


---


Ok, but you see the problem I am having is that when the numbers input to cosh and sinh are much greater than 1 the values blow up and we have overflow and numerical issues. Things become unstable. 

So what I'm suggesting is, first, let us set $c = 1$. 

That means time and length are being measured in the same units. Since $c$ normally converts between them,

$$
c=299,792,458 \mathrm{~m} / \mathrm{s},
$$

setting $c=1$ means you are choosing units so that 1 unit of time = 1 unit of length.

Second, if we decide to represent both in meters, then 1 meter in space is 1, and 1 meter of time becomes:

$$
1 \mathrm{~m} \text { of time }=\frac{1}{c} \mathrm{~s} .
$$

and since,

$$
c=299,792,458 \mathrm{~m} / \mathrm{s},
$$

this becomes:

$$
1 \mathrm{~m} \text { of time }=\frac{1}{299,792,458} \mathrm{~s} .
$$

In other words:

$$
\begin{aligned}
& 1 \mathrm{~s} \rightarrow 299,792,458 \mathrm{~m} \\
& 1 \mathrm{~m} \text { (of time) } \rightarrow 3.33564 \times 10^{-9} \mathrm{~s}
\end{aligned}
$$

IN decimal form this becomes roughly,

$$
\frac{1}{299{,}792{,}458}
$$

seconds is approximately

$$
3.33564 \times 10^{-9}\ \text{s}.
$$

In ordinary decimal form this is roughly

$$
0.00000000333564\ \text{s}.
$$

So:

* **1 meter of “time”** (when (c=1) and time is expressed in meters)
* corresponds to about **3.3 nanoseconds**.

This is consistent with the physical meaning: light travels **1 meter in about 3.3 ns**.

---

Therefore, to enter equivalent "steps" in meters for both space and time, we must first take the time unit of meters, multiplied by our steps. This will prevent our overflow problems with sinh, and cosh. But unfortunately, comes at a precision cost. 

I need you to find a way to mitigate that cost in my code. What are a variety of workarounds we can use? One might be to just use approximate values, or rounded values to the nearest complete precision discrete step. An alternative is using a larger float value. Explore what other scientists/physicists do in this case.
