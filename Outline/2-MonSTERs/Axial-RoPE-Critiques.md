Below is a **tight executive summary**, followed by the **actual mathematical reasons** axial RoPE has those problems. No fluff.

---

## Executive summary (what’s wrong with axial RoPE)

**Axial RoPE is limited because it factorizes multi-dimensional position into independent 1D rotations.**  
This makes it:

1. **Blind to diagonal and oblique directions**  
	– It cannot encode “move (1,1)” as a primitive direction; diagonals only emerge indirectly (and weakly).
2. **Axis-separable by construction**  
	– Position effects along x and y are encoded in disjoint subspaces and never interact at the positional level.
3. **Rigid and hand-designed**  
	– Frequencies are fixed, reused across axes, and evenly split, regardless of data anisotropy.
4. **Poor at hierarchical / multi-scale 2D structure**  
	– It encodes local offsets but not region-level or object-level spatial structure.
5. **Head- and layer-uniform**  
	– All heads see the same axial geometry, even though heads specialize differently.

In practice: axial RoPE works *okay* for local neighborhoods but underperforms whenever geometry, orientation, or extrapolation matter.

---

## The mathematical causes (why this happens)

## 1\. Definition of axial RoPE

For a 2D position $p = (x, y)$ , axial RoPE applies **independent 1D RoPE rotations**:

$$
R(x,y)
=
\begin{pmatrix}
R_x(x) & 0 \\
0 & R_y(y)
\end{pmatrix}
$$

where each $R_x, R_y \in SO(2)^{d/4}$ is a block-diagonal rotation matrix.

For queries and keys:

$$
q' = R(x_q,y_q)\,q,\quad
k' = R(x_k,y_k)\,k
$$

The attention inner product becomes:

$$
\langle q', k' \rangle
=
\langle q_x, R_x(\Delta x)\,k_x \rangle
+
\langle q_y, R_y(\Delta y)\,k_y \rangle
$$

with $\Delta x = x_q - x_k$ , $\Delta y = y_q - y_k$ .

---

## 2\. Root problem: block-diagonal separability

The rotation matrix is **block-diagonal across axes**.

That implies:

- Each 2D complex pair encodes **only one axis**.
- No positional feature ever depends on both $x$ *and* $y$ inside the same rotation.

Formally:

$$
\phi_i(x,y) \in \{\theta_i x,\ \theta_i y\}
\quad\text{but never}\quad
\theta_i^x x + \theta_i^y y
$$

So the positional phase gradients live in:

$$
\{(c,0), (0,c)\} \subset \mathbb{R}^2
$$

not in general $(c_1, c_2)$ .

---

## 3\. Why diagonals are fundamentally weak

Consider two relative displacements:

- Axis: $(\Delta x, 0)$
- Diagonal: $(\Delta x, \Delta x)$

In axial RoPE:

- Axis displacement affects **only x-channels**
- Diagonal displacement affects **x-channels and y-channels separately**

There is **no channel whose phase is proportional to**:

$$
\Delta x + \Delta y
$$

So diagonals are **not a primitive direction**; they are reconstructed only by later linear mixing of unrelated channels.

In contrast, mixed RoPE or MonSTERs have channels where:

$$
\phi(\Delta x,\Delta y) = \theta_x \Delta x + \theta_y \Delta y
$$

which makes diagonal sensitivity *intrinsic*, not emergent.

---

## 4\. Why attention can’t “fix it later”

A key subtlety:

> RoPE already uses the query–key dot product to convert absolute → relative position.

That dot product is *already spent* enforcing:

$$
R(p_q)^\top R(p_k) = R(p_q - p_k)
$$

So there is **no remaining multiplicative interaction** where x- and y-phases can mix.

Any mixing must happen **after attention**, via learned projections, which is weaker, data-hungry, and not position-equivariant.

---

## 5\. Frequency rigidity

Axial RoPE also assumes:

- Same frequency spectrum on every axis
- Even split of dimensions per axis
- Fixed base (e.g. 10 000)

Mathematically, it enforces:

$$
\theta_i^x = \theta_i^y = \theta_i
$$

which assumes isotropy and equal scale across dimensions — often false in images, grids, or spatiotemporal data.

---

## 6\. No spatial hierarchy

Because axial RoPE only encodes **pairwise offsets**, it lacks any mechanism for:

- grouping patches into regions,
- encoding “same object” vs “different object”,
- coarse vs fine spatial scale.

Mathematically, all information is encoded in:

$$
R(\Delta x, \Delta y)
$$

with no notion of *nested* or *composed* spatial structure.

---

## One-sentence takeaway

> **Axial RoPE factorizes space into independent 1D rotations; this makes diagonals, orientation, and higher-order spatial structure mathematically inaccessible at the positional-encoding level.**

That’s exactly why later work (RoPE-Mixed, HARoPE, ComRoPE, and your MonSTERs) all break axis separability in different ways.

If you want, I can also give you a **one-slide diagram explanation** or a **formal comparison table vs mixed RoPE vs MonSTERs**.

Here is a **short, executive-style summary** that sticks **only to what can be stated and justified mathematically**, with a brief preface.

---

## What axial RoPE is and why it was introduced

**Axial RoPE** is a straightforward extension of 1D RoPE to multi-dimensional data (e.g. images).  
It was introduced to:

- preserve RoPE’s key property
	$$
	R(p_q)^\top R(p_k) = R(p_q - p_k)
	$$
	so attention depends only on relative position, and
- avoid learning large 2D relative-bias tables by reusing cheap sinusoidal rotations.

To do this, axial RoPE **factorizes position across axes**.  
For a 2D position $p=(x,y)$ , it applies independent 1D RoPE rotations:

$$
R(x,y)
=
\begin{pmatrix}
R_x(x) & 0 \\
0 & R_y(y)
\end{pmatrix}
$$

with half the embedding dimensions allocated to $x$ and half to $y$ .

---

## Mathematically provable shortcomings of axial RoPE

### 1\. Axis separability (block-diagonal structure)

Axial RoPE’s rotation matrix is block-diagonal across axes.

As a consequence:

- every rotational phase depends on **exactly one coordinate**:
	$$
	\phi_i(x,y) \in \{\theta_i x,\ \theta_i y\}
	$$
- there is **no channel** whose phase depends jointly on $(x,y)$ .

This is not an implementation choice; it follows directly from the block-diagonal construction.

---

### 2\. No primitive encoding of diagonal or oblique directions

Because each phase depends on only one axis, axial RoPE cannot encode directions of the form:

$$
\theta_x x + \theta_y y
$$

within a single rotation.

Formally, the set of positional phase gradients spans only:

$$
\{(c,0), (0,c)\} \subset \mathbb{R}^2
$$

and not the full $\mathbb{R}^2$ .

Diagonal displacements $(\Delta x, \Delta y)$ therefore **do not correspond to a distinct positional phase**, but are decomposed into two independent axial effects. Any diagonal sensitivity must be reconstructed later by learned linear mixing, not by the positional encoding itself.

---

### 3\. No positional cross-axis interaction in attention

RoPE uses the query–key dot product to convert absolute position into relative position.  
In axial RoPE, this dot product decomposes additively:

$$
\langle q',k' \rangle
=
\langle q_x, R_x(\Delta x)k_x \rangle
+
\langle q_y, R_y(\Delta y)k_y \rangle
$$

There is **no multiplicative or rotational interaction between axes at the positional level**.  
Once the dot product is taken, the opportunity to mix $x$ and $y$ geometrically is gone.

This limitation follows directly from RoPE’s algebra and cannot be fixed downstream without breaking relative-position equivariance.

---

### 4\. Rigid frequency assignment

Axial RoPE enforces:

- identical frequency spectra on each axis,
- equal dimensional allocation per axis.

Mathematically, it assumes:

$$
\theta_i^x = \theta_i^y
$$

This hard-codes isotropy and equal scale across dimensions, regardless of the data. The model cannot reallocate positional capacity across axes within the RoPE mechanism itself.

---

## One-sentence takeaway

> **Axial RoPE factorizes multi-D position into independent 1D rotations; this makes cross-axis, diagonal, and oblique positional structure mathematically inaccessible at the positional-encoding level.**

That limitation is structural, not empirical, and is exactly what mixed-axis RoPE variants and MonSTER-style constructions remove.