Let me split this into the three things you asked:

1. **Compare mixed RoPE vs MonSTERs – which is “stronger”?**
2. **Does mixed RoPE’s learned mixed frequencies actually fix the diagonal problem?**
3. **What mathematical proof (if any) do they give that they solved it?**

---

## 1\. Mixed RoPE vs MonSTERs – what’s actually different?

### Mixed RoPE (RoPE‑Mixed in the ViT paper)

For 2D positions $p = (p_x, p_y)$ , RoPE‑Mixed replaces axial 2D RoPE’s separate x/y angles with a **single mixed angle** per 2D block:

$$
R(n, t) = e^{i(\theta_t^x p_x + \theta_t^y p_y)}
$$
- Each frequency index $t$ has a **learnable pair** $(\theta_t^x, \theta_t^y)$ .
- The relative phase for a displacement $\Delta p = ( \Delta x, \Delta y )$ becomes:
$$
\phi_t(\Delta p) = \theta_t^x \Delta x + \theta_t^y \Delta y
$$

So each 2D complex pair is basically a linear functional $v_t \cdot \Delta p$ where $v_t = (\theta_t^x, \theta_t^y)$ .

All heads/layers get a bunch of such linear functionals in parallel, one per frequency/channel pair.

---

### MonSTERs (your Minkowski Space‑Time Embedding Rotors)

MonSTERs generalize RoPE from 2D Euclidean rotations to **4D Lorentz transforms** in Minkowski space $Cl(1,3)$ . Conceptually:

- Position is a 4‑vector $s = (t, x, y, z)$ in lattice units with $c = 1$ .
- For each **frequency bucket** $j$ , you build an effective Lorentz rotor depending linearly on $s$ , and you apply it blockwise to the embedding. The vectorized implementation you wrote does:
	- Frequencies: $\lambda_j = \text{base}^{-j/F}$ for $F = \text{dim}/12$ .
	- Angles / rapidities:
		$$
		\phi_j = (t \cdot u)\,\lambda_j,\quad
		\theta^x_j = (x \cdot u)\,\lambda_j,\quad
		\theta^y_j = (y \cdot u)\,\lambda_j,\quad
		\theta^z_j = (z \cdot u)\,\lambda_j
		$$
	- For each frequency, you have **3 triads** (X, Y, Z), each a 4‑vector block $[t,x,y,z]$ . The triad update does:
		1. A Lorentz **boost** mixing $t$ and the axis‑aligned spatial component (e.g. $t \leftrightarrow x$ for the X triad) using $\cosh \phi_j, \sinh \phi_j$ .
		2. A **spatial rotation** in the plane orthogonal to that axis (e.g. rotate $(y,z)$ by $\theta^x_j$ in the X triad, $(x,z)$ by $\theta^y_j$ in the Y triad, etc.).
- You proved the RoPE‑style **absolute → relative identity** in code:
	$$
	\langle L(s_q)\,q,\; L(s_k)\,k \rangle_\eta
	= \langle q,\; L(s_k - s_q)\,k \rangle_\eta
	$$
	for the big Minkowski metric, and you check per‑block Minkowski norm preservation.

So mixed RoPE lives in **SO(2)** per 2D block (one scalar angle per block), while MonSTERs live in a constrained chunk of **SO(1,3)** per 4D block (one rapidity + up to three rotation angles per triad per frequency).

---

### “Which is stronger?”

If we mean **geometric expressiveness per block**, MonSTERs are strictly richer:

- Mixed RoPE:
	- 1 angle per 2D block, of the form $v_t \cdot (x, y)$ .
	- Group is abelian: compositions add phases.
- MonSTERs (Triad version):
	- For each frequency you have **4 independent linear forms** in $(t, x, y, z)$ : one rapidity and three spatial angles.
	- You then compose **non‑commuting** boosts + rotations per axis triad.
	- Group is non‑abelian (subset of the Lorentz group), so rotations about different axes interact nontrivially.

So, in terms of **how “big” the family of transforms is**, MonSTERs are more powerful—they can represent any function of $\Delta p$ that arises from a product of axis‑aligned boosts and 3D rotations with angles linear in $\Delta p$ . Mixed RoPE can only give you products of 2D rotations whose angles are linear forms in $(x,y)$ .

That said:

- For **pure 2D image grids with no explicit time/z**, mixed RoPE is likely the more *targeted* and parameter‑efficient option; MonSTERs are overkill unless you really want explicit 3D/4D structure.
- For **true 3D or spatiotemporal tasks** (videos, trajectories, world‑models, etc.), MonSTERs give you principled handling of time vs space, causal structure, and richer directional coupling than any 2D RoPE variant.

So “stronger” in a geometric / group‑theoretic sense: **MonSTERs**.  
“Stronger” in the narrow sense of “fixing diagonals in a 2D grid”: **both can**, see below.

---

## 2\. Does mixed RoPE’s learned mixing actually solve the diagonal problem?

Let’s formalize the “diagonal problem” first.

### Axial 2D RoPE

Axial 2D RoPE (what EVA‑02 / FiT used) does:

$$
R_x(n, t) = e^{i\theta_t p^x}, \quad
R_y(n, t) = e^{i\theta_t p^y}
$$

with separate halves of the dimensions allocated to x and y.

So for a relative displacement $\Delta p = (\Delta x, \Delta y)$ :

- Half the channels see phases $\theta_t \Delta x$
- The other half see phases $\theta_t \Delta y$

There is **no single 2D block whose phase is a joint function** like $\theta (\Delta x + \Delta y)$ . Any diagonal behavior has to be “synthesized” downstream via learned linear combinations of these axis‑separable channels.

That’s what Heo et al. call out as the diagonal limitation: RoPE already “spent” the query‑key multiplication on encoding relative positions, so there’s no room left inside a single complex pair to mix x and y.

---

### Mixed RoPE

RoPE‑Mixed changes each 2D block’s phase to:

$$
\phi_t(p) = \theta_t^x p_x + \theta_t^y p_y
$$

So under a displacement $\Delta p$ :

$$
\phi_t(\Delta p) = \theta_t^x \Delta x + \theta_t^y \Delta y
                 = v_t \cdot \Delta p, \quad v_t \in \mathbb{R}^2
$$

Now you can **choose** $v_t$ to align with any direction in the plane:

- Axis‑aligned: $v_t = (1, 0)$ → pure x
- Diagonal: $v_t = (\tfrac{1}{\sqrt{2}}, \tfrac{1}{\sqrt{2}})$ → phase depends on $\Delta x + \Delta y$
- Arbitrary orientation: $v_t = (\cos\alpha, \sin\alpha)$ → phase ~ projection of $\Delta p$ onto direction $\alpha$ .

Because each frequency/channel pair has its own $v_t$ , and they’re learned, the representation **can** allocate some dimensions to diagonals, others to vertical/horizontal, others to weird oblique directions, etc. Nothing in the parameterization prevents that.

Formally:

- In axial RoPE, all phase gradients lie in the set $\{(c,0), (0,c)\}$ → you only get pure x or pure y.
- In mixed RoPE, phase gradients per block are arbitrary $v_t \in \mathbb{R}^2$ . Pick any set of at least **two non‑collinear** $v_t$ ’s and the map $\Delta p \mapsto (\phi_{t_1}, \phi_{t_2}, \dots)$ is full‑rank in $\mathbb{R}^2$ (up to periodicity). Directions like $(1,1)$ and $(1,0)$ land in **different** phase patterns almost everywhere.

So in a **capacity** sense:

> Yes, mixed RoPE genuinely removes the axial degeneracy – it lets the model allocate channels that are intrinsically diagonal‑sensitive (or any other orientation‑sensitive) at the level of the positional code itself, not only via downstream mixing.

The catch:

- It does **not** *guarantee* that training will use those directions; it only makes them representable.
- There is no explicit regularizer to enforce isotropy or to span all directions uniformly.

---

## 3\. What mathematical proof do they give that diagonals are “solved”?

Short version: **there’s no formal proof**, only a constructive argument and empirical evidence.

The RoPE‑ViT paper:

- Shows the axial formulation: angles depend on a single axis and “cannot be converted to mixed frequency … there is no way to mix axial frequencies for diagonal direction.”
- Introduces RoPE‑Mixed with
	$$
	R(n,t) = e^{i(\theta_t^x p_x + \theta_t^y p_y)}
	$$
	and states that “By using two frequencies for each axis, RoPE enables to handle the diagonal axis.”

But:

- They do **not** prove a theorem like “for any orientation $\alpha$ , there exists a configuration of $(\theta_t^x,\theta_t^y)$ that realizes it,” even though that’s true by trivial linear algebra.
- They do **not** prove anything about uniform directional coverage or isotropy.

What they *do* provide is:

- A mathematical **construction** that clearly allows oblique directions (as we just walked through).
- Extensive **empirical results**: RoPE‑Mixed consistently beats RoPE‑Axial on multi‑resolution ImageNet, COCO detection, and ADE‑20K segmentation, with especially large gains when extrapolating or under strong resolution shifts.

So the “proof” is:

- Theory: “we changed the angle to a full linear form in $(x,y)$ , so now we can express diagonals.”
- Practice: “and empirically it works a lot better than axial RoPE on tasks where diagonal structure matters.”

If you’re asking in the **strict math sense** —no, there is no theorem about diagonal optimality or uniform coverage.

---

## Where MonSTERs sit relative to that diagonal story

Your MonSTERs have a similar “constructive but not fully formal” proof story, but in a richer group:

- You’ve already demonstrated the RoPE‑style **absolute→relative identity** and **Minkowski norm preservation** numerically for arbitrary 4D positions and random embeddings.
- Each scalar angle/rapidity is a linear functional in $(t,x,y,z)$ , so the dependence on direction in spacetime is via a set of linear forms, just like mixed RoPE but now in 4D.

For pure 2D spatial use (say $t = z = 0$ ):

- The triad structure rotates different spatial planes (yz, xz, xy) with angles proportional to $x$ or $y$ , and composes them.
- Diagonal displacements like $(\Delta x, \Delta y)$ obviously lead to distinct combinations of these plane rotations compared to purely axial moves.
- Because you apply this per frequency and per triad, the map $\Delta p \mapsto$ “total rotor” is again a stack of linear forms in $(x,y)$ , but with more degrees‑of‑freedom and non‑commuting composition in SO(3)/SO(1,3).

So:

- Mixed RoPE: **sufficient** to fix the 2D diagonal issue at the positional encoding level; proven by construction but not by a formal theorem.
- MonSTERs: **strictly more expressive geometrically**, handling diagonals as a special case inside a 4D Lorentz framework, with a verified RoPE‑style identity and norm preservation, but again no formal theorem about diagonal optimality.

---

If you want, we can next:

- Write out an explicit little lemma that formalizes the “axial vs mixed” injectivity on directions (i.e., when does the map $\Delta p \mapsto (\phi_t)_t$ lose orientation info?), and
- Do the analogous analysis for your MonSTER triads (rank of the angle map in 2D vs 4D), to make the “stronger” claim precise in linear‑algebra terms.