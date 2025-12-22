Positions are denoted within zero-centered Cartesian lattice with half-integer offsets when the origin is interstitial.
Uses signed Cartesian indices (x, y) with (0,0) always the origin, even when no cell exists there.
Same for problems/domains of all dimensions 1D, 2D, 3D, 4D
The grid is embedded in $\mathbb{R}^2$
The origin is always fixed at $(0,0)$
Each cell is identified by the coordinates of its center
Axes:
- x increases to the right (columns)
- y increases upward (rows)

So every cell has coordinates:

$$
(x, y) \in \mathbb{Z} \times \mathbb{Z} \quad \text { or } \quad(x, y) \in\left(\mathbb{Z}+\frac{1}{2}\right) \times\left(\mathbb{Z}+\frac{1}{2}\right)
$$

depending on parity.

Let:
- $\mathrm{W}=$ number of columns
- $\mathrm{H}=$ number of rows
- i = column index (0-based, left → right)
- j = row index (0-based, bottom → top)


$x=c-\frac{W-1}{2}, \quad y=\frac{H-1}{2}-r$

$(x, y)_{\text {centered }}$


Where:
- $W=$ number of columns, $H=$ number of rows
- $c \in\{0, \ldots, W-1\}$ left → right
- $r \in\{0, \ldots, H-1\}$ top → bottom (matrix-style rows)


Example $3 \times 3$ :

(-1, 1)  (0, 1)  (1, 1)
(-1, 0)  (0, 0)  (1, 0)
(-1,-1)  (0,-1)  (1,-1)


Example $4 \times 4$ :

(-1.5,  1.5)  (-0.5,  1.5)  (0.5,  1.5)  (1.5,  1.5)
(-1.5,  0.5)  (-0.5,  0.5)  (0.5,  0.5)  (1.5,  0.5)
(-1.5, -0.5)  (-0.5, -0.5)  (0.5, -0.5)  (1.5, -0.5)
(-1.5, -1.5)  (-0.5, -1.5)  (0.5, -1.5)  (1.5, -1.5)


Example $4 \times 3$ :

(-1.5, 1)  (-0.5, 1)  (0.5, 1)  (1.5, 1)
(-1.5, 0)  (-0.5, 0)  (0.5, 0)  (1.5, 0)
(-1.5,-1)  (-0.5,-1)  (0.5,-1)  (1.5,-1)


Example $\mathbf{3} \times \mathbf{4}$ :

(-1,  1.5)  (0,  1.5)  (1,  1.5)
(-1,  0.5)  (0,  0.5)  (1,  0.5)
(-1, -0.5)  (0, -0.5)  (1, -0.5)
(-1, -1.5)  (0, -1.5)  (1, -1.5)


One-line parity rule (quick reference)
- If W odd $\rightarrow \mathrm{x}$ is integer; if W even $\rightarrow \mathrm{x}$ is half-integer
- If H odd $\rightarrow \mathrm{y}$ is integer; if H even $\rightarrow \mathrm{y}$ is half-integer
- $(0,0)$ exists as a cell center only when $W$ and $H$ are both odd

---

### Zero-Centered Cartesian Cell Coordinates

We represent grid positions using a zero-centered Cartesian coordinate system defined over cell centers. The grid is embedded in ℝ² with a fixed geometric origin at (0, 0), which is preserved even when no cell center lies exactly at the origin (e.g., when one or both grid dimensions are even). This representation is dimension-agnostic and extends naturally to 1D, 2D, 3D, and higher-dimensional domains.

Each cell is identified by the coordinates of its center. The horizontal axis x increases to the right (corresponding to columns), and the vertical axis y increases upward (corresponding to rows). All coordinates are expressed in signed Cartesian form (x, y).

Let W denote the number of columns and H the number of rows. We index the grid using matrix-style indices (r, c), where c ∈ {0, …, W−1} enumerates columns from left to right, and r ∈ {0, …, H−1} enumerates rows from top to bottom. These indices are mapped to centered Cartesian coordinates according to

x = c − (W − 1)/2
y = (H − 1)/2 − r

This mapping ensures that the origin (0, 0) corresponds to the geometric center of the grid, independent of parity.

The parity of the grid dimensions determines whether coordinates lie on integer or half-integer lattices. If W is odd, x takes integer values; if W is even, x takes half-integer values. Similarly, if H is odd, y takes integer values; if H is even, y takes half-integer values. Consequently, a cell center exists at (0, 0) if and only if both W and H are odd. When either dimension is even, the origin lies between adjacent cells, and the nearest cell centers occur at offsets of ±1/2 along the corresponding axis.

This construction yields four possible cases:

• Odd × odd grids place cell centers on ℤ² and include a central cell at (0, 0).
• Even × even grids place cell centers on (ℤ + 1/2)², with the origin lying between four cells.
• Even × odd grids yield x ∈ ℤ + 1/2 and y ∈ ℤ, with a central row but no central column.
• Odd × even grids yield x ∈ ℤ and y ∈ ℤ + 1/2, with a central column but no central row.

For example, a 3×3 grid yields integer-valued coordinates with a center cell at (0, 0). A 4×4 grid yields half-integer coordinates such as (±1/2, ±1/2) and (±3/2, ±3/2), with no cell at the origin. A 4×3 grid yields half-integer x-coordinates and integer y-coordinates, while a 3×4 grid yields integer x-coordinates and half-integer y-coordinates.

The inverse mapping from centered coordinates back to array indices is given by

c = x + (W − 1)/2
r = (H − 1)/2 − y

which yields integer indices whenever (x, y) lies on the appropriate parity-dependent lattice.

This zero-centered formulation provides a uniform coordinate system across grid sizes and dimensions, simplifies symmetry reasoning, and enables parity-independent geometric operations such as translation, reflection, and rotation.