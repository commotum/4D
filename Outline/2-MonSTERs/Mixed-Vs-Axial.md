**Short answer:**  
**RoPE-Mixed** is the *mixed-frequency 2D rotary embedding* introduced in **Heo et al. (2024), *Rotary Position Embedding for Vision Transformer***.  
It is a 2D generalization of RoPE that **uses *learned* frequency pairs for both x- and y-axes simultaneously**, so each head applies rotations of the form:

$$
R(n) = e^{\,i(\theta_t^x p_x + \theta_t^y p_y)}
$$

rather than separating the axes as in standard **2D Axial RoPE**.

Below is the exact explanation with supporting evidence from the paper.

---

## What RoPE-Mixed actually is

In the *Rotary Position Embedding for Vision Transformer* paper:

### 1\. Axial 2D RoPE (the older method)

Axial 2D RoPE applies **independent 1D RoPE rotations** along x and y:

$$
R_x(x),\; R_y(y)
$$

with frequencies reused from 1D RoPE, just split across dimensions.

The paper explicitly states this construction **cannot encode diagonal directions**:

- axial frequency “ **is unable to handle diagonal directions since the frequencies only depend on a single axis**.”

and that axial RoPE relative phases:

- “ **cannot be converted to mixed frequency** ” of the form θₓ·Δx + θᵧ·Δy.

---

### 2\. RoPE-Mixed (the improved version)

RoPE-Mixed is the paper’s proposed solution:  
**Rotate with a *learned* linear combination of x and y coordinates.**

They define the 2D rotation as:

$$
\mathbf{R}(n,t) = e^{i(\theta_t^x p^x_n + \theta_t^y p^y_n)}
$$

This gives each channel its own **(θₓ, θᵧ)** pair. These are *learned parameters*.

Then the relative phase becomes:

$$
e^{\,i(\theta_t^x (p_x^n - p_x^m) + \theta_t^y (p_y^n - p_y^m))}
$$

This allows:

- diagonal directions
- arbitrary orientations
- richer 2D geometry
- head-specific learned frequency structure

They emphasize:

- RoPE-Mixed is a **generalization** of axial RoPE (recovering axial when θₓ or θᵧ = 0).
- It solves the diagonal-blindness problem.

---

### 3\. The paper contrasts RoPE-Mixed vs Axial RoPE experimentally

Across ViT-S/B/L and Swin-T/S/B:

- **RoPE-Mixed consistently outperforms RoPE-Axial** in both interpolation and extrapolation.  
	Example figures: multi-resolution curves on pages showing substantial margins at 320–512 resolution.

RoPE-Mixed is explicitly identified as the stronger model in:

- multi-resolution ImageNet
- object detection (DINO-ViTDet)
- ADE20k segmentation

---

## TL;DR definition

### RoPE-Mixed = 2D RoPE with learned mixed-axis frequencies

Instead of rotating by:

$$
e^{i\theta_x p_x}, \; e^{i\theta_y p_y}
$$

it rotates by:

$$
e^{\,i(\theta^x p_x + \theta^y p_y)}
$$

for every frequency channel.

This lets the model express **arbitrary 2D directions**, not only horizontal + vertical.

---

If you'd like, I can also produce:

- a table comparing Axial vs Mixed vs 1D RoPE
- a visual explanation of the rotation geometry
- a summary of how other papers (HARoPE, ComRoPE, YaRN, etc.) relate to RoPE-Mixed.