# The Challenge

An ARC‑AGI task consists of a small **training set** (typically three‑to‑five) of input–output pixel grids, each grid expressed with integer‑encoded colours.
All training pairs share a *hidden* visual transformation.
After observing these examples, the model receives **one further input grid** whose correct output is withheld.
The objective is to **predict that withheld output** by inferring and applying the common transformation.

ARC-AGI tasks are explicitly few-shot/meta-learning problems. Each task provides 3-5 correlated input-output grid pairs (demonstrations) that share a hidden visual transformation rule. The model must infer this rule from the correlations across examples and apply it to a new test input. This requires treating the task as a holistic "episode" where inter-example relationships (e.g., how the transformation evolves or is consistent across pairs) are key.

---

# The Dataset

Our training corpus contains **millions of explicit code–image pairs**, each of which is a p5.js sketch alongside the exact image(s) it produces.

* **Single images** – sketches that draw one static picture.
* **Before/After image pairs** – sketches that apply a visual operation, yielding a *before* and *after* frame.
* **Before/After image sets** – sketches that apply the *same* transformation to multiple inputs, producing aligned before/after sets.

The dataset is organised as a *graduated curriculum*:

| Stage | New visual concepts introduced                                           |
| ----- | ------------------------------------------------------------------------ |
| 1     | Black & white primitives (points, lines, triangles, rectangles, circles) |
| 2     | Spatial layouts, colour fills/outlines                                   |
| 3     | Basic 2‑D transforms (rotation, scale, shear)                            |
| 4     | Layer operations (clipping, masking), simple textures                    |
| 5     | Fundamental 3‑D primitives (boxes, spheres)                              |
| 6     | Lighting & material effects                                              |
| 7     | Fully composed 2‑D / 3‑D scenes                                          |

This stepwise progression lets the model master elementary concepts before combining them into complex scenes.

---

# MonSTER: Minkowski‑Space Time Embedding Rotors

**MonSTER** generalises Rotary Position Embeddings (RoPE) from 2‑D rotations to full 4‑D Lorentz transformations.
Built on the real Clifford algebra **Cl(1, 3)** with signature `(+,−,−,−)`, MonSTER computes a *block‑wise* relative spacetime transform

```
R_eff^(b)(Δt,Δx,Δy,Δz)
```

for each frequency block `b` and injects it into the attention score

```
score_{ij} = Σ_b Q_i^(b)ᵀ η R_eff^(b) K_j^(b)
```

MonSTER guarantees that identical spacetime displacements always induce the same geometric modulation, enabling transformers to reason jointly over space **and** time instead of a flattened 1‑D token order.

---

# The Pipeline

The procedure below trains on a set of `n` before/after pairs and produces p5.js code that (i) recreates every pair and (ii) generalises to the unseen test input.

## 1 · Data Preparation  *(image → token dictionary)*

| Input  | Two images per pair: I\_{t=0} (before) and I\_{t=1} (after) |
| ------ | ----------------------------------------------------------- |
| Output | A dictionary mapping 4‑D positions to colour vectors        |

For every pixel `(x,y)` in an image:

```python
entry = {
    (x, y, 0): {                 # 3‑D spatial coordinate (z = 0)
        "color": (0, R, G, B),   # 4‑vector / quaternion   (0RGB)
        "position": (x, y, 0, t) # embeds time step t ∈ {0,1}
    }
}
```

> **Δt** between the two frames equals **1** and is passed verbatim to MonSTER.

---

## 2 · Colour Upscaling

A learnable matrix `W_color ∈ ℝ^{d×4}` converts 4‑vector colours into d‑dimensional tokens:

```
v_color = W_color @ [0, R, G, B]^T
```

---

## 3 · Position Embedding (MonSTER)

For each token, compute the **relative** displacement `ΔP = (Δt, Δx, Δy, Δz)` with respect to every other token and obtain the block matrix stack

```python
R_eff_blocks = get_delta_monster(Δt, Δcoords, num_blocks=B)
```

These matrices modulate the **Q–K** dot product inside multi‑head attention.

---

## 4 · Transformer Encoding

Feed all tokens (from *both* frames) into a MonSTER‑enhanced, causal transformer:

```python
encodings = monster_transformer(tokens)  # shape: [T, d]
```

---

## 5 · Autoregressive Code Generation

Starting from `<START_P5>`, decode until `<END_P5>` or a length cap:

```python
p5_code = monster_transformer.generate(
    encodings, stop_token="<END_P5>", max_length=2048
)
```

---

## 6 · Execution & Self‑Correction Loop

1. **Run** the generated sketch in a sandboxed Node + p5.js runtime.
2. **Outcome cases**

   | Case               | Action                                                                  |
   | ------------------ | ----------------------------------------------------------------------- |
   | Runtime error      | Re‑enter **all tokens** *plus* the traceback, regenerate.               |
   | Runs ☑ but image ✗ | Compute similarity (SSIM / LPIPS); re‑enter tokens + score, regenerate. |
   | Runs ☑ and image ☑ | Mark *success*; persist code.                                           |

All attempts are logged:

```json
{
  "pair_id": k,
  "input_images": ["img_t0.png", "img_t1.png"],
  "generated_code": "...",
  "ground_truth_code": "...",
  "success": true,
  "similarity": 0.981
}
```

---

## 7 · Iterate over n Pairs

Repeat Steps 1–6 for every before/after pair, accumulating a **success\_set** of verified sketches.

---

## 8 · Meta‑Training on Code

Tokenise all successful sketches and feed them back into the transformer to learn the unifying transformation logic:

```python
unified_code = monster_transformer.generate(
    code_tokens_batch,
    prompt="Unify transformation logic"
)
```

---

## 9 · Test‑Time Inference

Supply the **unseen** test input grid (t=0 only).
Run the **unified** code to produce the predicted output grid and submit.

---

# Suggested Implementation Stack

| Layer              | Technology                                 |
| ------------------ | ------------------------------------------ |
| Data I/O           | Python + PIL / OpenCV                      |
| Core model         | JAX + Flax / Haiku                         |
| Colour tokenizer   | `nn.Linear(4, d)` or quaternion projection |
| Position embedding | **MonSTER**                                |
| Execution sandbox  | Node + p5.js via Puppeteer                 |
| Visual evaluation  | SSIM + LPIPS                               |
| Logging            | JSONL or SQLite                            |



| Grid Data         |  Time Step  |
|-------------------|-------------|
| Example Input  1  | Time Step 1 |
| Example Output 1  | Time Step 2 |
| Example Input  2  | Time Step 3 |
| Example Output 2  | Time Step 4 |
| Example Input  3  | Time Step 5 |
| Example Output 3  | Time Step 6 |
| Test Input        | Time Step 7 |
| Test Output       | Time Step 8 |