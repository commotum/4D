---
title: "CUDA Hamilton product optimization"
source: "https://chatgpt.com/c/68d86768-a9f0-832b-ae86-fe85e43b9c6a"
author:
  - "[[ChatGPT]]"
published:
created: 2025-09-27
description: "ChatGPT helps you get answers, find inspiration and be more productive. It is free to use and easy to try. Just ask and ChatGPT can help with writing, learning, brainstorming and more."
tags:
  - "clippings"
---
I want to build a hamilton product cuda kernel for nvidia GPUs. I need you to figure out exactly when it would be best to use a matrix operations to take advantage of tensor cores, and when it's faster just to use a fused kernel of arithmetic ops. IE if I have a single quaternion value multiplied by many other quaternions, or many other quaternions by a single value, or pairwise many different pairs, but a large batch of them etc.

---

To help you determine when to use matrix operations (to leverage NVIDIA tensor cores) versus a fused arithmetic CUDA kernel for Hamilton product computation, could you please clarify the following:

1. What GPU architecture(s) are you targeting (e.g., Ampere, Hopper, Turing)?
2. Are you using FP16, TF32, FP32, or INT types for the quaternion values?
3. Do you have specific batch sizes or memory access patterns in mind (e.g., streaming vs. random access)?
4. Are you optimizing for latency (single-shot execution) or throughput (high-volume batch processing)?
5. Will this run in isolation or as part of a larger kernel or pipeline?

Once I have that, I can dive into the performance tradeoffs and implementation strategies.

---

I think FP16 and BF16? and idk about architectures? Eventually the goal is something like this: # Type and Value Embeddings Standard token-based approaches in large language models (LLMs) typically rely on constrained vocabularies due to computational and representational limitations. Common data types, such as RGB colors or 64-bit integers (int64), present an immense combinatorial challenge. For instance, the RGB color space alone contains over 16 million unique values, vastly exceeding the vocabularies used by popular models like LLaMA-3.2 (\\~128K tokens), DeepSeek-V3 (\\~129K tokens), Claude (\\~65K tokens), GPT-4 (\\~100K tokens), GPT-4o (\\~200K tokens), and Gemma (\\~256K tokens). Likewise, the int64 data type spans approximately $9.22 \\times 10^{18}$ distinct values, rendering explicit tokenization computationally infeasible. To address this challenge, we propose representing tokens through both a \*\*type\*\* and a \*\*value\*\*, enabling a structured embedding approach that significantly reduces model complexity. Under this strategy, each data type is assigned a compact, low-dimensional representation systematically projected into a higher-dimensional embedding space through a learned linear transformation—an operation we term an \*up-projection\*. Taking RGB values as a concrete example, each RGB token can be efficiently modeled as a purely imaginary quaternion, mapping the hexadecimal range `[00, FF]` to the floating-point range `[-1.0, 1.0]` using BF16 precision. Consequently, each RGB token is succinctly represented by just four BF16 values. To achieve a high-dimensional token embedding (e.g., 512 dimensions), this representation requires only a learned real-valued matrix of shape $512 \\times 4$. Remarkably, this reduces the embedding parameters required for all possible RGB tokens—from explicitly storing embeddings for more than 16 million distinct values—to merely 2,048 parameters. Further efficiency can be attained by employing quaternion-valued weight matrices, effectively quartering parameter counts. While traditionally projecting four real dimensions would require 16 real weights, quaternion arithmetic allows the same projection using just four quaternion weights. Thus, the original real-valued matrix of size $512 \\times 4 = 2048$ parameters becomes just 128 quaternion weights (512 real-valued parameters), substantially expanding representational efficiency without increasing complexity. Additionally, retrieving the original RGB values from the high-dimensional embeddings is computationally straightforward due to the constant-time complexity $O(1)$ of quaternion inversion. By applying the inverse quaternion transformations, the original RGB values are recovered precisely from the token embeddings, providing an efficient and exact decoding mechanism suitable for real-world applications. Quaternion algebra is abundantly covered in existing literature, and the cited references provide thorough treatment; thus, we shall not beleaguer you here with their basic properties. Quaternions came from Hamilton after his really good work had been done, and though beautifully ingenious, have been an unmixed evil to those who have touched them in any way. - Lord Kelvin

---

Thanks for the detailed background. I’ll analyze when it’s best to use tensor core-accelerated matrix operations (like with `cublasGemmEx`) versus custom fused arithmetic CUDA kernels for different Hamilton product workloads—e.g., single quaternion vs. batch, batch vs. single, and batch vs. batch (pairwise).

I’ll assess the impact of data precision (FP16/BF16), tensor core usage across architectures (e.g., Ampere and newer), and expected performance tradeoffs based on memory access patterns and compute-to-memory ratio.

I’ll get back to you shortly with clear benchmarks, implementation strategies, and guidelines for when to favor one approach over the other.

Optimizing Hamilton Product Kernels on NVIDIA GPUs

## Efficient Hamilton Product on NVIDIA GPUs: Tensor Cores vs. Fused Arithmetic

## Tensor Cores and Quaternion Multiplication on GPUs

Modern NVIDIA GPUs include **Tensor Cores** optimized for matrix math. Using them for quaternion (Hamilton) products can massively boost throughput *if* the problem is structured as matrix operations in low precision (FP16/BF16). For example, on an A100 (Ampere), FP16/BF16 matrix multiplies can hit ~312 TFLOPs, versus ~19.5 TFLOPs for standard FP32 ALU operations. However, Tensor Cores operate on fixed-size matrix *fragments* (e.g. 16×16, 8×32 tiles); there’s no direct 4×4 support exposed in CUDA. This means a single quaternion product (which is 4×4×4 matrix multiply if formulated naively) cannot directly use a Tensor Core without tiling or padding. In short, harnessing Tensor Cores for quaternions requires batching or reshaping many operations into larger matrix multiplies.

On the other hand, a **fused arithmetic** kernel can compute each quaternion multiplication with ~16 FMAs entirely in registers. Quaternion math isn’t *truly* “free” just because it’s fewer ops than a 4×4 matrix multiply – it involves several dot-products, cross terms, and sign flips. As one engineer notes, when GPUs have dedicated hardware for 4×4 matrix FMA, you **“can’t necessarily rely on the intuition about number of operations”**; the quaternion formula’s negations and dot products *aren’t* free, and a 4×4 matrix multiply could be as cheap or cheaper on GPU. In other words, hardware may execute a small matrix multiply very efficiently, potentially outpacing an unoptimized sequence of scalar ops. Deciding which approach wins depends on the *scenario and scale* of your quaternion multiplications, as detailed below.

## One-to-Many Multiplication (Single Quaternion \* Many)

If you need to multiply one quaternion by a large number of others (or vice-versa), you can treat the single quaternion as a 4×4 matrix acting on a vector of quaternion components. For instance, left-multiplying many quaternions by a fixed quaternion *Q* can be done as a matrix multiply `M(Q) * V`, where `M(Q)` is the 4×4 matrix representation of quaternion *Q*. This allows leveraging library GEMM functions (cuBLAS or CUTLASS) to multiply a 4×4 by a 4×N matrix of stacked quaternions. With FP16/BF16 inputs, this will engage Tensor Cores on Volta/Ampere GPUs and execute extremely fast on hardware. It’s essentially computing many Hamilton products in parallel via one big matrix op.

**When is this beneficial?** If *N* (the number of quaternions in the batch) is sufficiently large to utilize the GPU fully, the matrix/Tensor Core approach shines. A single 4×4×(4×N) multiply can run as one high-throughput kernel, amortizing launch overhead and saturating the GPU’s matrix units. In practice, when N is on the order of thousands (or more), using a matrix kernel is often faster than launching thousands of tiny quaternion kernels. The dedicated matrix hardware can execute multiple 4×4 multiplies per clock across warps, so doing it as a GEMM gains efficiency. **However, for smaller batches**, the benefits fade – the overhead of setting up a GEMM and padding the data may outweigh the speedup. Earlier NVIDIA libraries even required dimensions be multiples of 8 to use Tensor Cores (because 8×FP16 = 16 bytes). Newer versions relax this, but **performance is still best when sizes align to 8** elements. In the quaternion case (4 elements), you’d likely pad to 8 or aggregate operations to meet this alignment for optimal speed.

Also note that constructing the 4×4 matrix for *Q* is trivial (just a few sign flips of *Q* ’s components), but if *Q* is changing per multiplication, you’d be generating many 4×4 matrices which is extra work/memory. In the one-to-many scenario, *Q* is constant, so this overhead is negligible (one 4×4 matrix reused for all multiplies). The many quaternions can reside contiguously in memory and be loaded efficiently. The Tensor Core matrix approach will be compute-bound in this scenario – ideal for GPU, since the single *Q* reuse means higher arithmetic intensity. By contrast, a naive kernel where each thread multiplies *Q* with one target quaternion might become memory-bound for large N (each thread only does 16 FLOPs but reads/writes 8–12 floats). If N is huge, both approaches will eventually bottleneck on memory throughput, but the matrix method has the advantage of fewer kernel launches and potentially better caching of the constant *Q*.

**When to prefer fused arithmetic:** If the batch size is small or moderate (say only dozens or a few hundred quaternions), it’s usually faster and simpler to use a custom CUDA kernel that broadcasts the one quaternion and multiplies each target in parallel. Each thread can load *Q* (e.g. from **constant** or shared memory) and its own quaternion, then produce the product in registers with minimal latency. For such sizes, the Tensor Core isn’t fully utilized (a 4×N multiply with N small wastes most of the 16×16 tile capacity), and the launch overhead of cuBLAS or WMMA may dominate. In fact, using an explicit matrix for a tiny workload can backfire due to extra memory ops: one experiment found that representing rotations as 4×4 matrices in a shader made it **80% slower** than using quaternions directly. The quaternion approach saved sending an extra float4 per instance (lower memory footprint), which improved cache usage and outweighed the increased ALU work. This underscores that for smaller cases, avoiding unnecessary data expansion is key. As a rule of thumb, if your one-to-many multiply involves only a few hundred outputs or less, stick with a fused arithmetic kernel. If it involves thousands to millions of outputs, lean toward a matrix/Tensor Core solution.

*Summary (One-to-Many):*

- **Use matrix/Tensor Core approach** if a single quaternion is applied to a large batch (thousands+) of others, especially on Ampere+ GPUs with FP16/BF16. This maximizes throughput by using specialized hardware. Ensure to pad/align the 4-element dimension to 8 for best performance (e.g. treat as 8 with zeros, or batch multiple multiplies together).
- **Use a fused multiply kernel** if the batch is small or moderate. A custom CUDA kernel can keep the quaternion in registers and avoid overhead. Memory access is minimized (just 4 inputs + 4 outputs per thread), and you’ll avoid under-utilizing the Tensor Cores for tiny problems.

*(The same logic applies in reverse for many quaternions times one constant quaternion – e.g. multiplying a batch on the right by a fixed quaternion. You can form a 4×4 matrix for the right-multiply and proceed similarly.)*

## Many-to-Many (Pairwise Batched Multiplications)

This scenario is a **batch of independent quaternion products**, i.e. you have two arrays *A\[i\]* and *B\[i\]* of quaternions and want to compute *C\[i\] = A\[i\] ⊗ B\[i\]* for i=1…N. Here, each multiplication is separate, and no single quaternion is reused across all operations. It’s harder to directly leverage a single large GEMM, because each pair produces its own result. There are two main approaches:

- **Fused parallel kernel:** Launch a CUDA kernel with N threads (or warps), where each thread reads one pair *(A\[i\], B\[i\])*, computes the Hamilton product in registers, and writes *C\[i\]*. This is straightforward and typically very efficient for large N. All multiplications happen in parallel, and each thread’s workload is small but independent. The computation per thread (16 FLOPs) is low, so the kernel will be somewhat memory-bandwidth-bound (each product reads 8 floats and writes 4). But with thousands of threads, modern GPUs can hide latency and approach memory throughput limits. In practice, this approach scales well: if N is large (e.g. 10

---

It looks like you got cut off?

---

yep—sorry about that. here’s the rest, plus a tight set of rules-of-thumb and code patterns you can use right away.

---

## When to use tensor-core GEMMs vs. a fused arithmetic kernel for Hamilton products

## Quick intuition (numbers you can reason with)

A Hamilton product per pair is ~28 FLOPs (≈16 mul + ≈12 add) on 8–16 bytes per quaternion depending on precision:

- **FP16/BF16 storage**:  
	read A (8 B) + B (8 B) → write C (8 B) ≈ **24 B** per product → **~1.2 FLOPs/byte** → **memory-bound**.
- **If you can reuse** one quaternion across many products, arithmetic intensity jumps because the reused operand can stay in registers/L1/shared, and you only stream the other array → **compute becomes competitive** and Tensor Cores can help a lot.

This leads to the decision tree below.

---

## Decision guide

### 1) One-to-Many (fixed Q times many quaternions) — also “Many-to-One”

- **Batch N ≤ ~1–2k**: use a **fused kernel** (broadcast Q in registers/const memory).  
	You avoid GEMM setup/tiling overhead and under-utilized tensor tiles.
- **Batch N ≥ ~4k–8k** (Ampere/Hopper, FP16/BF16 with FP32 accumulate): use a **matrix path on Tensor Cores**.  
	Form the left-(or right-) multiplication matrix **M(Q) ∈ ℝ⁴×⁴**, pad to 8 rows if needed, and compute `Y = M(Q) · X` where `X ∈ ℝ⁴×N` holds quaternions column-wise. With (cuBLASLt/CUTLASS) you’ll saturate TCs and amortize launch overhead.
- **Extremely large N (≥ ~64k)**: definitely **Tensor Cores**. You’re firmly compute-efficient and you also reduce kernel launches.

> Heuristic thresholds depend on GPU, but on A100/H100 the crossover for 4×N tends to be in the low-thousands when you enable TC & striding correctly. If your kernel is in a larger pipeline (no extra mem round-trip), the fused path’s crossover shifts upward (i.e., fused wins for longer).

### 2) Pairwise Many-to-Many (elementwise A\[i\] ⊗ B\[i\] for large N)

- **Default**: use a **fused arithmetic kernel**. There’s no operand reuse, so GEMM doesn’t lift arithmetic intensity. Each pair is 24 B I/O (FP16/BF16) for ~28 FLOPs → **memory-bound**; TCs won’t help much.
- **Exception**: If you can **block** the problem to reuse one side (e.g., compute all A\[i\] ⊗ Q for many i, then change Q), switch to the **One-to-Many** strategy per block and leverage TCs for those sub-batches.
- **Batched GEMM?** 4×4 strided batched GEMM rarely beats a tuned fused kernel unless your batch is **huge** (≥ ~100k) and you can **group** calls (cuBLASLt grouped GEMM) and/or **pack** multiple independent 4×4s into larger tiles. Otherwise launch overhead + poor tile occupancy dominate.

### 3) Latency vs. Throughput

- **Latency/interactive** (small batches, single shots): **fused kernel**.
- **Throughput/offline** (big batches, training, preprocessing): **Tensor Core matrix** where you can find reuse or accumulate enough work to fill tiles.

---

## Implementation patterns

### A) Fused Hamilton product kernel (FP16/BF16 with FP32 accumulate)

- **Layout**: store quaternions as `float4` (or `__half2` pairs) in **SoA** (w\[\], x\[\], y\[\], z\[\]) or tightly packed AoS aligned to 16 bytes.
- **Broadcast** Q via `__constant__` or load once per block into shared and then registers.
- **Do math in FP32**, cast back to FP16/BF16 on store (Ampere/Hopper do cheap converts).
- **Vectorized loads/stores** (`float4`, `uint2` for halves) for coalescing.

Pseudo-sketch (abbrev):

```cpp
cppstruct hquat { __half w, x, y, z; };

__constant__ float Q_const[4]; // w,x,y,z in fp32

__global__ void hamilton_fused(const hquat* __restrict__ A,
                               const hquat* __restrict__ B, // optional (pairwise)
                               hquat* __restrict__ C, int N, bool pairwise) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float aw, ax, ay, az;
  if (pairwise) {
    // load A[i], B[i]
  } else {
    // load V[i], and use Q_const as the other operand
  }

  // Hamilton product in fp32
  // (w1,x1,y1,z1) ⊗ (w2,x2,y2,z2)
  float w = aw*bw - ax*bx - ay*by - az*bz;
  float x = aw*bx + ax*bw + ay*bz - az*by;
  float y = aw*by - ax*bz + ay*bw + az*bx;
  float z = aw*bz + ax*by - ay*bx + az*bw;

  // cast to half/bf16 and store
}
```

**Tuning tips**

- Use `-use_fast_math`, unroll the 4-component ops.
- Choose `blockDim.x` 256–512; launch enough blocks to oversubscribe SMs.
- For pairwise: prefetch with `__ldg` or rely on L1; keep occupancy high—this is bandwidth-bound.

### B) One-to-Many via Tensor Cores (cuBLASLt / CUTLASS)

1. Build the 4×4 real matrix **M(Q)** for left-multiply (or R(Q) for right-multiply).  
	For `Q=(w,x,y,z)`, a common left-multiply matrix is:
2. Pack inputs as `X ∈ ℝ^{4×N}` (column-major preferred for cuBLAS) with 4-row leading dimension.
3. **Pad rows to 8** (or pack two quaternions per column → 8 rows) to better hit Tensor Core tile multiples.
4. Call **`cublasLtMatmul`** (or CUTLASS GEMM) with:
	- **A**: `M(Q)` (4/8 × 4), **B**: `X` (4 × N), **C**: `Y` (4 × N).
	- **ComputeType**: FP32 accumulate, **ScaleType** FP32, **A/B data types** FP16 or BF16.
	- Set **epilogue** if you want post-ops (bias, activation) fused.
5. If you have many independent Q’s each applied to large batches, use **grouped matmul** in cuBLASLt to amortize launches and keep TCs busy.

**CUTLASS** has examples of small-K GEMMs and batched/grouped GEMM; choose shapes that map to `m8n8k16` / `m16n8k16` tiles on Ampere/Hopper and verify TC usage with Nsight Compute (check `sm__sass_thread_inst_executed_pipe_tensor.sum`).

### C) Pairwise with “packed tiles” (advanced)

If you *must* try Tensor Cores for pairwise: pack multiple independent 4×4 products into a **block-diagonal** super-matrix so a single GEMM does many small multiplies at once. This is doable with custom kernels/CUTLASS iterators, but it **adds packing/unpacking cost**. It only pays off at very large N and when you can keep the data resident on chip across multiple steps. Otherwise, the simple fused kernel still wins.

---

## Data types

- **Prefer BF16 over FP16** if your value range needs more exponent range; both hit TCs on Ampere+ with FP32 accumulate.
- If you need more accuracy in long chains of multiplies, keep **accumulate in FP32** and **store in FP16/BF16**.
- On Hopper, **FP8** exists, but quaternion stability usually suffers—only consider with loss-aware training and scalers.

---

## Memory layout & bandwidth

- **AoS (float4/half4)** is easy and coalesces well if arrays are contiguous.  
	**SoA** (four separate arrays) can help specific vectorization patterns, but most kernels run great with AoS aligned to 16B (use `reinterpret_cast<const float4*>`).
- For FP16/BF16, use `__half2` and do pairwise ops when possible; you can map quaternion math to pairs to reduce instruction count (just mind the swizzles).
- Pin down **coalesced** access: thread `i` reads `A[i]` / `B[i]`, writes `C[i]`. Avoid strided or gather patterns unless you stage in shared memory.

---

## Concrete rules of thumb (Ampere/Hopper, FP16/BF16)

- **One-to-Many / Many-to-One**
	- N < ~2k → **Fused kernel** (broadcast Q); lowest latency.
	- N ≥ ~4k–8k → **Tensor Core GEMM** (pad rows to 8; FP32 accumulate).
	- N ≥ ~64k → **Tensor Cores**; consider cuBLASLt grouped GEMM if multiple Q blocks.
- **Pairwise Many-to-Many**
	- Use **fused kernel** almost always.
	- Consider TC packing only if N is huge, you can keep data resident, and you’re already using CUTLASS with custom iterators.

---

## Sanity checks with profilers

- **Nsight Compute**: confirm TC utilization (`pipe_tensor`) for GEMM path; for fused path check memory throughput vs. device peak to see if you’re bandwidth-limited.
- **Roofline**: pairwise fused will sit on the bandwidth roof; one-to-many GEMM should move up toward the compute roof as reuse increases.

---

## Skeletons you can drop in

### cuBLASLt (one-to-many), BF16/FP16 → FP32 accumulate

- Create `cublasLtMatmulDesc_t` with `CUBLAS_COMPUTE_32F`.
- Set `A` =M(Q) (4×4, row-major), `B` =X (4×N), `C` =Y (4×N).  
	Use `cublasLtMatrixLayout` with strides so you can reuse `A` without copies.
- Enable **heuristic preference** for TC (set math mode “allow TF32 on FP32” if you ever switch to FP32).

### Fused kernel (pairwise)

- Launch `<<<(N+255)/256, 256>>>`.
- Load two `float4` (or `uint2` for halves), do fp32 math, store one `float4`.
- If chaining ops, **fuse** as much as possible (normalization, scale, bias) to avoid extra DRAM passes.

---

## TL;DR

- **If you can reuse one quaternion across many products, batch it as a GEMM and hit Tensor Cores.** Pad to 8 rows, FP32 accumulate, and you’ll win for N in the low-thousands and up.
- **If every product is pairwise and independent, a fused arithmetic kernel is king.** You’re bandwidth-bound; TC won’t help much unless you do nontrivial packing.
- Optimize memory layout and vectorized I/O first; then choose TC vs. fused based on batch size and reuse.

If you want, I can sketch a minimal CUTLASS config for the 4×N case and a tuned fused kernel (FP16/BF16 I/O, FP32 math) that you can benchmark on your GPU.

---

so when you say 4k-8k for using tensorcores would that be 4-8 k per token? since each token would have a different initial q even if the up projection matrix was different?

---

ah, good catch — let me pin this down in the context of your **type/value embedding idea** (where each token gets its own quaternion $q$ , then is projected up with a learned weight matrix).

---

### what “4k–8k” really meant in my note

that threshold was about the **effective column dimension $N$** in the GEMM

$$
Y = M(q)\,X,
$$

where $M(q)$ is the 4×4 matrix form of a fixed quaternion and $X$ is a 4×N stack of input quaternions.

- if you only have **one $q$** reused across thousands of other quaternions, then $N$ is the number of “other quaternions” you’re multiplying it against. once $N \gtrsim 4{,}000$ , the GEMM path usually wins over a custom kernel because tensor cores are fully fed.
- if you instead have **many different $q$** (one per token), you don’t have reuse. then you’re effectively in the **pairwise case** — each $M(q_i)$ is different and only applied once. in that situation, you cannot form one big GEMM; you’d need a *batched GEMM* with thousands of tiny 4×4 multiplies. that almost never outperforms a fused kernel unless you aggressively pack them into block-diagonal tiles (a lot of overhead). so here the “4k–8k” cutoff does **not** apply; you stay in the fused kernel regime.

---

### mapping back to your embedding pipeline

for your up-projection idea:

- **the learned 512×4 weight matrix** (real or quaternion-valued) is shared across tokens. that’s a **perfect GEMM use-case** — you’re multiplying a 512×4 matrix by a 4×B batch of token quaternions. $B$ is the batch size (tokens in flight). here you absolutely want tensor cores once $B$ is even a few dozen, because the op is large and dense.
- **per-token quaternion pre-processing** (e.g. mapping RGB value to an imaginary quaternion) is just a handful of scalar ops; fused kernel is fine.
- **Hamilton products between token-dependent quaternions** (if you ever do them pairwise, one-shot per token) → treat as the pairwise case. that stays fused kernel, since each $q$ is unique and not reused.

---

### distilled rule for you

- **Shared linear maps** (like your up-projection weights): always **GEMM on tensor cores**.
- **One quaternion reused across thousands of tokens**: **GEMM on tensor cores** once batch ≥ a few thousand.
- **Each token has its own unique quaternion** (no reuse): treat as **pairwise** → **fused arithmetic kernel**; tensor cores won’t help.

---

do you want me to sketch how you could pack many small 4×4 quaternion multiplications into a larger GEMM tile (block-diagonal trick) so you could still get some TC throughput, or are you leaning toward just doing the fused kernel for those token-unique cases?

---

but it's not a simple matrix? it's a set of concatenated quaternions that need to have the hamilton product done to them?

---

ah right — important distinction. 👍  
you’re not just “multiplying a quaternion by a dense vector” (which is a straight 4×4 matrix-vector multiply). you’ve got **a bunch of quaternions laid out back-to-back**, and you need to apply the Hamilton product formula to each one. that’s subtly different from a plain GEMM.

---

### two possible situations

#### 1\. same left-quaternion qq for the whole batch

- math: $c_i = q \otimes b_i$ , with $b_i$ varying.
- here you can view it as a **linear map** on $\mathbb{R}^4$ : the Hamilton product with a fixed $q$ *is* equivalent to multiplying by a structured 4×4 real matrix $M(q)$ .
- if you stack your $b_i$ s as a 4×N matrix, then yes:
	$$
	C = M(q) \cdot B
	$$
	is a legitimate GEMM. that’s where tensor cores shine (once N is large enough).
- the “concatenated quaternions” are just columns in that 4×N matrix.

#### 2\. each pair has its own left-quaternion qiq\_i

- math: $c_i = q_i \otimes b_i$ .
- now you don’t have one $M(q)$ , you have N of them. you cannot concatenate them into a single dense GEMM without making a big **block-diagonal** matrix:
	$$
	\text{diag}(M(q_1), M(q_2), …, M(q_N)) \cdot 
	\begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_N \end{bmatrix}.
	$$
	That’s very sparse and wastes almost all multiply slots unless you custom-pack it. cuBLAS won’t help; tensor cores won’t be fully utilized.
- in practice, this falls back to the **fused arithmetic kernel**: one thread per pair does the Hamilton product directly in registers.

---

### so: when it’s “concatenated quaternions”

- if **all share the same $q$** → treat them as a 4×N matrix and feed tensor cores.
- if **each has a different $q_i$** → no reuse → just do fused arithmetic kernel.

---

### why the cutoff matters

the “4k–8k” batch size threshold I gave only applies to case (1) (single $q$ reused). for case (2) (each token has its own $q$ ), there’s no batch size big enough to make tensor cores efficient without exotic packing. you’ll almost always be memory-bound, so a hand-tuned fused kernel is the right tool.

---

would it help if I show you explicitly how to lay out your “concatenated quaternion array” so it maps nicely to the 4×N matrix view in case (1)? that way you could drop it straight into cuBLAS/cuBLASLt and check tensor core utilization in Nsight.

---

what do they do here for left and right multiplication when they use matrices? Why do they say they use matrices? """ This file is part of the pyquaternion python module Author: Kieran Wynn Website: https://github.com/KieranWynn/pyquaternion Documentation: http://kieranwynn.github.io/pyquaternion/ Version: 1.0.0 License: The MIT License (MIT) Copyright (c) 2015 Kieran Wynn Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. quaternion.py - This file defines the core Quaternion class """ from \_\_future\_\_ import absolute\_import, division, print\_function # Add compatibility for Python 2.7+ from math import sqrt, pi, sin, cos, asin, acos, atan2, exp, log from copy import deepcopy import numpy as np # Numpy is required for many vector operations class Quaternion: """Class to represent a 4-dimensional complex number or quaternion. Quaternion objects can be used generically as 4D numbers, or as unit quaternions to represent rotations in 3D space. Attributes: q: Quaternion 4-vector represented as a Numpy array """ def \_\_init\_\_(self, \*args, \*\*kwargs): """Initialise a new Quaternion object. See Object Initialisation docs for complete behaviour: https://kieranwynn.github.io/pyquaternion/#object-initialisation """ s = len(args) if s == 0: # No positional arguments supplied if kwargs: # Keyword arguments provided if ("scalar" in kwargs) or ("vector" in kwargs): scalar = kwargs.get("scalar", 0.0) if scalar is None: scalar = 0.0 else: scalar = float(scalar) vector = kwargs.get("vector", \[\]) vector = self.\_validate\_number\_sequence(vector, 3) self.q = np.hstack((scalar, vector)) elif ("real" in kwargs) or ("imaginary" in kwargs): real = kwargs.get("real", 0.0) if real is None: real = 0.0 else: real = float(real) imaginary = kwargs.get("imaginary", \[\]) imaginary = self.\_validate\_number\_sequence(imaginary, 3) self.q = np.hstack((real, imaginary)) elif ("axis" in kwargs) or ("radians" in kwargs) or ("degrees" in kwargs) or ("angle" in kwargs): try: axis = self.\_validate\_number\_sequence(kwargs\["axis"\], 3) except KeyError: raise ValueError( "A valid rotation 'axis' parameter must be provided to describe a meaningful rotation." ) angle = kwargs.get('radians') or self.to\_radians(kwargs.get('degrees')) or kwargs.get('angle') or 0.0 self.q = Quaternion.\_from\_axis\_angle(axis, angle).q elif "array" in kwargs: self.q = self.\_validate\_number\_sequence(kwargs\["array"\], 4) elif "matrix" in kwargs: optional\_args = {key: kwargs\[key\] for key in kwargs if key in \['rtol', 'atol'\]} self.q = Quaternion.\_from\_matrix(kwargs\["matrix"\], \*\*optional\_args).q else: keys = sorted(kwargs.keys()) elements = \[kwargs\[kw\] for kw in keys\] if len(elements) == 1: r = float(elements\[0\]) self.q = np.array(\[r, 0.0, 0.0, 0.0\]) else: self.q = self.\_validate\_number\_sequence(elements, 4) else: # Default initialisation self.q = np.array(\[1.0, 0.0, 0.0, 0.0\]) elif s == 1: # Single positional argument supplied if isinstance(args\[0\], Quaternion): self.q = args\[0\].q return if args\[0\] is None: raise TypeError("Object cannot be initialised from {}".format(type(args\[0\]))) try: r = float(args\[0\]) self.q = np.array(\[r, 0.0, 0.0, 0.0\]) return except TypeError: pass # If the single argument is not scalar, it should be a sequence self.q = self.\_validate\_number\_sequence(args\[0\], 4) return else: # More than one positional argument supplied self.q = self.\_validate\_number\_sequence(args, 4) def \_\_hash\_\_(self): return hash(tuple(self.q)) def \_validate\_number\_sequence(self, seq, n): """Validate a sequence to be of a certain length and ensure it's a numpy array of floats. Raises: ValueError: Invalid length or non-numeric value """ if seq is None: return np.zeros(n) if len(seq) == n: try: l = \[float(e) for e in seq\] except ValueError: raise ValueError("One or more elements in sequence <{!r}> cannot be interpreted as a real number".format(seq)) else: return np.asarray(l) elif len(seq) == 0: return np.zeros(n) else: raise ValueError("Unexpected number of elements in sequence. Got: {}, Expected: {}.".format(len(seq), n)) # Initialise from matrix @classmethod def \_from\_matrix(cls, matrix, rtol=1e-05, atol=1e-08): """Initialise from matrix representation Create a Quaternion by specifying the 3x3 rotation or 4x4 transformation matrix (as a numpy array) from which the quaternion's rotation should be created. """ try: shape = matrix.shape except AttributeError: raise TypeError("Invalid matrix type: Input must be a 3x3 or 4x4 numpy array or matrix") if shape == (3, 3): R = matrix elif shape == (4, 4): R = matrix\[:-1\]\[:,:-1\] # Upper left 3x3 sub-matrix else: raise ValueError("Invalid matrix shape: Input must be a 3x3 or 4x4 numpy array or matrix") # Check matrix properties if not np.allclose(np.dot(R, R.conj().transpose()), np.eye(3), rtol=rtol, atol=atol): raise ValueError("Matrix must be orthogonal, i.e. its transpose should be its inverse") if not np.isclose(np.linalg.det(R), 1.0, rtol=rtol, atol=atol): raise ValueError("Matrix must be special orthogonal i.e. its determinant must be +1.0") def decomposition\_method(matrix): """ Method supposedly able to deal with non-orthogonal matrices - NON-FUNCTIONAL! Based on this method: http://arc.aiaa.org/doi/abs/10.2514/2.4654 """ x, y, z = 0, 1, 2 # indices K = np.array(\[ \[R\[x, x\]-R\[y, y\]-R\[z, z\], R\[y, x\]+R\[x, y\], R\[z, x\]+R\[x, z\], R\[y, z\]-R\[z, y\]\], \[R\[y, x\]+R\[x, y\], R\[y, y\]-R\[x, x\]-R\[z, z\], R\[z, y\]+R\[y, z\], R\[z, x\]-R\[x, z\]\], \[R\[z, x\]+R\[x, z\], R\[z, y\]+R\[y, z\], R\[z, z\]-R\[x, x\]-R\[y, y\], R\[x, y\]-R\[y, x\]\], \[R\[y, z\]-R\[z, y\], R\[z, x\]-R\[x, z\], R\[x, y\]-R\[y, x\], R\[x, x\]+R\[y, y\]+R\[z, z\]\] \]) K = K / 3.0 e\_vals, e\_vecs = np.linalg.eig(K) print('Eigenvalues:', e\_vals) print('Eigenvectors:', e\_vecs) max\_index = np.argmax(e\_vals) principal\_component = e\_vecs\[max\_index\] return principal\_component def trace\_method(matrix): """ This code uses a modification of the algorithm described in: https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf which is itself based on the method described here: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/ Altered to work with the column vector convention instead of row vectors """ m = matrix.conj().transpose() # This method assumes row-vector and postmultiplication of that vector if m\[2, 2\] < 0: if m\[0, 0\] > m\[1, 1\]: t = 1 + m\[0, 0\] - m\[1, 1\] - m\[2, 2\] q = \[m\[1, 2\]-m\[2, 1\], t, m\[0, 1\]+m\[1, 0\], m\[2, 0\]+m\[0, 2\]\] else: t = 1 - m\[0, 0\] + m\[1, 1\] - m\[2, 2\] q = \[m\[2, 0\]-m\[0, 2\], m\[0, 1\]+m\[1, 0\], t, m\[1, 2\]+m\[2, 1\]\] else: if m\[0, 0\] < -m\[1, 1\]: t = 1 - m\[0, 0\] - m\[1, 1\] + m\[2, 2\] q = \[m\[0, 1\]-m\[1, 0\], m\[2, 0\]+m\[0, 2\], m\[1, 2\]+m\[2, 1\], t\] else: t = 1 + m\[0, 0\] + m\[1, 1\] + m\[2, 2\] q = \[t, m\[1, 2\]-m\[2, 1\], m\[2, 0\]-m\[0, 2\], m\[0, 1\]-m\[1, 0\]\] q = np.array(q).astype('float64') q \*= 0.5 / sqrt(t) return q return cls(array=trace\_method(R)) # Initialise from axis-angle @classmethod def \_from\_axis\_angle(cls, axis, angle): """Initialise from axis and angle representation Create a Quaternion by specifying the 3-vector rotation axis and rotation angle (in radians) from which the quaternion's rotation should be created. Params: axis: a valid numpy 3-vector angle: a real valued angle in radians """ mag\_sq = np.dot(axis, axis) if mag\_sq == 0.0: raise ZeroDivisionError("Provided rotation axis has no length") # Ensure axis is in unit vector form if (abs(1.0 - mag\_sq) > 1e-12): axis = axis / sqrt(mag\_sq) theta = angle / 2.0 r = cos(theta) i = axis \* sin(theta) return cls(r, i\[0\], i\[1\], i\[2\]) @classmethod def random(cls): """Generate a random unit quaternion. Uniformly distributed across the rotation space As per: http://planning.cs.uiuc.edu/node198.html """ r1, r2, r3 = np.random.random(3) q1 = sqrt(1.0 - r1) \* (sin(2 \* pi \* r2)) q2 = sqrt(1.0 - r1) \* (cos(2 \* pi \* r2)) q3 = sqrt(r1) \* (sin(2 \* pi \* r3)) q4 = sqrt(r1) \* (cos(2 \* pi \* r3)) return cls(q1, q2, q3, q4) # Representation def \_\_str\_\_(self): """An informal, nicely printable string representation of the Quaternion object. """ return "{:.3f} {:+.3f}i {:+.3f}j {:+.3f}k".format(self.q\[0\], self.q\[1\], self.q\[2\], self.q\[3\]) def \_\_repr\_\_(self): """The 'official' string representation of the Quaternion object. This is a string representation of a valid Python expression that could be used to recreate an object with the same value (given an appropriate environment) """ return "Quaternion({!r}, {!r}, {!r}, {!r})".format(self.q\[0\], self.q\[1\], self.q\[2\], self.q\[3\]) def \_\_format\_\_(self, formatstr): """Inserts a customisable, nicely printable string representation of the Quaternion object The syntax for `format_spec` mirrors that of the built in format specifiers for floating point types. Check out the official Python \[format specification mini-language\](https://docs.python.org/3.4/library/string.html#formatspec) for details. """ if formatstr.strip() == '': # Defualt behaviour mirrors self.\_\_str\_\_() formatstr = '+.3f' string = \\ "{:" + formatstr +"} " + \\ "{:" + formatstr +"}i " + \\ "{:" + formatstr +"}j " + \\ "{:" + formatstr +"}k" return string.format(self.q\[0\], self.q\[1\], self.q\[2\], self.q\[3\]) # Type Conversion def \_\_int\_\_(self): """Implements type conversion to int. Truncates the Quaternion object by only considering the real component and rounding to the next integer value towards zero. Note: to round to the closest integer, use int(round(float(q))) """ return int(self.q\[0\]) def \_\_float\_\_(self): """Implements type conversion to float. Truncates the Quaternion object by only considering the real component. """ return float(self.q\[0\]) def \_\_complex\_\_(self): """Implements type conversion to complex. Truncates the Quaternion object by only considering the real component and the first imaginary component. This is equivalent to a projection from the 4-dimensional hypersphere to the 2-dimensional complex plane. """ return complex(self.q\[0\], self.q\[1\]) def \_\_bool\_\_(self): return not (self == Quaternion(0.0)) def \_\_nonzero\_\_(self): return not (self == Quaternion(0.0)) def \_\_invert\_\_(self): return (self == Quaternion(0.0)) # Comparison def \_\_eq\_\_(self, other): """Returns true if the following is true for each element: `absolute(a - b) <= (atol + rtol * absolute(b))` """ if isinstance(other, Quaternion): r\_tol = 1.0e-13 a\_tol = 1.0e-14 try: isEqual = np.allclose(self.q, other.q, rtol=r\_tol, atol=a\_tol) except AttributeError: raise AttributeError("Error in internal quaternion representation means it cannot be compared like a numpy array.") return isEqual return self.\_\_eq\_\_(self.\_\_class\_\_(other)) # Negation def \_\_neg\_\_(self): return self.\_\_class\_\_(array= -self.q) # Absolute value def \_\_abs\_\_(self): return self.norm # Addition def \_\_add\_\_(self, other): if isinstance(other, Quaternion): return self.\_\_class\_\_(array=self.q + other.q) return self + self.\_\_class\_\_(other) def \_\_iadd\_\_(self, other): return self + other def \_\_radd\_\_(self, other): return self + other # Subtraction def \_\_sub\_\_(self, other): return self + (-other) def \_\_isub\_\_(self, other): return self + (-other) def \_\_rsub\_\_(self, other): return -(self - other) # Multiplication def \_\_mul\_\_(self, other): if isinstance(other, Quaternion): return self.\_\_class\_\_(array=np.dot(self.\_q\_matrix(), other.q)) return self \* self.\_\_class\_\_(other) def \_\_imul\_\_(self, other): return self \* other def \_\_rmul\_\_(self, other): return self.\_\_class\_\_(other) \* self def \_\_matmul\_\_(self, other): if isinstance(other, Quaternion): return self.q.\_\_matmul\_\_(other.q) return self.\_\_matmul\_\_(self.\_\_class\_\_(other)) def \_\_imatmul\_\_(self, other): return self.\_\_matmul\_\_(other) def \_\_rmatmul\_\_(self, other): return self.\_\_class\_\_(other).\_\_matmul\_\_(self) # Division def \_\_div\_\_(self, other): if isinstance(other, Quaternion): if other == self.\_\_class\_\_(0.0): raise ZeroDivisionError("Quaternion divisor must be non-zero") return self \* other.inverse return self.\_\_div\_\_(self.\_\_class\_\_(other)) def \_\_idiv\_\_(self, other): return self.\_\_div\_\_(other) def \_\_rdiv\_\_(self, other): return self.\_\_class\_\_(other) \* self.inverse def \_\_truediv\_\_(self, other): return self.\_\_div\_\_(other) def \_\_itruediv\_\_(self, other): return self.\_\_idiv\_\_(other) def \_\_rtruediv\_\_(self, other): return self.\_\_rdiv\_\_(other) # Exponentiation def \_\_pow\_\_(self, exponent): # source: https://en.wikipedia.org/wiki/Quaternion#Exponential.2C\_logarithm.2C\_and\_power exponent = float(exponent) # Explicitly reject non-real exponents norm = self.norm if norm > 0.0: try: n, theta = self.polar\_decomposition except ZeroDivisionError: # quaternion is a real number (no vector or imaginary part) return Quaternion(scalar=self.scalar \*\* exponent) return (self.norm \*\* exponent) \* Quaternion(scalar=cos(exponent \* theta), vector=(n \* sin(exponent \* theta))) return Quaternion(self) def \_\_ipow\_\_(self, other): return self \*\* other def \_\_rpow\_\_(self, other): return other \*\* float(self) # Quaternion Features def \_vector\_conjugate(self): return np.hstack((self.q\[0\], -self.q\[1:4\])) def \_sum\_of\_squares(self): return np.dot(self.q, self.q) @property def conjugate(self): """Quaternion conjugate, encapsulated in a new instance. For a unit quaternion, this is the same as the inverse. Returns: A new Quaternion object clone with its vector part negated """ return self.\_\_class\_\_(scalar=self.scalar, vector=-self.vector) @property def inverse(self): """Inverse of the quaternion object, encapsulated in a new instance. For a unit quaternion, this is the inverse rotation, i.e. when combined with the original rotation, will result in the null rotation. Returns: A new Quaternion object representing the inverse of this object """ ss = self.\_sum\_of\_squares() if ss > 0: return self.\_\_class\_\_(array=(self.\_vector\_conjugate() / ss)) else: raise ZeroDivisionError("a zero quaternion (0 + 0i + 0j + 0k) cannot be inverted") @property def norm(self): """L2 norm of the quaternion 4-vector. This should be 1.0 for a unit quaternion (versor) Slow but accurate. If speed is a concern, consider using \_fast\_normalise() instead Returns: A scalar real number representing the square root of the sum of the squares of the elements of the quaternion. """ mag\_squared = self.\_sum\_of\_squares() return sqrt(mag\_squared) @property def magnitude(self): return self.norm def \_normalise(self): """Object is guaranteed to be a unit quaternion after calling this operation UNLESS the object is equivalent to Quaternion(0) """ if not self.is\_unit(): n = self.norm if n > 0: self.q = self.q / n def \_fast\_normalise(self): """Normalise the object to a unit quaternion using a fast approximation method if appropriate. Object is guaranteed to be a quaternion of approximately unit length after calling this operation UNLESS the object is equivalent to Quaternion(0) """ if not self.is\_unit(): mag\_squared = np.dot(self.q, self.q) if (mag\_squared == 0): return if (abs(1.0 - mag\_squared) < 2.107342e-08): mag = ((1.0 + mag\_squared) / 2.0) # More efficient. Pade approximation valid if error is small else: mag = sqrt(mag\_squared) # Error is too big, take the performance hit to calculate the square root properly self.q = self.q / mag @property def normalised(self): """Get a unit quaternion (versor) copy of this Quaternion object. A unit quaternion has a `norm` of 1.0 Returns: A new Quaternion object clone that is guaranteed to be a unit quaternion """ q = Quaternion(self) q.\_normalise() return q @property def polar\_unit\_vector(self): vector\_length = np.linalg.norm(self.vector) if vector\_length <= 0.0: raise ZeroDivisionError('Quaternion is pure real and does not have a unique unit vector') return self.vector / vector\_length @property def polar\_angle(self): return acos(self.scalar / self.norm) @property def polar\_decomposition(self): """ Returns the unit vector and angle of a non-scalar quaternion according to the following decomposition q = q.norm() \* (e \*\* (q.polar\_unit\_vector \* q.polar\_angle)) source: https://en.wikipedia.org/wiki/Polar\_decomposition#Quaternion\_polar\_decomposition """ return self.polar\_unit\_vector, self.polar\_angle @property def unit(self): return self.normalised def is\_unit(self, tolerance=1e-14): """Determine whether the quaternion is of unit length to within a specified tolerance value. Params: tolerance: \[optional\] maximum absolute value by which the norm can differ from 1.0 for the object to be considered a unit quaternion. Defaults to `1e-14`. Returns: `True` if the Quaternion object is of unit length to within the specified tolerance value. `False` otherwise. """ return abs(1.0 - self.\_sum\_of\_squares()) < tolerance # if \_sum\_of\_squares is 1, norm is 1. This saves a call to sqrt() def \_q\_matrix(self): """Matrix representation of quaternion for multiplication purposes. """ return np.array(\[ \[self.q\[0\], -self.q\[1\], -self.q\[2\], -self.q\[3\]\], \[self.q\[1\], self.q\[0\], -self.q\[3\], self.q\[2\]\], \[self.q\[2\], self.q\[3\], self.q\[0\], -self.q\[1\]\], \[self.q\[3\], -self.q\[2\], self.q\[1\], self.q\[0\]\]\]) def \_q\_bar\_matrix(self): """Matrix representation of quaternion for multiplication purposes. """ return np.array(\[ \[self.q\[0\], -self.q\[1\], -self.q\[2\], -self.q\[3\]\], \[self.q\[1\], self.q\[0\], self.q\[3\], -self.q\[2\]\], \[self.q\[2\], -self.q\[3\], self.q\[0\], self.q\[1\]\], \[self.q\[3\], self.q\[2\], -self.q\[1\], self.q\[0\]\]\]) def \_rotate\_quaternion(self, q): """Rotate a quaternion vector using the stored rotation. Params: q: The vector to be rotated, in quaternion form (0 + xi + yj + kz) Returns: A Quaternion object representing the rotated vector in quaternion from (0 + xi + yj + kz) """ self.\_normalise() return self \* q \* self.conjugate def rotate(self, vector): """Rotate a 3D vector by the rotation stored in the Quaternion object. Params: vector: A 3-vector specified as any ordered sequence of 3 real numbers corresponding to x, y, and z values. Some types that are recognised are: numpy arrays, lists and tuples. A 3-vector can also be represented by a Quaternion object who's scalar part is 0 and vector part is the required 3-vector. Thus it is possible to call `Quaternion.rotate(q)` with another quaternion object as an input. Returns: The rotated vector returned as the same type it was specified at input. Raises: TypeError: if any of the vector elements cannot be converted to a real number. ValueError: if `vector` cannot be interpreted as a 3-vector or a Quaternion object. """ if isinstance(vector, Quaternion): return self.\_rotate\_quaternion(vector) q = Quaternion(vector=vector) a = self.\_rotate\_quaternion(q).vector if isinstance(vector, list): l = \[x for x in a\] return l elif isinstance(vector, tuple): l = \[x for x in a\] return tuple(l) else: return a @classmethod def exp(cls, q): """Quaternion Exponential. Find the exponential of a quaternion amount. Params: q: the input quaternion/argument as a Quaternion object. Returns: A quaternion amount representing the exp(q). See \[Source\](https://math.stackexchange.com/questions/1030737/exponential-function-of-quaternion-derivation for more information and mathematical background). Note: The method can compute the exponential of any quaternion. """ tolerance = 1e-17 v\_norm = np.linalg.norm(q.vector) vec = q.vector if v\_norm > tolerance: vec = vec / v\_norm magnitude = exp(q.scalar) return Quaternion(scalar = magnitude \* cos(v\_norm), vector = magnitude \* sin(v\_norm) \* vec) @classmethod def log(cls, q): """Quaternion Logarithm. Find the logarithm of a quaternion amount. Params: q: the input quaternion/argument as a Quaternion object. Returns: A quaternion amount representing log(q):= (log(|q|), v/|v|acos(w/|q|)). Note: The method computes the logarithm of general quaternions. See \[Source\](https://math.stackexchange.com/questions/2552/the-logarithm-of-quaternion/2554#2554) for more details. """ v\_norm = np.linalg.norm(q.vector) q\_norm = q.norm tolerance = 1e-17 if q\_norm < tolerance: # 0 quaternion - undefined return Quaternion(scalar=-float('inf'), vector=float('nan')\*q.vector) if v\_norm < tolerance: # real quaternions - no imaginary part return Quaternion(scalar=log(q\_norm), vector=\[0, 0, 0\]) vec = q.vector / v\_norm return Quaternion(scalar=log(q\_norm), vector=acos(q.scalar/q\_norm)\*vec) @classmethod def exp\_map(cls, q, eta): """Quaternion exponential map. Find the exponential map on the Riemannian manifold described by the quaternion space. Params: q: the base point of the exponential map, i.e. a Quaternion object eta: the argument of the exponential map, a tangent vector, i.e. a Quaternion object Returns: A quaternion p such that p is the endpoint of the geodesic starting at q in the direction of eta, having the length equal to the magnitude of eta. Note: The exponential map plays an important role in integrating orientation variations (e.g. angular velocities). This is done by projecting quaternion tangent vectors onto the quaternion manifold. """ return q \* Quaternion.exp(eta) @classmethod def sym\_exp\_map(cls, q, eta): """Quaternion symmetrized exponential map. Find the symmetrized exponential map on the quaternion Riemannian manifold. Params: q: the base point as a Quaternion object eta: the tangent vector argument of the exponential map as a Quaternion object Returns: A quaternion p. Note: The symmetrized exponential formulation is akin to the exponential formulation for symmetric positive definite tensors \[Source\](http://www.academia.edu/7656761/On\_the\_Averaging\_of\_Symmetric\_Positive-Definite\_Tensors) """ sqrt\_q = q \*\* 0.5 return sqrt\_q \* Quaternion.exp(eta) \* sqrt\_q @classmethod def log\_map(cls, q, p): """Quaternion logarithm map. Find the logarithm map on the quaternion Riemannian manifold. Params: q: the base point at which the logarithm is computed, i.e. a Quaternion object p: the argument of the quaternion map, a Quaternion object Returns: A tangent vector having the length and direction given by the geodesic joining q and p. """ return Quaternion.log(q.inverse \* p) @classmethod def sym\_log\_map(cls, q, p): """Quaternion symmetrized logarithm map. Find the symmetrized logarithm map on the quaternion Riemannian manifold. Params: q: the base point at which the logarithm is computed, i.e. a Quaternion object p: the argument of the quaternion map, a Quaternion object Returns: A tangent vector corresponding to the symmetrized geodesic curve formulation. Note: Information on the symmetrized formulations given in \[Source\](https://www.researchgate.net/publication/267191489\_Riemannian\_L\_p\_Averaging\_on\_Lie\_Group\_of\_Nonzero\_Quaternions). """ inv\_sqrt\_q = (q \*\* (-0.5)) return Quaternion.log(inv\_sqrt\_q \* p \* inv\_sqrt\_q) @classmethod def absolute\_distance(cls, q0, q1): """Quaternion absolute distance. Find the distance between two quaternions accounting for the sign ambiguity. Params: q0: the first quaternion q1: the second quaternion Returns: A positive scalar corresponding to the chord of the shortest path/arc that connects q0 to q1. Note: This function does not measure the distance on the hypersphere, but it takes into account the fact that q and -q encode the same rotation. It is thus a good indicator for rotation similarities. """ q0\_minus\_q1 = q0 - q1 q0\_plus\_q1 = q0 + q1 d\_minus = q0\_minus\_q1.norm d\_plus = q0\_plus\_q1.norm if d\_minus < d\_plus: return d\_minus else: return d\_plus @classmethod def distance(cls, q0, q1): """Quaternion intrinsic distance. Find the intrinsic geodesic distance between q0 and q1. Params: q0: the first quaternion q1: the second quaternion Returns: A positive amount corresponding to the length of the geodesic arc connecting q0 to q1. Note: Although the q0^(-1)\*q1!= q1^(-1)\*q0, the length of the path joining them is given by the logarithm of those product quaternions, the norm of which is the same. """ q = Quaternion.log\_map(q0, q1) return q.norm @classmethod def sym\_distance(cls, q0, q1): """Quaternion symmetrized distance. Find the intrinsic symmetrized geodesic distance between q0 and q1. Params: q0: the first quaternion q1: the second quaternion Returns: A positive amount corresponding to the length of the symmetrized geodesic curve connecting q0 to q1. Note: This formulation is more numerically stable when performing iterative gradient descent on the Riemannian quaternion manifold. However, the distance between q and -q is equal to pi, rendering this formulation not useful for measuring rotation similarities when the samples are spread over a "solid" angle of more than pi/2 radians (the spread refers to quaternions as point samples on the unit hypersphere). """ q = Quaternion.sym\_log\_map(q0, q1) return q.norm @classmethod def slerp(cls, q0, q1, amount=0.5): """Spherical Linear Interpolation between quaternions. Implemented as described in https://en.wikipedia.org/wiki/Slerp Find a valid quaternion rotation at a specified distance along the minor arc of a great circle passing through any two existing quaternion endpoints lying on the unit radius hypersphere. This is a class method and is called as a method of the class itself rather than on a particular instance. Params: q0: first endpoint rotation as a Quaternion object q1: second endpoint rotation as a Quaternion object amount: interpolation parameter between 0 and 1. This describes the linear placement position of the result along the arc between endpoints; 0 being at `q0` and 1 being at `q1`. Defaults to the midpoint (0.5). Returns: A new Quaternion object representing the interpolated rotation. This is guaranteed to be a unit quaternion. Note: This feature only makes sense when interpolating between unit quaternions (those lying on the unit radius hypersphere). Calling this method will implicitly normalise the endpoints to unit quaternions if they are not already unit length. """ # Ensure quaternion inputs are unit quaternions and 0 <= amount <=1 q0.\_fast\_normalise() q1.\_fast\_normalise() amount = np.clip(amount, 0, 1) dot = np.dot(q0.q, q1.q) # If the dot product is negative, slerp won't take the shorter path. # Note that v1 and -v1 are equivalent when the negation is applied to all four components. # Fix by reversing one quaternion if dot < 0.0: q0.q = -q0.q dot = -dot # sin\_theta\_0 can not be zero if dot > 0.9995: qr = Quaternion(q0.q + amount \* (q1.q - q0.q)) qr.\_fast\_normalise() return qr theta\_0 = np.arccos(dot) # Since dot is in range \[0, 0.9995\], np.arccos() is safe sin\_theta\_0 = np.sin(theta\_0) theta = theta\_0 \* amount sin\_theta = np.sin(theta) s0 = np.cos(theta) - dot \* sin\_theta / sin\_theta\_0 s1 = sin\_theta / sin\_theta\_0 qr = Quaternion((s0 \* q0.q) + (s1 \* q1.q)) qr.\_fast\_normalise() return qr @classmethod def intermediates(cls, q0, q1, n, include\_endpoints=False): """Generator method to get an iterable sequence of `n` evenly spaced quaternion rotations between any two existing quaternion endpoints lying on the unit radius hypersphere. This is a convenience function that is based on `Quaternion.slerp()` as defined above. This is a class method and is called as a method of the class itself rather than on a particular instance. Params: q\_start: initial endpoint rotation as a Quaternion object q\_end: final endpoint rotation as a Quaternion object n: number of intermediate quaternion objects to include within the interval include\_endpoints: \[optional\] if set to `True`, the sequence of intermediates will be 'bookended' by `q_start` and `q_end`, resulting in a sequence length of `n + 2`. If set to `False`, endpoints are not included. Defaults to `False`. Yields: A generator object iterating over a sequence of intermediate quaternion objects. Note: This feature only makes sense when interpolating between unit quaternions (those lying on the unit radius hypersphere). Calling this method will implicitly normalise the endpoints to unit quaternions if they are not already unit length. """ step\_size = 1.0 / (n + 1) if include\_endpoints: steps = \[i \* step\_size for i in range(0, n + 2)\] else: steps = \[i \* step\_size for i in range(1, n + 1)\] for step in steps: yield cls.slerp(q0, q1, step) def derivative(self, rate): """Get the instantaneous quaternion derivative representing a quaternion rotating at a 3D rate vector `rate` Params: rate: numpy 3-array (or array-like) describing rotation rates about the global x, y and z axes respectively. Returns: A unit quaternion describing the rotation rate """ rate = self.\_validate\_number\_sequence(rate, 3) return 0.5 \* self \* Quaternion(vector=rate) def integrate(self, rate, timestep): """Advance a time varying quaternion to its value at a time `timestep` in the future. The Quaternion object will be modified to its future value. It is guaranteed to remain a unit quaternion. Params: rate: numpy 3-array (or array-like) describing rotation rates about the global x, y and z axes respectively. timestep: interval over which to integrate into the future. Assuming \*now\* is `T=0`, the integration occurs over the interval `T=0` to `T=timestep`. Smaller intervals are more accurate when `rate` changes over time. Note: The solution is closed form given the assumption that `rate` is constant over the interval of length `timestep`. """ self.\_fast\_normalise() rate = self.\_validate\_number\_sequence(rate, 3) rotation\_vector = rate \* timestep rotation\_norm = np.linalg.norm(rotation\_vector) if rotation\_norm > 0: axis = rotation\_vector / rotation\_norm angle = rotation\_norm q2 = Quaternion(axis=axis, angle=angle) self.q = (self \* q2).q self.\_fast\_normalise() @property def rotation\_matrix(self): """Get the 3x3 rotation matrix equivalent of the quaternion rotation. Returns: A 3x3 orthogonal rotation matrix as a 3x3 Numpy array Note: This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one. """ self.\_normalise() product\_matrix = np.dot(self.\_q\_matrix(), self.\_q\_bar\_matrix().conj().transpose()) return product\_matrix\[1:\]\[:, 1:\] @property def transformation\_matrix(self): """Get the 4x4 homogeneous transformation matrix equivalent of the quaternion rotation. Returns: A 4x4 homogeneous transformation matrix as a 4x4 Numpy array Note: This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one. """ t = np.array(\[\[0.0\], \[0.0\], \[0.0\]\]) Rt = np.hstack(\[self.rotation\_matrix, t\]) return np.vstack(\[Rt, np.array(\[0.0, 0.0, 0.0, 1.0\])\]) @property def yaw\_pitch\_roll(self): """Get the equivalent yaw-pitch-roll angles aka. intrinsic Tait-Bryan angles following the z-y'-x'' convention Returns: yaw: rotation angle around the z-axis in radians, in the range `[-pi, pi]` pitch: rotation angle around the y'-axis in radians, in the range `[-pi/2, -pi/2]` roll: rotation angle around the x''-axis in radians, in the range `[-pi, pi]` The resulting rotation\_matrix would be R = R\_x(roll) R\_y(pitch) R\_z(yaw) Note: This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one. """ self.\_normalise() yaw = np.arctan2(2 \* (self.q\[0\] \* self.q\[3\] - self.q\[1\] \* self.q\[2\]), 1 - 2 \* (self.q\[2\] \*\* 2 + self.q\[3\] \*\* 2)) pitch = np.arcsin(2 \* (self.q\[0\] \* self.q\[2\] + self.q\[3\] \* self.q\[1\])) roll = np.arctan2(2 \* (self.q\[0\] \* self.q\[1\] - self.q\[2\] \* self.q\[3\]), 1 - 2 \* (self.q\[1\] \*\* 2 + self.q\[2\] \*\* 2)) return yaw, pitch, roll def \_wrap\_angle(self, theta): """Helper method: Wrap any angle to lie between -pi and pi Odd multiples of pi are wrapped to +pi (as opposed to -pi) """ result = ((theta + pi) % (2 \* pi)) - pi if result == -pi: result = pi return result def get\_axis(self, undefined=np.zeros(3)): """Get the axis or vector about which the quaternion rotation occurs For a null rotation (a purely real quaternion), the rotation angle will always be `0`, but the rotation axis is undefined. It is by default assumed to be `[0, 0, 0]`. Params: undefined: \[optional\] specify the axis vector that should define a null rotation. This is geometrically meaningless, and could be any of an infinite set of vectors, but can be specified if the default (`[0, 0, 0]`) causes undesired behaviour. Returns: A Numpy unit 3-vector describing the Quaternion object's axis of rotation. Note: This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one. """ tolerance = 1e-17 self.\_normalise() norm = np.linalg.norm(self.vector) if norm < tolerance: # Here there are an infinite set of possible axes, use what has been specified as an undefined axis. return undefined else: return self.vector / norm @property def axis(self): return self.get\_axis() @property def angle(self): """Get the angle (in radians) describing the magnitude of the quaternion rotation about its rotation axis. This is guaranteed to be within the range (-pi:pi) with the direction of rotation indicated by the sign. When a particular rotation describes a 180 degree rotation about an arbitrary axis vector `v`, the conversion to axis / angle representation may jump discontinuously between all permutations of `(-pi, pi)` and `(-v, v)`, each being geometrically equivalent (see Note in documentation). Returns: A real number in the range (-pi:pi) describing the angle of rotation in radians about a Quaternion object's axis of rotation. Note: This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one. """ self.\_normalise() norm = np.linalg.norm(self.vector) return self.\_wrap\_angle(2.0 \* atan2(norm, self.scalar)) @property def degrees(self): return self.to\_degrees(self.angle) @property def radians(self): return self.angle @property def scalar(self): """ Return the real or scalar component of the quaternion object. Returns: A real number i.e. float """ return self.q\[0\] @property def vector(self): """ Return the imaginary or vector component of the quaternion object. Returns: A numpy 3-array of floats. NOT guaranteed to be a unit vector """ return self.q\[1:4\] @property def real(self): return self.scalar @property def imaginary(self): return self.vector @property def w(self): return self.q\[0\] @property def x(self): return self.q\[1\] @property def y(self): return self.q\[2\] @property def z(self): return self.q\[3\] @property def elements(self): """ Return all the elements of the quaternion object. Returns: A numpy 4-array of floats. NOT guaranteed to be a unit vector """ return self.q def \_\_getitem\_\_(self, index): index = int(index) return self.q\[index\] def \_\_setitem\_\_(self, index, value): index = int(index) self.q\[index\] = float(value) def \_\_copy\_\_(self): result = self.\_\_class\_\_(self.q) return result def \_\_deepcopy\_\_(self, memo): result = self.\_\_class\_\_(deepcopy(self.q, memo)) memo\[id(self)\] = result return result @staticmethod def to\_degrees(angle\_rad): if angle\_rad is not None: return float(angle\_rad) / pi \* 180.0 @staticmethod def to\_radians(angle\_deg): if angle\_deg is not None: return float(angle\_deg) / 180.0 \* pi

---

Great question. In that file they implement quaternion multiplication by turning it into **ordinary 4×4 real matrix–vector products**. Concretely they build two fixed 4×4 matrices from a quaternion $q=(w,x,y,z)$ :

### Left-multiplication matrix L(q)L(q)

This is exactly what `Quaternion._q_matrix()` returns.  
Then their `__mul__` does:

i.e. $q \otimes r = L(q)\, r$ where $r$ is the 4-vector `[rw, rx, ry, rz]`.

### Right-multiplication matrix R(q)R(q)

This is what `Quaternion._q_bar_matrix()` returns.  
It satisfies $r \otimes q = R(q)\, r$ .

So when they say “we use matrices,” they mean:

- **Left multiply** a quaternion $r$ by $q$ via **matrix–vector**: $L(q)\,r$ .
- **Right multiply** a quaternion $r$ by $q$ via **matrix–vector**: $R(q)\,r$ .

This is purely an implementation of the Hamilton product as a **linear map on $\mathbb{R}^4$**. It has some nice side-effects:

- Makes `__mul__` one `np.dot` call and keeps the left/right cases straight.
- Lets them compose things like rotations neatly. For example, their `rotation_matrix` builds the 3×3 real rotation from
	(This yields the familiar 3×3 SO(3) matrix for a unit quaternion.)
- In numeric libraries (NumPy/CUDA/BLAS land) you can batch many left-mults or right-mults by stacking quaternions as columns and doing one big matmul.

Key takeaway: they’re **not** saying “we convert to a 3×3 rotation matrix to multiply quaternions.” They’re saying “Hamilton multiplication by a fixed quaternion is linear in the other operand, so we represent it as multiplying by a **4×4** matrix $L(q)$ or $R(q)$ .”

---

why do they represent it as a 4x4 by a 4x4 though? And if we did the same would we still face the same dilemma we currently do about which operations are fastest?

---

Short answer:

- They don’t *always* do “4×4 by 4×4.”  
	In `__mul__` they do **4×4 by 4×1** (matrix–vector): build the left-multiply matrix $L(q)$ and compute $q\otimes r = L(q)\,r$ .
- You do see **4×4 by 4×4** elsewhere (e.g., `rotation_matrix`) because composing left/right linear maps or building a 3×3 SO(3) block naturally ends up as products of those 4×4 maps: $L(q)\,R(q)^\top$ .

Why use matrices at all?

- Hamilton multiplication by a fixed quaternion is a **linear map on $\mathbb{R}^4$**, so it has a 4×4 matrix representation. That makes the code simple (one `dot`) and lets you batch: stack many quaternions as columns $B\in\mathbb{R}^{4\times N}$ and do **one** matmul $C=L(q)B$ .

Does using this matrix form change the “which is faster” dilemma?

- Not really. The same tradeoffs still apply:
	**Pairwise (each pair has its own $q_i$ )**
	- You’d need $C_i=L(q_i) b_i$ for each $i$ . Without reuse, you can’t form one dense GEMM; doing thousands of tiny 4×4×1 matvecs (or strided batched 4×4 GEMMs) rarely beats a **fused arithmetic kernel**. Building/loading a 4×4 for each $q_i$ is extra memory traffic that hurts even more.
	**One-to-many (one fixed $q$ applied to many $b_i$ )**
	- Here the matrix view shines: compute $L(q)$ once and do $C=L(q)B$ with $B\in\mathbb{R}^{4\times N}$ . If $N$ is large, you can map this to a GEMM and hit **Tensor Cores** (with padding/packing to match tile sizes). This is where the matrix representation pays off.
	**Many-to-one** is symmetric to one-to-many using the right-multiply matrix $R(q)$ .

Performance-wise on GPUs (FP16/BF16):

- The **matrix representation is just a way to organize the same math**. Speed depends on *reuse and batch size*:
	- **Reuse present (one $q$ , many $b_i$ )** → form $L(q)$ and use a GEMM on Tensor Cores (pad the 4-row dimension to 8, or pack two quats per column). This can beat a fused kernel once the column count is in the low-thousands and you’re throughput-oriented.
	- **No reuse (pairwise $q_i\otimes b_i$ )** → stick to a **fused kernel** that does the Hamilton product in registers and writes once. Converting each $q_i$ to a 4×4 and then doing a tiny matvec adds overhead and extra loads; Tensor Cores won’t help without heavy packing tricks.

So: representing left/right multiplication as 4×4 matrices is algebraically clean and enables batching, but it **doesn’t** remove the core decision. You’ll still choose:

- **GEMM/Tensor Cores** when a single $L(q)$ or $R(q)$ is reused across a large stack of quaternions (one-to-many / many-to-one).
- **Fused arithmetic** when every product is unique (pairwise many-to-many), or when batch sizes are small/latency-sensitive.

If you want, I can sketch:

- a minimal CUDA fused kernel (FP16/BF16 I/O, FP32 accumulate), and
- a cuBLASLt/CUTLASS snippet that takes $L(q)$ and multiplies a 4×N packed buffer (with an 8-row pad) to light up Tensor Cores.

---

what size dim token would achieve this? once the column count is in the low-thousands and you’re throughput-oriented.

---

Short answer: it’s **not** the token *dimension* that needs to be “in the low-thousands”—it’s the **number of tokens you process at once with the *same* left/right quaternion** (i.e., the **column count** $N$ in a $4 \times N$ stack). Think: **batch size × sequence length × any grouping you fuse into one matmul**.

### What “columns” means here

When we do the Tensor-Core-friendly form

$$
C \;=\; L(q)\;B,\quad L(q)\in\mathbb{R}^{4\times4},\; B\in\mathbb{R}^{4\times N},
$$

the “column count” $N$ is how many distinct quaternions you’re multiplying by the **same** $q$ in one go. It’s **how many tokens are in flight together for that operation**, not the embedding size of a single token.

### Rule-of-thumb thresholds (Ampere/Hopper, FP16/BF16, FP32 accumulate)

- **N ≲ 512–1,000**: fused kernel often wins (lower overhead, less underutilization).
- **N ≈ 1k–4k**: gray zone; either can win depending on your packing (pad rows to 8), streams, and memory traffic.
- **N ≥ 4k–8k**: **Tensor-core GEMM** generally wins and keeps winning as N grows.

(If you pad 4 rows to **8** rows—or pack two quats vertically—you’ll reach TC sweet spots sooner.)

### Concrete ways to reach “low-thousands” columns

You just need enough **tokens-in-flight** for that op with the **same $q$**:

- **Language batch**: batch\_size × seq\_len
	- e.g., 4 sequences × 2048 tokens = **8192 columns** → great for TCs.
	- e.g., 16 × 256 = **4096** → also great.
- **Image-like tokens** (e.g., RGB quats): H×W per image, or multiply across a microbatch
	- 256×256 = **65,536** per image → trivially TC-friendly.
	- 128×128 × batch 4 = **65,536**.
- **Inference microbatching**: microbatch 32 requests × 128 tokens each = **4096**.

If your pipeline gives each token its **own $q_i$** (no reuse), then you’re in the **pairwise** case and this TC trick doesn’t apply; use the fused arithmetic kernel. If you can **restructure** to reuse a $q$ across many tokens (or collect per-type operations), you can hit those N values and let Tensor Cores rip.

### Quick checklist

- Are you applying the **same quaternion** to many tokens right now?
	- **Yes** → stack those tokens so $N \ge 4k$ (ideally), pad 4→8 rows, use cuBLASLt/CUTLASS.
	- **No** (each token has its own $q_i$ ) → **fused kernel**.
- Embedding **dimension** (e.g., 512) doesn’t determine this; it’s about **how many columns (tokens)** you run at once for that multiply.
- If you’re close (e.g., $N\sim 1\!-\!2k$ ), you can:
	- **Microbatch/fuse** neighboring work to increase $N$ .
	- **Pack** two quaternions per column to make the row dimension 8 (better TC tiling).
	- **Use grouped matmul** if you have a few different $q$ ’s, each reused across a decent chunk of tokens.

---

we're assuming each of the tokens has a distinct value, so in that case it is the dim that matters

---

Totally fair—if every token has its **own** quaternion $q_i$ , then the “reuse-one- $q$ ” trick doesn’t apply. But two different cases get conflated:

## Case A — Up-projection with a shared weight (your 512×4)

You’re doing

$$
Y = W\,Q,\quad W\in\mathbb{R}^{D\times 4},\; Q\in\mathbb{R}^{4\times N},
$$

where $Q=[q_1,\dots,q_N]$ are **distinct per token**, and $W$ is **shared**. Here the **dimension $D$** *does* matter, because this is a regular GEMM:

- $M=D$ (rows), $K=4$ , $N=$ tokens in flight.
- Tensor cores want FP16/BF16 and tile-friendly sizes. Since $K=4$ is too small for TC’s preferred $K$ multiple of 8, **pad $K$ to 8** (just add 4 zeros per column) or pack two quaternions vertically so the effective $K=8$ .
- With $D\ge 256$ (e.g., 512), you **don’t** need “thousands of tokens” to win. As a rule of thumb on Ampere/Hopper:
	- $D=512, K=8$ (padded), **$N\ge 64\!-\!128$** already feeds tensor cores well.
	- $N\ge 256$ is great; beyond that you’re just chasing saturation.
- So yes: in this **shared $W$** scenario, **dim $D$** is the lever, and modest batch/seq ( $N$ ) is enough once you pad $K$ to 8. Use cuBLASLt/CUTLASS with FP32 accumulate.

## Case B — Pairwise Hamilton products ci=qi⊗bic\_i=q\_i\\otimes b\_i

Here each output uses a different left/right multiplier; no shared matrix. You can’t bundle them into one dense GEMM without block-diagonal packing. In this case:

- The work per pair is ~28 FLOPs on ~24 bytes I/O (FP16/BF16) → **memory-bound**.
- “Dim” doesn’t help; you’re not doing a $D\times 4$ map, you’re doing **elementwise quaternion products**.
- Best is a **fused arithmetic kernel** (do the 16 mul/12 add in FP32, store FP16/BF16). Tensor cores won’t help unless you do heavy packing tricks (usually not worth it).

## Quick rules

- **Your up-projection (shared $W$ )**: make it a GEMM. Pad $K=4\to 8$ . With $D=512$ , **$N\ge 64$** already benefits from tensor cores; **$N\ge 128\!-\!256$** is solid.
- **Per-token Hamilton products** (no shared left/right $q$ ): stay with a **fused kernel**; “dim” doesn’t change the memory-bound nature.

If you want, I can jot a cuBLASLt config for $D\times 8$ × $8\times N$ (BF16→FP32 accumulate) and a minimal fused kernel for the pairwise path so you can A/B them.

1/2