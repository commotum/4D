Below is a best-effort **timeline** of papers that match your “**positional encoding improvement proposal**” classification (Transformer attention as the core setting; paper explicitly critiques prior PE/RPE; core contribution is a **change to positional encoding / position injection**, not “just add more dims”). Where something is **borderline** (e.g., PE is important but not the paper’s main thrust), I mark it.

---

## 2018 — Relative position inside attention (the modern “RPE” lineage)

* **Self-Attention with Relative Position Representations** (Shaw, Uszkoreit, Vaswani, 2018)
  **Critique:** absolute position added to inputs is an awkward fit for attention’s permutation-invariance; poor inductive bias for distance.
  **Improvement:** inject **relative distance embeddings** directly into attention (key/value side). ([arXiv][1])

---

## 2019 — Relative PE designed for long-context reuse

* **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context** (Dai et al., 2019)
  **Critique:** absolute PE breaks “state reuse” across segments and causes temporal confusion in recurrence/memory.
  **Improvement:** a **relative positional encoding formulation** compatible with segment-level recurrence for long-context LM. ([arXiv][2])

---

## 2020 — “Content vs position” disentangling / untying

* **DeBERTa: Decoding-enhanced BERT with Disentangled Attention** (He et al., 2020)
  **Critique:** standard position injection entangles content/position too early and too tightly.
  **Improvement:** **disentangled attention** that separately models content and (relative) position in attention scoring. ([arXiv][3])

* **Rethinking Positional Encoding in Language Pre-training (TUPE)** (Ke et al., ICLR-era OpenReview, ~2020/2021)
  **Critique:** mixing token and position correlations (and treating `[CLS]` position like ordinary tokens) is noisy / suboptimal.
  **Improvement:** **untie** positional and token projections; separate positional correlation path (TUPE variants). ([OpenReview][4])

---

## 2021 — Big year: RoPE, bias-only PE, and vision-focused RPE

* **RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)** (Su et al., 2021)
  **Critique:** absolute PE is inflexible; many relative schemes don’t mesh cleanly with some attention variants.
  **Improvement:** **rotary** position embedding that makes relative offsets emerge naturally in dot-products. ([arXiv][5])

* **Train Short, Test Long: Attention with Linear Biases (ALiBi)** (Press, Smith, Lewis, 2021)
  **Critique:** common PEs extrapolate poorly to longer lengths than trained.
  **Improvement:** remove token-additive PE; add a **distance-proportional linear bias** directly to attention logits. ([arXiv][6])

* **Rethinking and Improving Relative Position Encoding for Vision Transformer (iRPE)** (Wu et al., ICCV 2021)
  **Critique:** “RPE works well in NLP” but is **unclear/controversial in vision**; existing methods have tradeoffs when flattened to 2D images.
  **Improvement:** **2D image-aware RPE variants** (directional distance modeling + attention interaction design). ([arXiv][7])

* **A Simple and Effective Positional Encoding for Transformers** (Chen et al., EMNLP 2021)
  **Critique:** absolute PE doesn’t directly express relative relations; prior RPE designs have practical issues.
  **Improvement:** proposes a simplified, effective PE/RPE scheme (language setting). ([ACL Anthology][8])

* **Swin Transformer (relative position bias)** (Liu et al., 2021) — **borderline** (PE isn’t the main thesis, but it is a deliberate PE choice)
  **Critique:** absolute position embeddings in ViTs are less suited to shifted-window attention and spatial generalization.
  **Improvement:** **learnable relative position bias** within windows. ([arXiv][9])

---

## 2022 — Formalizing extrapolatable relative PE + fixing RoPE drift

* **KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation** (Chi et al., NeurIPS 2022)
  **Critique:** existing RPE variants extrapolate inconsistently and lack a unifying principle.
  **Improvement:** **kernelize** positional differences via CPD/PD kernel framework to derive extrapolatable RPEs. ([arXiv][10])

* **XPos / “Length-Extrapolatable Transformer” line (RoPE + decay)** (Sun et al., circulated 2022; widely cited 2023)
  **Critique:** RoPE can become unstable / degrade for long extrapolation.
  **Improvement:** add an **exponential decay** to RoPE rotation (XPos) to stabilize long-range behavior. ([arXiv][11])

---

## 2023 — Context extension via “position remapping” and RoPE scaling

* **Extending Context Window of Large Language Models via Positional Interpolation (PI)** (Chen et al., 2023)
  **Critique:** direct extrapolation past trained context can blow up attention scores and fail catastrophically.
  **Improvement:** **down-scale/interpolate position indices** so inference stays in-range for RoPE; minimal finetune. ([arXiv][12])

* **YaRN: Efficient Context Window Extension of Large Language Models** (Peng et al., 2023 preprint; later ICLR)
  **Critique:** RoPE models fail to generalize beyond training length; existing extension methods are compute-hungry.
  **Improvement:** **frequency-aware RoPE interpolation** + related scaling tricks for efficient long-context finetuning. ([arXiv][13])

* **A Length-Extrapolatable Transformer** (Sun et al., ACL 2023)
  **Critique:** length extrapolation is tied to attention’s positional “resolution.”
  **Improvement:** design changes including **relative position embedding** targeted at improving extrapolation indicators. ([ACL Anthology][14])

---

## 2024 — Refining RoPE extension + vision-specific rotary fixes + directed-attention PE replacements

* **Resonance RoPE: Improving Context Length Generalization of Large Language Models** (Wang et al., 2024)
  **Critique:** RoPE interpolation/remapping still leaves an OOD gap for positions.
  **Improvement:** refine RoPE feature interpolation for OOD positions (“resonance” shaping). ([arXiv][15])

* **Rotary Position Embedding for Vision Transformer (RoPE-Mixed / improved 2D RoPE)** (Heo et al., ECCV 2024)
  **Critique:** common **axial 2D RoPE** misses diagonal structure important in vision.
  **Improvement:** **mixed-axis frequency** 2D RoPE variant for images. ([ECVA][16])

* **LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate** (Fuller et al., NeurIPS 2024)
  **Critique:** standard patch position encodings cause a distribution shift when changing patch counts / resolution.
  **Improvement:** “drop-in replacement” position handling using **directed attention heads + 2D masks / distance-penalized bias** to improve extrapolation. ([arXiv][17])

* **(RoPE scaling analysis + new scaling)** EMNLP 2024 paper on RoPE scaling methods (Wu et al., 2024)
  **Critique:** many RoPE scaling methods are empirical and poorly grounded in RoPE’s internal distribution.
  **Improvement:** proposes a more principled scaling approach (within the RoPE-scaling family). ([ACL Anthology][18])

* **Exploring Context Window of LLMs via … (NTK-aware / scaled RoPE discussion)** (Dong et al., NeurIPS 2024) — **borderline** (more analysis-heavy, but includes PE scaling focus)
  ([NeurIPS Proceedings][19])

---

## 2025 — Explicit “what vs where” separation + context-aware biases

* **Decoupling the “What” and “Where” With Polar Coordinate Positional Embeddings (PoPE)** (Gopalakrishnan et al., 2025)
  **Critique:** RoPE entangles content (“what”) and position (“where”), harming tasks needing independent matches.
  **Improvement:** **PoPE** polar-coordinate positional embedding removing the confound; improves zero-shot length extrapolation. ([arXiv][20])

* **Context-aware Biases for Length Extrapolation (CABLE)** (Veisi et al., EMNLP 2025)
  **Critique:** fixed-form biases/PEs can be too rigid for long-context generalization.
  **Improvement:** learns **context-aware additive relative positional biases**. ([ACL Anthology][21])

---

### References

[1]: https://arxiv.org/abs/1803.02155?utm_source=chatgpt.com "Self-Attention with Relative Position Representations"
[2]: https://arxiv.org/abs/1901.02860?utm_source=chatgpt.com "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
[3]: https://arxiv.org/abs/2006.03654?utm_source=chatgpt.com "DeBERTa: Decoding-enhanced BERT with Disentangled ..."
[4]: https://openreview.net/pdf?id=09-528y2Fgf&utm_source=chatgpt.com "RETHINKING POSITIONAL ENCODING IN LANGUAGE ..."
[5]: https://arxiv.org/abs/2104.09864?utm_source=chatgpt.com "RoFormer: Enhanced Transformer with Rotary Position Embedding"
[6]: https://arxiv.org/abs/2108.12409?utm_source=chatgpt.com "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
[7]: https://arxiv.org/abs/2107.14222?utm_source=chatgpt.com "Rethinking and Improving Relative Position Encoding for Vision Transformer"
[8]: https://aclanthology.org/2021.emnlp-main.236.pdf?utm_source=chatgpt.com "A Simple and Effective Positional Encoding for Transformers"
[9]: https://arxiv.org/pdf/2103.14030?utm_source=chatgpt.com "arXiv:2103.14030v2 [cs.CV] 17 Aug 2021"
[10]: https://arxiv.org/abs/2205.09921?utm_source=chatgpt.com "KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation"
[11]: https://arxiv.org/pdf/2212.10554?utm_source=chatgpt.com "arXiv:2212.10554v1 [cs.CL] 20 Dec 2022"
[12]: https://arxiv.org/abs/2306.15595?utm_source=chatgpt.com "Extending Context Window of Large Language Models via Positional Interpolation"
[13]: https://arxiv.org/abs/2309.00071?utm_source=chatgpt.com "YaRN: Efficient Context Window Extension of Large Language Models"
[14]: https://aclanthology.org/2023.acl-long.816.pdf?utm_source=chatgpt.com "A Length-Extrapolatable Transformer"
[15]: https://arxiv.org/abs/2403.00071?utm_source=chatgpt.com "[2403.00071] Resonance RoPE: Improving Context Length ..."
[16]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01584.pdf?utm_source=chatgpt.com "Rotary Position Embedding for Vision Transformer"
[17]: https://arxiv.org/abs/2405.13985?utm_source=chatgpt.com "LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate"
[18]: https://aclanthology.org/2024.emnlp-main.414.pdf?utm_source=chatgpt.com "Extending Context Window of Large Language Models ..."
[19]: https://proceedings.neurips.cc/paper_files/paper/2024/file/1403ab1a427050538ec59c7f570aec8b-Paper-Conference.pdf?utm_source=chatgpt.com "Exploring Context Window of Large Language Models via ..."
[20]: https://arxiv.org/abs/2509.10534?utm_source=chatgpt.com "Decoupling the \"What\" and \"Where\" With Polar Coordinate Positional Embeddings"
[21]: https://aclanthology.org/2025.emnlp-main.1545.pdf?utm_source=chatgpt.com "Context-aware Biases for Length Extrapolation"

---

Here’s a **second-pass (aggressive) sweep focused on 2024–2025**, prioritizing papers that **explicitly** do “*critique → propose new/modified positional encoding (or attention-logit positional bias)*,” even when the title markets itself as **context extension**.

I’m listing only items that *strongly* match your criteria; a few “analysis-forward” papers that are often confused for PE proposals are marked **(borderline / analysis)**.

---

## 2024

* **LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens** (Ding et al., 2024)
  **Critique:** naïve RoPE scaling / long-context extension causes non-uniformity and mismatch issues as length grows.
  **Improvement:** a **RoPE-extension scheme** (non-uniform / optimized adjustments) explicitly framed as a PE-side fix for very long contexts. ([arXiv][1])

* **HiRoPE: Length Extrapolation for Code Models Using Hierarchical Rotary Position Embedding** (Zhang et al., ACL 2024)
  **Critique:** “flat” RoPE doesn’t respect code’s hierarchical structure; long-code completion stresses standard position handling.
  **Improvement:** **Hierarchical RoPE** (structure-aware rotary positions) as the core mechanism for length extrapolation in code LMs. ([ACL Anthology][2])

* **Resonance RoPE: Improving Context Length Generalization of Large Language Models** (Wang et al., 2024)
  **Critique:** existing RoPE interpolation/remapping approaches still show distribution shift / generalization loss at longer lengths.
  **Improvement:** a **refined RoPE modification** (“resonance”-style shaping) aimed at stronger length generalization. ([ACL Anthology][3])

* **Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs** (Ma et al., NeurIPS 2024)
  **Critique:** explains why **NoPE** and standard PE fail beyond an “effective range” and argues PE can be extended with the right design.
  **Improvement:** introduces **weave PE** + **Stair PE** and the **Mesa-Extrapolation** method (chunk/triangular attention + weave PE) as a PE-centric route to extrapolation. ([arXiv][4])

* **Rotary Position Embedding for Vision Transformer** (Heo et al., 2024)
  **Critique:** RoPE is underexplored in vision; 2D usage is non-trivial and common implementations have gaps.
  **Improvement:** **practical 2D RoPE implementations/variants** with analysis and recommended design choices for ViTs. ([arXiv][5])

* **Length Generalization of Causal Transformers without Position Encoding** (Wang et al., Findings ACL 2024) — **borderline (NoPE-centric, but still “position mechanism”)**
  **Critique:** explicit PEs aren’t the only way; NoPE can generalize but has failure modes tied to attention distribution.
  **Improvement:** proposes **head temperature tuning** to expand NoPE’s usable context (not a PE per se, but a direct critique+fix for position handling). ([arXiv][6])

---

## 2025

* **HARPE: Head-Adaptive Rotary Position Encoding** (Lian et al., COLING 2025)
  **Critique:** multi-stage long-context training + manual RoPE base tuning is brittle; single-stage training with one big base can be suboptimal.
  **Improvement:** **per-head RoPE base frequencies** (head-adaptive RoPE) trained directly toward target context length. ([ACL Anthology][7])

* **ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices** (Yu et al., 2025; CVPR 2025 version exists)
  **Critique:** RoPE’s fixed, hand-defined rotation matrices limit the transformation space and flexibility/robustness.
  **Improvement:** generalizes RoPE using **trainable commuting angle matrices**, framing commutativity as key for offset-consistent behavior. ([arXiv][8])

* **CABLE: Context-aware Biases for Length Extrapolation** (Veisi et al., 2025)
  **Critique:** static/additive RPE biases (or fixed PE schemes) can be too rigid; long-context behavior benefits from context-conditioning.
  **Improvement:** **context-conditioned positional bias scores** added to attention logits (a dynamic PE/bias mechanism). ([ACL Anthology][9])

* **Wavelet-based Positional Representation for Long Context** (Oka et al., ICLR 2025)
  **Critique:** frames **ALiBi** as behaving like windowed attention and argues it struggles with deep dependencies due to receptive-field limitations.
  **Improvement:** a **wavelet-transform-based positional representation** to capture multiple scales without restricting attention’s field. ([OpenReview][10])

* **Understanding the RoPE Extensions of Long-Context LLMs** (Zhong et al., COLING 2025) — **borderline / analysis**
  **Critique:** many RoPE extensions are used in practice without a clear attention-perspective explanation.
  **Contribution:** primarily **analysis/understanding**, not a brand-new PE (useful as a map of the space, but not itself a PE proposal). ([ACL Anthology][11])

* **CoPE: A Lightweight Complex Positional Encoding** (Amballa, 2025)
  **Critique:** traditional PE methods have limitations (especially for long sequences), and some exhibit long-term decay or incompatibilities.
  **Improvement:** replaces PE with **complex-valued encoding** (real=content, imag=position) + **phase-aware attention** in early layers. ([arXiv][12])

* **TAPA: Positional Encoding via Token-Aware Phase Attention** (Wang et al., 2025)
  **Critique:** argues RoPE introduces an intrinsic distance-dependent bias limiting long-context modeling; many RoPE extensions are post-hoc retuning.
  **Improvement:** inserts a **learnable phase function into attention** (token-aware phase attention) as a new PE mechanism for extrapolation. ([arXiv][13])

---

### Quick “what you might have missed” buckets (2024–2025)

* **“RoPE variants that change the *generator* of rotation”**: ComRoPE ([arXiv][8])
* **“RoPE variants that change *which heads* get which frequencies”**: HARPE ([ACL Anthology][7])
* **“Non-RoPE PE that rewires *how position enters attention*”**: TAPA ([arXiv][13]), CoPE ([arXiv][12])
* **“Bias/logit-level position that becomes *context-dependent*”**: CABLE ([ACL Anthology][9])
* **“Multi-scale / signal-processing inspired position”**: Wavelet-based PR ([arXiv][14])
* **“Chunk/weave PE for extrapolation without full retraining”**: Mesa-Extrapolation + Stair/Weave PE ([arXiv][4])

[1]: https://arxiv.org/pdf/2402.13753?utm_source=chatgpt.com "LongRoPE: Extending LLM Context Window Beyond 2 ..."
[2]: https://aclanthology.org/2024.acl-long.735/?utm_source=chatgpt.com "Length Extrapolation for Code Models Using Hierarchical ..."
[3]: https://aclanthology.org/2024.findings-acl.32.pdf?utm_source=chatgpt.com "Resonance RoPE: Improving Context Length ..."
[4]: https://arxiv.org/abs/2410.15859?utm_source=chatgpt.com "Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs"
[5]: https://arxiv.org/abs/2403.13298?utm_source=chatgpt.com "Rotary Position Embedding for Vision Transformer"
[6]: https://arxiv.org/abs/2404.12224?utm_source=chatgpt.com "Length Generalization of Causal Transformers without Position Encoding"
[7]: https://aclanthology.org/2025.coling-main.326/?utm_source=chatgpt.com "A Novel Single-Stage Approach to Long Context Extension ..."
[8]: https://arxiv.org/abs/2506.03737?utm_source=chatgpt.com "ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices"
[9]: https://aclanthology.org/2025.emnlp-main.1545.pdf?utm_source=chatgpt.com "Context-aware Biases for Length Extrapolation"
[10]: https://openreview.net/forum?id=OhauMUNW8T&utm_source=chatgpt.com "Wavelet-based Positional Representation for Long Context"
[11]: https://aclanthology.org/2025.coling-main.600.pdf?utm_source=chatgpt.com "Understanding the RoPE Extensions of Long-Context LLMs"
[12]: https://arxiv.org/abs/2508.18308?utm_source=chatgpt.com "CoPE: A Lightweight Complex Positional Encoding"
[13]: https://arxiv.org/abs/2509.12635?utm_source=chatgpt.com "Positional Encoding via Token-Aware Phase Attention"
[14]: https://arxiv.org/html/2502.02004v1?utm_source=chatgpt.com "Wavelet-based Positional Representation for Long Context"

---

Below is a **timeline containing every paper Jianlin Su explicitly mentions in that post**, in the same style as before (year → title → critique → improvement). I’m treating “mentions” literally: if it appears as a cited paper/model in the blog text, it’s in here.

---

## 2017 — Trainable absolute PE becomes the default in early seq models

* **Convolutional Sequence to Sequence Learning** (Gehring et al., 2017)
  **Critique:** (context in the blog) early seq models still needed an explicit mechanism to inject order.
  **Improvement:** uses **trainable absolute positional embeddings** (as a learnable table) rather than a fixed formula. ([Scientific Spaces][1])

* **Attention Is All You Need** (Vaswani et al., 2017)
  **Critique:** self-attention is permutation-invariant; without position, token order is lost.
  **Improvement:** proposes **sinusoidal (trigonometric) absolute positional encoding** with a closed-form rule enabling extrapolation in principle. ([Scientific Spaces][1])

---

## 2018 — Classic relative position in attention

* **Self-Attention with Relative Position Representations** (Shaw, Uszkoreit, Vaswani, 2018)
  **Critique:** absolute PE added to inputs is not the only (or best) way; attention can be made position-aware directly.
  **Improvement:** inject **relative position representations** into attention (key/value-side modifications), becoming the “classic style” RPE template. ([Scientific Spaces][1])

* **BERT** (Devlin et al., 2018)
  **Critique:** (implied via blog framing) transformers still require PE; the simplest form is learnable absolute tables.
  **Improvement:** uses **trainable absolute positional embeddings** (learned lookup). ([Scientific Spaces][1])

* **GPT** (Radford et al., 2018)
  **Critique:** same framing as BERT in the blog—absolute learned PE is straightforward but not extrapolatable by default.
  **Improvement:** uses **trainable absolute positional embeddings** (learned lookup). ([Scientific Spaces][1])

---

## 2019 — Relative PE redesigned for segment recurrence + a named “style” emerges

* **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context** (Dai et al., 2019)
  **Critique:** absolute PE clashes with segment-level recurrence/memory (positions don’t stay consistent across segments).
  **Improvement:** the “**XLNet style**” decomposition: relative-position terms in the attention score with trainable vectors (u, v) and a sinusoidal-based relative signal (no truncation in the described form). ([Scientific Spaces][1])

* **XLNet: Generalized Autoregressive Pretraining for Language Understanding** (Yang et al., 2019)
  **Critique:** (blog context) popularized Transformer-XL’s relative-position attention and helped cement the “XLNet style” naming.
  **Improvement:** adopts Transformer-XL’s **relative positional attention formulation** in a strong pretraining regime. ([Scientific Spaces][1])

* **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)** (Raffel et al., 2019)
  **Critique:** argues for decoupling input content and position interactions; simplifies positional handling.
  **Improvement:** “**T5 style**” relative PE: **learned relative position bias** added directly to attention logits, with **relative-position bucketing** (fine bins nearby, coarser bins far away). ([Scientific Spaces][1])

---

## 2020 — Learned position dynamics, disentangling schemes, and “unconventional” PE

* **Learning to Encode Position for Transformer with Continuous Dynamical Model (FLOATER)** (ICML 2020)
  **Critique:** fixed-form absolute PE is restrictive; want a learnable generator with extrapolation-friendly behavior.
  **Improvement:** model positional encoding via a **continuous dynamical system / Neural ODE** (recursive/continuous-time PE). ([Scientific Spaces][1])

* **Rethinking Positional Encoding in Language Pre-training (TUPE)** (Ke et al., referenced as ICLR 2021 in the blog)
  **Critique:** content and position correlations shouldn’t be overly entangled; you can simplify what position contributes to attention.
  **Improvement:** introduces a **position-aware attention bias / decoupled treatment** of positional effects (presented in the blog as closely related in spirit to the T5-style bias term). ([Scientific Spaces][1])

* **DeBERTa: Decoding-enhanced BERT with Disentangled Attention** (He et al., 2020)
  **Critique:** different combinations of content–position interaction terms matter; T5’s “keep only bias” is not the only choice.
  **Improvement:** “**DeBERTa style**” relative PE: discards the pure position–position term while **keeping and reintroducing** the input–position and position–input terms via relative encodings (a different permutation of the qk expansion terms). ([Scientific Spaces][1])

* **How Much Position Information Do Convolutional Neural Networks Encode?** (ICLR 2020)
  **Critique:** CNNs “seem” position-aware without explicit PE—why?
  **Improvement:** shows position can be **leaked via zero padding**, effectively giving distance-to-boundary signals (included as a “broadening horizons” contrast to attention). ([Scientific Spaces][1])

* **Encoding Word Order in Complex Embeddings** (ICLR 2020)
  **Critique:** real-valued PE schemes aren’t the only route; order can be encoded via phase/complex structure.
  **Improvement:** “**complex number style**” positional encoding (Complex Order) and uses it in a **fully complex-valued modeling pipeline**, not merely as a real-valued trick. ([Scientific Spaces][1])

---

Absolutely — same timeline format, now with **canonical arXiv / PDF links** for each paper Su mentions.

(Per your note: **all of these share the same blog-post reference**, so I’m not repeating the blog citation per item; I’m just attaching the **paper links**.)

---

## 2017 — Trainable absolute PE becomes the default in early seq models

* **Convolutional Sequence to Sequence Learning** (Gehring et al., 2017)
  **Critique:** (context in the post) early seq models still needed an explicit mechanism to inject order.
  **Improvement:** uses **trainable absolute positional embeddings** (as a learnable table) rather than a fixed formula. ([arXiv][1])
  **Links (arXiv / PDF):**

  ```text
  arXiv: https://arxiv.org/abs/1705.03122
  PDF:   https://arxiv.org/pdf/1705.03122
  PMLR:  https://proceedings.mlr.press/v70/gehring17a/gehring17a.pdf
  ```

* **Attention Is All You Need** (Vaswani et al., 2017)
  **Critique:** self-attention is permutation-invariant; without position, token order is lost.
  **Improvement:** proposes **sinusoidal (trigonometric) absolute positional encoding** with a closed-form rule enabling extrapolation in principle. ([arXiv][2])
  **Links (arXiv / PDF):**

  ```text
  arXiv: https://arxiv.org/abs/1706.03762
  PDF:   https://arxiv.org/pdf/1706.03762
  NeurIPS page: https://papers.nips.cc/paper/7181-attention-is-all-you-need
  ```

---

## 2018 — Classic relative position in attention

* **Self-Attention with Relative Position Representations** (Shaw, Uszkoreit, Vaswani, 2018)
  **Critique:** absolute PE added to inputs is not the only (or best) way; attention can be made position-aware directly.
  **Improvement:** inject **relative position representations** into attention (key/value-side modifications), becoming the “classic style” RPE template. ([arXiv][3])
  **Links (arXiv / PDF):**

  ```text
  arXiv: https://arxiv.org/abs/1803.02155
  PDF:   https://arxiv.org/pdf/1803.02155
  ACL Anthology: https://aclanthology.org/N18-2074/
  ```

* **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (Devlin et al., 2018)
  **Critique:** (implied via Su’s framing) transformers still require PE; the simplest form is learnable absolute tables.
  **Improvement:** uses **trainable absolute positional embeddings** (learned lookup). ([arXiv][4])
  **Links (arXiv / PDF):**

  ```text
  arXiv: https://arxiv.org/abs/1810.04805
  PDF:   https://arxiv.org/pdf/1810.04805
  ACL PDF: https://aclanthology.org/N19-1423.pdf
  ```

* **Improving Language Understanding by Generative Pre-Training (GPT-1)** (Radford et al., 2018)
  **Critique:** same framing as BERT in the post—absolute learned PE is straightforward but not extrapolatable by default.
  **Improvement:** uses **trainable absolute positional embeddings** (learned lookup). ([OpenAI][5])
  **Links (PDF):**

  ```text
  PDF (OpenAI): https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
  ```

---

## 2019 — Relative PE redesigned for segment recurrence + a named “style” emerges

* **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context** (Dai et al., 2019)
  **Critique:** absolute PE clashes with segment-level recurrence/memory (positions don’t stay consistent across segments).
  **Improvement:** the “**XLNet style**” decomposition: relative-position terms in the attention score with trainable vectors (u, v) and a sinusoidal-based relative signal (as described by Su). ([arXiv][6])
  **Links (arXiv / PDF):**

  ```text
  arXiv: https://arxiv.org/abs/1901.02860
  ACL PDF: https://aclanthology.org/P19-1285.pdf
  ```

* **XLNet: Generalized Autoregressive Pretraining for Language Understanding** (Yang et al., 2019)
  **Critique:** (Su’s context) popularized Transformer-XL’s relative-position attention and helped cement the “XLNet style” naming.
  **Improvement:** adopts Transformer-XL’s **relative positional attention formulation** in a strong pretraining regime. ([arXiv][7])
  **Links (arXiv / PDF):**

  ```text
  arXiv: https://arxiv.org/abs/1906.08237
  PDF:   https://arxiv.org/pdf/1906.08237
  ```

* **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)** (Raffel et al., 2019)
  **Critique:** argues for decoupling input content and position interactions; simplifies positional handling.
  **Improvement:** “**T5 style**” relative PE: **learned relative position bias** added directly to attention logits, with **relative-position bucketing**. ([arXiv][8])
  **Links (arXiv / PDF):**

  ```text
  arXiv: https://arxiv.org/abs/1910.10683
  PDF:   https://arxiv.org/pdf/1910.10683
  JMLR PDF: https://jmlr.org/papers/volume21/20-074/20-074.pdf
  ```

---

## 2020 — Learned position dynamics, disentangling schemes, and “unconventional” PE

* **Learning to Encode Position for Transformer with Continuous Dynamical Model (FLOATER)** (Liu et al., ICML 2020)
  **Critique:** fixed-form absolute PE is restrictive; want a learnable generator with extrapolation-friendly behavior.
  **Improvement:** model positional encoding via a **continuous dynamical system / Neural ODE** (recursive/continuous-time PE). ([arXiv][9])
  **Links (arXiv / PDF):**

  ```text
  arXiv: https://arxiv.org/abs/2003.09229
  PDF:   https://arxiv.org/pdf/2003.09229
  ```

* **Rethinking Positional Encoding in Language Pre-training (TUPE)** (Ke et al., 2020; published ICLR 2021)
  **Critique:** addition of word+position embeddings mixes heterogeneous correlations; `[CLS]` shouldn’t be treated like ordinary positions.
  **Improvement:** **untied positional encoding**: separate parameterizations for word-context correlation vs positional correlation, then combine. ([arXiv][10])
  **Links (arXiv / PDF):**

  ```text
  arXiv: https://arxiv.org/abs/2006.15595
  PDF:   https://arxiv.org/pdf/2006.15595
  OpenReview PDF: https://openreview.net/pdf?id=09-528y2Fgf
  ```

* **DeBERTa: Decoding-enhanced BERT with Disentangled Attention** (He et al., 2020; published ICLR 2021)
  **Critique:** different “qk expansion” interaction terms matter; position/content interaction should be structured, not entangled.
  **Improvement:** “**DeBERTa style**” relative PE: disentangled attention that separates content and position vectors and their interactions. ([arXiv][11])
  **Links (arXiv / PDF):**

  ```text
  arXiv: https://arxiv.org/abs/2006.03654
  PDF:   https://arxiv.org/pdf/2006.03654
  ```

* **How Much Position Information Do Convolutional Neural Networks Encode?** (Islam, Jia, Bruce, 2020)
  **Critique:** CNNs appear position-aware without explicit PE—where does that come from?
  **Improvement:** argues a lot of absolute position can be **leaked via padding/boundaries** (Su highlights zero-padding leakage). ([arXiv][12])
  **Links (arXiv / PDF):**

  ```text
  arXiv: https://arxiv.org/abs/2001.08248
  OpenReview PDF: https://openreview.net/pdf/2267055f8221e283014aba7ef46092ba93ff450f.pdf
  ```

* **Encoding Word Order in Complex Embeddings** (Wang et al., 2019; published ICLR 2020)
  **Critique:** standard position embeddings capture absolute positions but not richer order relations; real-valued PE isn’t the only route.
  **Improvement:** “**complex number style**” encoding (Complex Order) and a **complex-valued network pipeline** to exploit phase/order structure. ([arXiv][13])
  **Links (arXiv / PDF):**

  ```text
  arXiv: https://arxiv.org/abs/1912.12333
  PDF:   https://arxiv.org/pdf/1912.12333
  OpenReview PDF: https://openreview.net/pdf?id=Hke-WTVtwr
  ```

---

If you want, next round we can do something very mechanical (and satisfying): take this “Su-mentions list” and tag each entry as **(A)** “direct PE proposal,” **(B)** “model that uses PE,” or **(C)** “position analysis/other,” while keeping the exact same timeline look.

[1]: https://arxiv.org/abs/1705.03122?utm_source=chatgpt.com "Convolutional Sequence to Sequence Learning"
[2]: https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com "[1706.03762] Attention Is All You Need"
[3]: https://arxiv.org/abs/1803.02155?utm_source=chatgpt.com "Self-Attention with Relative Position Representations"
[4]: https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
[5]: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf?utm_source=chatgpt.com "Improving Language Understanding by Generative Pre- ..."
[6]: https://arxiv.org/abs/1901.02860?utm_source=chatgpt.com "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
[7]: https://arxiv.org/abs/1906.08237?utm_source=chatgpt.com "XLNet: Generalized Autoregressive Pretraining for Language Understanding"
[8]: https://arxiv.org/abs/1910.10683?utm_source=chatgpt.com "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
[9]: https://arxiv.org/abs/2003.09229?utm_source=chatgpt.com "Learning to Encode Position for Transformer with Continuous Dynamical Model"
[10]: https://arxiv.org/abs/2006.15595?utm_source=chatgpt.com "Rethinking Positional Encoding in Language Pre-training"
[11]: https://arxiv.org/abs/2006.03654?utm_source=chatgpt.com "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"
[12]: https://arxiv.org/abs/2001.08248?utm_source=chatgpt.com "How Much Position Information Do Convolutional Neural Networks Encode?"
[13]: https://arxiv.org/abs/1912.12333?utm_source=chatgpt.com "Encoding word order in complex embeddings"
