# **Research Findings/Timeline:** **Towards a Spatiotemporally Aware** **Transformer Architecture for ARC-AGI**

# **Table of Contents**

1. **Initial Objective: A Transformer Pipeline for ARC-AGI via Code-Image Compression**  
    1.1. The ARC-AGI Challenge: A Test of Fluid Intelligence  
    1.2. The p5.js Hypothesis: Bridging Code and Visual Abstraction  
    1.3. Proposed Encoder-Decoder Architecture  
    1.4. Latent Space and Tokenization Goals  
    1.5. Optimization Strategy: Reconstruction Loss

2. **The Flattening Problem: Realizing the Limitations of Sequential Spatialization**  
    2.1. Standard Transformer Approaches to Multidimensional Data  
    2.2. Loss of Structural Information and Its Implications for ARC-AGI  
    2.3. The “Jagged Edge” of LLM Capabilities

3. **A New Trajectory: Generalizing RoPE for True 4D Positional Understanding**  
    3.1. The Need for Robust Positional Encodings  
    3.2. Rotary Positional Embeddings (RoPE) as a Starting Point  
    3.3. Limitations of Real Vector Representations for Geometric Data  
    3.4. Exploring Higher Division Algebras: Quaternions and Clifford Algebra

4. **Breakthrough: MonSTER – Minkowski Space-Time Embedding Rotors**  
    4.1. Conceptual Foundation: Clifford(1,3) Algebra and Minkowski Spacetime  
    4.2. Mathematical Formulation of MonSTER  
     4.2.1. Relative Spacetime Displacement  
     4.2.2. Block-Wise Lorentz Transformations  
     4.2.3. Preserving the Minkowski Dot Product  
    4.3. Advantages Over Existing Positional Encoding Schemes  
    4.4. Implications for Native Spacetime Intelligence

5. **The Autoregressive Challenge in 4D: Rethinking Token Generation**  
    5.1. Sequential Generation in Current Models  
    5.2. Difficulties in Adapting Autoregression to 4D  
     5.2.1. Lattice-Based Generation  
     5.2.2. Position-Content Alternation  
     5.2.3. Patched or Grouped Outputs  
    5.3. The Need for a New Output Paradigm

6. **Current Objectives: A 4D Transformer Pipeline and Unified Data Model**  
    6.1. Dual Thrust: Dataset Creation and Model Architecture  
    6.2. Dataset: 4D-Addressed Code-Image Pairs and Holarchic Tokens  
     6.2.1. The Holarchic 4D Token: \[Parent, Name, Type, Value, Position (t, x, y, z)\]  
     6.2.2. Representing Code with 4D Tokens  
     6.2.3. Representing Images and ARC Tasks with 4D Tokens  
    6.3. Proposed Model Architecture: Leveraging MonSTER and Latent Attention  
     6.3.1. Inspiration from DeepSeek: Multi-Head Latent Attention (MLA)  
     6.3.2. Structured Input Token Processing: \[Name, Type, Value\]  
     6.3.3. Latent Space Projection and MonSTER Integration  
     6.3.4. Concatenated Latent Representation  
     6.3.5. Simultaneous, Structured Output Generation

7. **Summer Goals, Future Evaluation, and Vision**  
    7.1. Immediate Focus: Dataset and Architecture Implementation  
    7.2. Evaluation on ARC-AGI  
    7.3. Long-Term Vision: Advancing General AI

# **1\. Initial Objective: A Transformer Pipeline for ARC-AGI via Code-Image Compression**

My research was initiated with a highly ambitious objective: to develop a transformer-based architecture and associated pipeline specifically designed to address the ARC-AGI (Abstraction and Reasoning Corpus) challenge. The ARC-AGI benchmark, introduced by François Chollet, is engineered to evaluate the general reasoning and fluid intelligence of AI systems, a domain where current models, despite their other successes, demonstrate significant shortcomings.

## **1.1. The ARC-AGI Challenge: A Test of Fluid Intelligence**

The ARC-AGI challenge comprises a collection of novel visual puzzles. Each puzzle is presented as one or more pairs of input-output grids composed of colored cells. The core task for the AI is to infer the underlying transformation rule from these demonstrated pairs and then apply this rule to a new input grid to produce the correct output grid. What makes ARC particularly challenging is its design to be “resistant to memorization.” Each puzzle is entirely novel to the solver, meaning success hinges on on-the-fly reasoning and skill acquisition efficiency rather than recalling patterns from vast training data. This contrasts sharply with many benchmarks where LLMs excel, often through sophisticated pattern matching learned from extensive corpora.

ARC tasks are analogous to human IQ tests, particularly Raven’s Progressive Matrices, requiring the recognition of abstract patterns and logical transformations without textual instruction. Despite their apparent simplicity for humans (even children can solve many ARC tasks), state-of-the-art LLMs like GPT-4 have shown near-zero accuracy. This “Achilles’ heel” highlights a fundamental gap in current AI’s ability to perform human-like fluid reasoning and abstraction.

## **1.2. The p5.js Hypothesis: Bridging Code and Visual Abstraction**

My initial hypothesis posited that a key to unlocking ARC-AGI might lie in bridging procedural code generation with visual understanding. p5.js, a JavaScript library for creative coding, provides an explicit framework for generating visuals through code. We reasoned that if a model could learn to map between complex p5.js code (which describes how to draw an image) and the resultant image (the “what” that is drawn), it might internalize concepts of procedural abstraction, spatial relationships, and object composition in a way that is directly relevant to ARC’s demands. The ARC tasks often involve programmatic-like transformations (e.g., “find all objects of type X and move them Y units”). Learning from p5.js, where such transformations are explicit in the code, seemed like a promising avenue.

## **1.3. Proposed Encoder-Decoder Architecture**

To test this hypothesis, I planned an encoder-decoder transformer architecture:

* **Encoder**: The encoder would take as input a p5.js source code snippet. Its task would be to learn a compressed representation, or latent code, that captures the essential visual and procedural information embedded in the script.

* **Decoder**: The decoder would take this latent code as input and attempt to reconstruct the corresponding 512×512 pixel RGB image that the p5.js code generates.

The core idea was that the “bottleneck” of the latent space would force the model to learn highly efficient and abstract representations. If the model could successfully reconstruct complex images from a compact token representation derived from code, it would imply a significant level of understanding of how code translates to visual structure.

## **1.4. Latent Space and Tokenization Goals**

A critical design parameter was the size and nature of the latent space. I aimed for a highly compressed representation, targeting a latent code of **≤256 tokens** to represent a 512×512 RGB image. An RGB image of this size has 512 × 512 × 3 \= 786,432 individual color values. Compressing this information into approximately 256 tokens represents a compression ratio of over 3000:1. This aggressive compression was deemed necessary to force the model to learn high-level abstractions rather than memorizing pixel patterns. Each “token” in this latent space would ideally represent a meaningful visual primitive or a procedural element derived from the p5.js code.

## **1.5. Optimization Strategy: Reconstruction Loss**

The primary training signal for this encoder-decoder model would be reconstruction loss. This measures the difference between the image generated by the decoder (from the latent code) and the ground-truth image produced by executing the original p5.js script. I planned to use a combination of loss functions:

* **Mean Squared Error (MSE)**: A common pixel-wise loss that measures the average squared difference between predicted and actual pixel values. While straightforward, MSE can sometimes lead to blurry reconstructions as it tends to average possibilities.

* **Structural Similarity Index Measure (SSIM)**: SSIM is designed to be more perceptually aligned with human vision, considering changes in structural information, luminance, and contrast. I anticipated that SSIM would encourage the model to preserve the coherent structures within the images, which is crucial for ARC-like tasks.

The model would be trained via standard backpropagation, iteratively adjusting its weights to minimize this combined reconstruction loss. The expectation was that minimizing this loss would drive the model to create latent codes that are not only compact but also rich in the semantic and structural information necessary to reconstruct the target images accurately.

# **2\. The Flattening Problem: Realizing the Limitations of Sequential Spatialization**

As research progressed on the transformer architecture for p5.js code-to-image tasks, an important conceptual hurdle emerged: the fundamental way transformers process information.

## **2.1. Standard Transformer Approaches to Multidimensional Data**

Modern transformer architectures—such as Vision Transformers (ViT), Perceiver-IO, and models handling 3D data (e.g., NeRF)—typically rely on a “flatten-and-index” strategy for spatial inputs:

* **Image Flattening**  
   ViT divides an image into a grid of patches (e.g., 16×16 pixels), flattens each patch into a vector, and treats the resulting sequence of patch embeddings as input, much like a sequence of word embeddings in NLP.

* **Positional Encoding**  
   To preserve some notion of spatial locality, positional encodings are added to each patch embedding. These may take the form of learned absolute embeddings (one per patch index) or 2D-aware encodings (for example, fixed sinusoidal grids or learned axial codes).

* **Sequential Processing**  
   Despite these measures, the core self-attention mechanism still processes tokens as a one-dimensional list. The continuous geometry of 2D or 3D space becomes discretized and linearized. Even with a 2D positional encoding, attention operates over tokens tagged with 2D labels rather than operating directly in 2D.

In practice, many attempts to “spatialize” attention still flatten an image, video, or point cloud into a flat token list, appending per-axis positional codes. Treating spatial dimensions (x, y, z) and temporal dimension (t) as independent scalars means that any cross-dimensional geometry, metric invariances, or causal couplings (as in spacetime) are not inherently encoded. Instead, the model must re-learn them from scratch via attention weights. Space and time thus remain metadata attached to a sequential list, rather than being embedded as a single, coupled manifold.

## **2.2. Loss of Structural Information and Its Implications for ARC-AGI**

This flattening paradigm, while practical for reusing existing transformer mechanics, leads to the loss or obfuscation of critical multidimensional structure:

* **Adjacency and Connectivity**  
  In a 2D grid, pixel adjacency defines local texture and shape. Once flattened, pixels that were adjacent in 2D may become far apart in the 1D sequence—especially across row boundaries. For example, in a simple 3×3 tic-tac-toe board, spatial cues like “corner-to-corner” or “edge-to-center” relationships vanish when the grid is vectorized into a 1×9 sequence. This intuitive spatial structure is lost, forcing the model to relearn it.

* **Geometric Invariances**  
  Natural visual operations—rotation, scaling, translation—have complex effects on flattened sequences that do not reflect their simple geometric nature. Consequently, the model must implicitly discover these invariances during training, rather than having them built into the representation.

* **Higher-Order Relationships**  
  Concepts such as containment, symmetry, and relative configurations of multiple objects become difficult to discern once the native spatial coordinate system is dismantled.

For ARC-AGI specifically, these losses are critically detrimental. ARC tasks often require recognizing transformations based on precise spatial configurations, symmetries, and object-level manipulations. A model that “sees” an ARC grid as a list of colored pixels with one-dimensional position tags is at a severe disadvantage compared to a system—like the human visual system—that interprets the grid as a coherent 2D canvas with objects, shapes, relative positions, and orientations. Thus, the “flatten-and-index” paradigm enforces an artificial one-dimensional timeline that collapses richer spatial and relational structures, effectively erasing information essential for solving ARC-AGI’s visuospatial reasoning problems.

## **2.3. The “Jagged Edge” of LLM Capabilities**

The difficulty with spatial reasoning, despite proficiency in language and code generation, exemplifies the “Jagged Intelligence” phenomenon described by Andrej Karpathy: state-of-the-art LLMs can perform remarkably well on certain tasks (for example, fluent text or code generation), yet struggle with seemingly simpler tasks—especially visuospatial puzzles. This “jagged frontier” echoes Hans Moravec’s 1980s observation that it is comparatively easy to make computers excel at intelligence-test–style problems but extremely difficult to grant them the perceptual and motor skills of a human toddler.

Recognizing the flattening problem thus highlighted a critical jagged edge: current transformer architectures, when applied naively to visuospatial tasks, are fundamentally limited by their 1D processing nature. The path forward demands embeddings that do more than assign unique identifiers to temporal or linearized indices; instead, they must inherently encode the intrinsic dependencies of higher-dimensional spaces and the coupling of space and time.

# **3\. A New Trajectory: Generalizing RoPE for True 4D Positional Understanding**

The realization that current spatialization techniques were insufficient led to a fundamental pivot in my research strategy. Instead of trying to force multidimensional data into a 1D sequential framework, I sought a way to empower transformers to understand and operate natively within higher-dimensional spaces. This led us to reconsider the role and nature of positional encodings.

---

## **3.1. The Need for Robust Positional Encodings**

As established, the self-attention mechanism in transformers is permutation-invariant; it has no inherent sense of sequence order or spatial position. Positional encodings are therefore essential to inject this crucial information. The choice of positional encoding scheme significantly impacts a model's ability to learn dependencies and generalize.

* **Absolute Positional Encodings (APE):**  
   Early transformers used either:

  1. **Fixed sinusoidal APEs**  
      These map each absolute position to a unique vector using sine and cosine functions of different frequencies. Sinusoidal encodings offered some hope for extrapolation to longer sequences because of their periodic nature and the fact that relative positions correspond to linear transformations in the encoding space. However, in practice, their generalization to significantly longer sequences often remains limited.

  2. **Learned APEs**  
      A distinct vector is learned for each position up to a maximum length. Although potentially more adaptable to specific training data, learned APEs struggle even more with extrapolation beyond their training length and lack a strong inductive bias for relative positioning.

* **Relative Positional Encodings (RoPE):**  
   RoPEs aim to directly encode the difference or distance between token positions. For example, approaches like Shaw et al. (2018) add a learned bias or vector based on the relative offset (j−i)(j \- i) directly into the attention computation. This makes the model inherently translation-invariant (shifting the entire sequence doesn’t change relative distances) and aids generalization to longer sequences. The Music Transformer, for instance, used a form of relative self-attention to capture musical motifs over long durations. However, naive RoPEs can introduce computational overhead or require managing embeddings for many possible relative distances.

The ideal positional encoding would:

1. Seamlessly integrate relative position information  
2. Support extrapolation to varying input sizes  
3. Operate efficiently without adding substantial parameters or computational complexity

---

## **3.2. Rotary Positional Embeddings (RoPE) as a Starting Point**

Rotary Positional Embeddings (RoPE), introduced by Su et al. (2021), emerged as a particularly promising candidate that met many of these criteria for 1D sequences. RoPE offers a unique way to inject positional information by rotating the query and key vectors in the self-attention mechanism by an angle dependent on their absolute position.

* **Mechanism:**  
   Instead of adding a positional vector, RoPE multiplies the query q\\mathbf{q} and key k\\mathbf{k} vectors by position-dependent rotation matrices R(m)R(m) and R(n)R(n) respectively (for positions mm and nn). The rotation is typically applied in 2D subspaces (pairs of coordinates) of the embedding, with rotation angles proportional to the position index and varying across frequencies (analogous to sinusoidal encoding frequencies).

* **Relative Dependence:**  
   The crucial insight is that the dot product in the attention score,  
   ⟨qmR(m),knR(n)⟩, \\langle \\mathbf{q}\_m R(m), \\mathbf{k}\_n R(n)\\rangle,  
   becomes a function of the relative position (m−n)(m \- n) due to the properties of rotation matrices (R(m) R(n)T=R(m−n))\\bigl(R(m)\\,R(n)^{T} \= R(m \- n)\\bigr). This means RoPE effectively encodes relative positions implicitly within the attention mechanism itself.

* **Advantages:**

  1. Flexibility in sequence length and good extrapolation capabilities  
  2. A natural decay in inter-token dependency with increasing relative distance (a useful locality bias)  
  3. No additional learnable parameters

Its widespread adoption in models like LLaMA, PaLM, and GPT-J attests to its efficacy. Given RoPE’s success and elegant mathematical formulation for 1D sequences (often visualized as rotations in the complex plane for each 2D block), the natural question arose: **Could this be generalized to handle the three spatial dimensions and one temporal dimension required for my goals?**

---

## **3.3. Limitations of Real Vector Representations for Geometric Data**

My exploration into generalizing RoPE quickly led to a broader consideration of how neural networks represent geometric information. Traditional real-valued embeddings (dense vectors in Rn\\mathbb{R}^n) face inherent limitations when tasked with capturing complex geometric structures or transformations:

1. **Rotations and Symmetries:**  
   Representing a 3D rotation with standard real-valued neural network layers typically requires learning a 3×33 \\times 3 matrix with nine parameters. Ensuring orthogonality (to preserve lengths and angles) is not guaranteed without specific constraints or architectural choices. Simple vector addition or element-wise multiplication—common in neural networks—do not naturally correspond to rotations.

2. **Hierarchies:**  
   Euclidean space is not well-suited for embedding tree-like hierarchies efficiently. The number of nodes in a balanced tree grows exponentially with depth, but Euclidean distances grow only polynomially with radius. This mismatch leads to “crowding” and distortion when embedding large hierarchies in low-dimensional Euclidean spaces.

3. **Internal Dependencies:**  
   For multidimensional input features where dimensions are intrinsically correlated (e.g., the R, G, B channels of a pixel, or the x,y,zx, y, z components of a 3D point), treating these as independent scalar inputs in a real-valued network means the model must learn these correlations from scratch. This can be inefficient and may fail to preserve essential relationships.

4. **Parameter Inefficiency:**  
   To approximate complex geometric transformations or capture coupled features, real-valued networks might require significantly more parameters than networks whose algebraic structure inherently supports these operations.

---

## **3.4. Exploring Higher Division Algebras: Quaternions and Clifford Algebra**

These limitations motivated my investigation into hypercomplex number systems and geometric algebras, which offer more structured ways to represent and manipulate multidimensional data:

* **Quaternions:**  
  Quaternions (H\\mathbb{H}) extend complex numbers to four dimensions (one real, three imaginary units i,j,ki, j, k) and their multiplication (the Hamilton product) naturally encodes 3D rotations. Quaternion Neural Networks (QNNs) leverage this by using quaternion-valued weights and activations. This allows them to treat 4D feature vectors as single entities, inherently learn rotational transformations, and often achieve better performance with fewer parameters by enforcing weight sharing through the Hamilton product. QNNs have shown success in modeling 3D spatial relations and internal dependencies in multidimensional signals—such as color pixels or speech features. For instance, quaternion RNNs/LSTMs can capture inter-dimensional correlations in speech signals, leading to significant parameter reduction (e.g., 3.3× fewer parameters than real LSTMs for improved speech recognition accuracy).

* **Clifford Algebras (Geometric Algebras):**  
  Clifford algebras provide a powerful and unified framework for geometry, generalizing complex numbers, quaternions, and vector algebra. A Clifford algebra is defined over a vector space equipped with a quadratic form (like a dot product). For our goal of representing 4D spacetime, the Clifford algebra Cl(1,3)\\mathrm{Cl}(1,3), associated with the Minkowski metric (+,−,−,−)(+,-,-,-), seemed particularly relevant. This algebra naturally incorporates vectors, bivectors (representing oriented planes, rotations, or Lorentz boosts), and higher-grade elements, allowing for the representation of Lorentz transformations as algebraic operations (rotors).

The key insight was that using an algebraic structure that inherently matches the geometry of the problem domain (such as 3D rotations for quaternions, or Lorentz transformations for Cl(1,3)\\mathrm{Cl}(1,3) in spacetime) could provide a much stronger inductive bias and more efficient representations than attempting to learn these structures from scratch with generic real-valued vectors. This theoretical underpinning set the stage for the development of **MonSTER**.

# **4\. Breakthrough: MonSTER – Minkowski SpaceTime Embedding Rotors**

Armed with the insights from RoPE’s success, the limitations of standard spatialization, and the potential of geometric algebras, MonSTER (Minkowski SpaceTime Embedding Rotors) was developed. MonSTER is conceived as a principled generalization of RoPE from 2D planar rotations (effectively, complex number multiplications for each block) to full 4D Lorentz transformations operating within the Clifford algebra Cl(1,3). Its primary purpose is to allow transformer models to natively process and reason about data embedded in a 4D Minkowski spacetime.

---

## **4.1. Conceptual Foundation: Clifford(1,3) Algebra and Minkowski Spacetime**

The choice of Cl(1,3) algebra with a (+,−,−,−) metric signature is deliberate. This is the mathematical framework of Special Relativity, describing spacetime where the interval

ds2=c2dt2−dx2−dy2−dz2ds^2 \= c^2 dt^2 \- dx^2 \- dy^2 \- dz^2

is invariant under Lorentz transformations (rotations in space and boosts, which mix space and time). By encoding positional information using transformations that respect this Minkowski metric, MonSTER aims to provide the transformer with a notion of:

* **Spacetime Intervals**: The “distance” or relationship between two events (tokens at different spacetime positions) is measured by the Lorentz-invariant interval, not just Euclidean distance or sequential offset.

* **Causal Structure**: The sign of the spacetime interval (ds2\>0ds^2 \> 0 for timelike, ds2\<0ds^2 \< 0 for spacelike, ds2=0ds^2 \= 0 for lightlike) implicitly encodes causal relationships. While MonSTER does not explicitly enforce causality, operating in Minkowski space provides the geometric substrate for the model to learn such relationships.

* **Lorentz Invariance**: Ideally, the attention mechanism’s perception of the relationship between two tokens should be independent of the inertial reference frame, a property guaranteed by Lorentz transformations.

MonSTER moves beyond simply tagging tokens with (t,x,y,z)(t, x, y, z) coordinates to embedding them so that attention computations inherently respect spacetime geometry.

---

## **4.2. Mathematical Formulation of MonSTER**

MonSTER calculates a unique 4D Lorentz transformation, ReffR\_{\\mathrm{eff}}, based directly on the relative spacetime displacement ΔP=(Δt,Δx,Δy,Δz)\\Delta P \= (\\Delta t, \\Delta x, \\Delta y, \\Delta z) between a query token at PqP\_q and a key token at PkP\_k. ReffR\_{\\mathrm{eff}} is generated block-wise: the embedding dimension is divided into blocks (typically 4D each), and a distinct (but frequency-scaled) Lorentz transformation is applied to each block. This multi-frequency approach—analogous to RoPE—allows for a richer, multi-scale representation of relative positions.

### **4.2.1. Relative Spacetime Displacement**

First, the raw displacement is computed:

ΔPraw=Pk−Pq=(Δtraw,Δxraw,Δyraw,Δzraw).\\Delta P\_{\\text{raw}} \= P\_k \- P\_q \= (\\Delta t\_{\\text{raw}}, \\Delta x\_{\\text{raw}}, \\Delta y\_{\\text{raw}}, \\Delta z\_{\\text{raw}}).

For applications like ARC-AGI, these raw deltas might correspond to pixel differences in grid coordinates and step differences in time (e.g., input vs. output frame).

### **4.2.2. Block-wise Lorentz Transformations**

For each block b∈{1,…,B}b \\in \\{1, \\ldots, B\\}:

1. **Scaled Displacements**  
    Δtscaled,b=Δtraw  ×  inv\_freq\_timebΔsscaled,b=(Δxraw×inv\_freq\_spaceb,  Δyraw×inv\_freq\_spaceb,  Δzraw×inv\_freq\_spaceb)\\Delta t\_{\\text{scaled}, b} \= \\Delta t\_{\\text{raw}} \\;\\times\\; \\text{inv\\\_freq\\\_time}\_b \\\\\[6pt\] \\Delta s\_{\\text{scaled}, b} \= \\bigl(\\Delta x\_{\\text{raw}}\\times \\text{inv\\\_freq\\\_space}\_b,\\;\\Delta y\_{\\text{raw}}\\times \\text{inv\\\_freq\\\_space}\_b,\\;\\Delta z\_{\\text{raw}}\\times \\text{inv\\\_freq\\\_space}\_b\\bigr)  
    where inv\_freq\_timeb\\text{inv\\\_freq\\\_time}\_b and inv\_freq\_spaceb\\text{inv\\\_freq\\\_space}\_b are derived from base values (e.g., 10000\) and the block index bb, ensuring different “wavelengths” for each block.

2. **Spatial Rotation Angle and Axis**  
    θb=∥Δsscaled,b∥2,u^rot,b={Δsscaled,bθb,θb≠0,default axis (e.g., (0,0,1)),θb=0.\\theta\_b \= \\|\\Delta s\_{\\text{scaled}, b}\\|\_2, \\qquad \\hat{u}^{\\mathrm{rot}, b} \= \\begin{cases} \\displaystyle \\frac{\\Delta s\_{\\text{scaled}, b}}{\\theta\_b}, & \\theta\_b \\neq 0, \\\\\[8pt\] \\text{default axis (e.g., }(0,0,1)\\text{)}, & \\theta\_b \= 0\. \\end{cases}  
    A 3×3 spatial rotation matrix R3,bR\_{3,b} is constructed using Rodrigues’ formula from u^rot,b\\hat{u}^{\\mathrm{rot}, b} and θb\\theta\_b. Embedding into 4×4 yields Mrot,bM\_{\\text{rot}, b}, which acts as identity on the time component and R3,bR\_{3,b} on spatial components.

3. **Boost Rapidity and Axis**  
    The scaled temporal displacement Δtscaled,b\\Delta t\_{\\text{scaled}, b} determines the boost rapidity φb\\varphi\_b. In the provided code, φb=Δtscaled,b\\varphi\_b \= \\Delta t\_{\\text{scaled}, b} directly, though a refinement φb=Cttanh⁡(Δtscaled,b/Ct)\\varphi\_b \= C\_t \\tanh(\\Delta t\_{\\text{scaled}, b}/C\_t) may be used for numerical stability at large deltas. The boost axis u^boost,b\\hat{u}^{\\mathrm{boost}, b} is typically chosen to match u^rot,b\\hat{u}^{\\mathrm{rot}, b}. A 4×4 Lorentz boost matrix Mboost,bM\_{\\text{boost}, b} is constructed using cosh⁡(φb)\\cosh(\\varphi\_b) and sinh⁡(φb)\\sinh(\\varphi\_b), preserving orthonormality within the Minkowski metric.

4. **Combine Transformations**  
    Reff,b  =  Mboost,b Mrot,b.R\_{\\text{eff}, b} \\;=\\; M\_{\\text{boost}, b}\\,M\_{\\text{rot}, b}.  
    Operationally, spatial rotation is applied first, then the boost. Since rotation and boost share the same axis, these operations commute: Mrot,bMboost,b=Mboost,bMrot,bM\_{\\text{rot}, b}M\_{\\text{boost}, b} \= M\_{\\text{boost}, b}M\_{\\text{rot}, b}.

Repeating this for all BB blocks yields a stack of 4×4 matrices that encode the relative spacetime transformation across multiple frequency scales.

### **4.2.3. Preserving the Minkowski Dot Product**

A crucial aspect of MonSTER is ensuring each Reff,bR\_{\\text{eff}, b} is a true Lorentz transformation that preserves the Minkowski metric

η=diag(1,−1,−1,−1).\\eta \= \\mathrm{diag}(1, \-1, \-1, \-1).

That is,

Reff,bT η Reff,b  =  η.R\_{\\text{eff}, b}^\\mathsf{T} \\,\\eta\\, R\_{\\text{eff}, b} \\;=\\; \\eta.

This guarantee means that when these transformations modulate query and key vectors within self-attention—via a Minkowski dot product such as

qbT η (Reff,b kb)or(R(Pq) qb)T η (R(Pk) kb),\\mathbf{q}\_b^\\mathsf{T}\\,\\eta\\,\\bigl(R\_{\\text{eff}, b}\\,\\mathbf{k}\_b\\bigr)\\quad\\text{or}\\quad \\bigl(R(P\_q)\\,\\mathbf{q}\_b\\bigr)^\\mathsf{T}\\,\\eta\\,\\bigl(R(P\_k)\\,\\mathbf{k}\_b\\bigr),

the resulting attention scores depend on Lorentz-invariant spacetime intervals rather than just Euclidean or linearized offsets. Constructing Mrot,bM\_{\\text{rot}, b} via Rodrigues’ formula and Mboost,bM\_{\\text{boost}, b} via standard Lorentz boost formulas inherently yields matrices that satisfy this preservation (within numerical precision). The underlying code, as presented, assumes these constructions are correct; if needed, explicit orthonormalization can correct small numerical errors to enforce Reff,bTηReff,b=ηR\_{\\text{eff}, b}^\\mathsf{T}\\eta R\_{\\text{eff}, b} \= \\eta.

---

## **4.3. Advantages over Existing Positional Encoding Schemes**

MonSTER offers several key advantages for processing spatiotemporal data:

* **Native 4D Representation**  
  Unlike schemes that flatten data or treat dimensions independently, MonSTER embeds positions directly within a 4D spacetime framework.

* **Geometric Inductive Bias**  
  By constructing embeddings via Lorentz transformations, MonSTER provides a strong inductive bias for spacetime geometry, including spatial distance, temporal duration, and their coupling through the Minkowski metric.

* **Relativity Principle**  
  Respecting Lorentz invariance lays the groundwork for transformer models whose understanding of events is independent of any specific observer’s reference frame—a highly desirable property for general intelligence.

* **Generalization of RoPE**  
  MonSTER retains RoPE’s attractive properties—parameter-free encoding, multi-scale frequency blocks, and implicit relative position encoding—while extending applicability from 1D sequences to full 4D spacetime.

---

## **4.4. Implications for Native Spacetime Intelligence**

The successful development of MonSTER provides a pathway to imbue transformer architectures with “native spacetime intelligence.” Instead of learning spatial and temporal relationships as arbitrary patterns from 1D sequences, a MonSTER-equipped transformer computes attention based on geometric relationships within a 4D manifold. This approach promises significantly improved generalization and sample efficiency on tasks involving complex spatiotemporal reasoning—such as ARC-AGI, video understanding, physics simulations, or multi-agent robotics—by preserving and leveraging the true geometry of the problem domain.

# **5\. The Autoregressive Challenge in 4D: Rethinking Token Generation**

While MonSTER provided a robust solution for representing 4D positional information on the input side of a transformer, it simultaneously highlighted a new challenge: how to structure the output tokens in a way that respects this 4D framework.

---

## **5.1. Sequential Generation in Current Models**

The dominant paradigm for generation in transformers (especially LLMs) is autoregressive. Tokens are produced one at a time, with each new token conditioned on the sequence of previously generated tokens. In this model, a token’s position is implicitly defined by its order in the generated sequence. This works well for 1D data like text or linearized representations of other modalities.

---

## **5.2. Difficulties in Adapting Autoregression to 4D**

Adapting this strictly sequential, 1D autoregressive approach to a truly 4D output space (as implied by MonSTER’s capabilities) presents several conceptual and practical difficulties:

### **5.2.1. Lattice-Based Generation**

* **Concept**: Generate tokens for a 3D spatial lattice at discrete timesteps (e.g., VQ-VAE might learn a 3D grid of discrete latent codes for a video volume).

* **Limitations**:

  * Assumes a fixed-size grid, which becomes very inefficient if content is sparse (many spacetime positions would be “empty” and require generating blank or placeholder tokens).  
  * Defining a single canonical raster-scan order for 3D or 4D that preserves locality is non-trivial and can break natural symmetries of the space.

### **5.2.2. Position-Content Alternation**

* **Concept**: The model alternates between generating:

  * A 4D position token (t,x,y,z)(t, x, y, z) specifying where to output next, and  
  * A content token specifying what to output at that location.

* **Advantages**: More flexible regarding sparsity, since the model only emits content for occupied positions (similar to stroke-based image generation models like SketchRNN or event-based music generation using time-shift events).

* **Challenges**:

  * Doubles the length of the output sequence.  
  * Introduces a complex “action space” for position tokens—forcing the model to learn both content and an optimal traversal strategy through 4D space.  
  * Conceptually akin to the “movement–action” cycle explored in cortical-column-inspired transformer research.

### **5.2.3. Patched or Grouped Outputs**

* **Concept**: Output patches or groups of tokens simultaneously, in a structured order—for example, generating an image row by row or a video frame by frame, with parallel generation within each row/frame.

* **Examples**:

  * MaskGIT generates images by iteratively filling tokens in a discrete grid, starting with a coarse overview and refining details.

* **Limitations**:

  * Still operates on a predefined grid of token slots.  
  * Extending to an arbitrary 4D volume with variable numbers of “active” tokens—without imposing a fixed grid or complex masking/scheduling strategy—is challenging.

---

Each of these approaches presents trade-offs regarding computational efficiency, representational fidelity, and learning complexity. For instance, forcing a 4D structure into a 1D sequence for autoregression can obscure the very spatiotemporal relationships MonSTER is designed to capture. Simply generating all positions in a dense 4D grid is often intractable at high resolutions.

---

## **5.3. The Need for a New Output Paradigm**

This difficulty highlighted that a truly 4D-aware transformer needs not just 4D input embeddings but also a generation mechanism that can naturally produce outputs structured in 4D. The existing autoregressive paradigm, while powerful for sequences, does not straightforwardly extend to generating sparse, variably structured data in a high-dimensional continuous or discrete space without imposing linearization or fixed gridding—approaches that may be suboptimal.

This realization pushed me to design output structures that are inherently hierarchical and spatially/temporally addressed. These insights directly inform the **Holarchic 4D Spacetime Knowledge Model** and the proposed architecture described in the next section.

# **6\. Current Objectives: A 4D Transformer Pipeline and Unified Data Model**

Given the development of MonSTER and the recognized challenges in 4D output generation, my research has converged on two primary, intertwined objectives. These are aimed at creating a complete pipeline that can process and generate hierarchically structured, 4D-addressed data, with the ultimate goal of testing this system on ARC-AGI and other tasks requiring deep spatiotemporal reasoning.

---

## **6.1. Dual Objectives: Dataset Creation and Model Architecture**

The two main objectives of the current work are:

1. **Dataset Creation**  
   Developing a novel dataset composed of (source code, image) pairs and potentially (source code, image set) pairs. [Examples here.](https://training-theta-five.vercel.app/) The defining feature of this dataset will be the explicit assignment of 4D (t, x, y, z) positions to both the constituent elements of the source code (e.g., functions, statements, variables) and the pixels or meaningful regions within the corresponding images.

2. **Model Architecture Development**  
   Designing and implementing a transformer-based model architecture capable of effectively learning from this 4D-addressed dataset and leveraging MonSTER for its positional understanding. This architecture must also be capable of generating structured outputs that respect these 4D relationships.

---

## **6.2. Dataset: 4D-Addressed Code-Image Pairs and Holarchic Tokens**

A core hypothesis is that intelligence arises from understanding compositional, hierarchical structures situated in a consistent reference frame. To this end, I am designing the dataset around a Holarchic 4D Token Model.

### **6.2.1. The Holarchic 4D Token:**

**\[Parent, Name, Type, Value, Position (t, x, y, z)\]**

This token structure is inspired by Arthur Koestler’s concept of “holons” (entities that are simultaneously wholes and parts) and fact-based systems like Entity-Attribute-Value-Time (EAVT) used in databases such as Datomic. Each piece of information, regardless of domain, is represented as a 5-tuple token:

* **Parent**  
  A reference to the immediate container token, establishing a nested hierarchy (e.g., a function token is the parent of its statement tokens). The root of a hierarchy has `Parent = None`.

* **Name**  
  An optional human-readable identifier or role label for the token (e.g., function name “foo”, variable name “x”, or a label like “InputGrid1”).

* **Type**  
  The semantic or syntactic category of the token (e.g., “FunctionDef”, “Pixel”, “ImageGrid”, “Operator”, “Expression”). This aids in interpreting the token’s value and its expected children.

* **Value**  
  The literal data content of the token (e.g., a numeric value, a color code, a code snippet, or a reference to another entity if it’s a relational token).

* **Position (t, x, y, z)**  
  A 4D coordinate anchoring the token in a specific spacetime context:  
  * **t (time):** Temporal coordinate or timestamp (e.g., code version, execution step, input vs. output state in ARC).  
  * **x, y, z (space):** Spatial coordinates whose interpretation is domain-specific (e.g., line/column in code, pixel row/column in an image, layer index).

This structure allows for representing not just atomic data but also its compositional structure, its location in a relevant frame of reference, and its state over time. It aims to provide a universal schema for knowledge across domains.

---

### **6.2.2. Representing Code with 4D Tokens**

For program source code, the hierarchy flows from **Project/Codebase** down to **Files**, then to **Functions/Classes**, **Statements**, and finally **Expressions/Lexical Tokens**.

* **Spatial Coordinates (x, y, z):**

  * **y:** Line number  
  * **x:** Column number within the file  
  * **z:** File index or module/directory depth  
     Higher-level constructs like functions use the coordinates of their starting definition.

* **Temporal Coordinate (t):**

  * Represents code versions (e.g., from commits) or stages in a static analysis or transformation process. For a static snapshot, `t` might be uniform.

This results in a representation akin to an Abstract Syntax Tree (AST) where each node is augmented with precise source location and versioning information.

*(See example table in \[reference\].)*

---

### **6.2.3. Representing Images and ARC Tasks with 4D Tokens**

For ARC-AGI tasks, which involve transformations of colored pixel grids:

* **Hierarchy:**  
   `Task → IO_Pair (Input/Output) → ImageGrid → Pixel.`

* **Pixel Token:**  
  * **Parent:** The ImageGrid to which it belongs  
  * **Name:** Optional  
  * **Type:** “Pixel”  
  * **Value:** Integer color code

* **Spatial Coordinates (x, y, z):**  
  * **x:** Column index within the ImageGrid  
  * **y:** Row index within the ImageGrid  
  * **z:** Distinguishes different images within a task (e.g., `z = 0` for Input1, `z = 1` for Output1, `z = 2` for Input2, etc.).

* **Temporal Coordinate (t):**  
  * Crucial for representing the transformation:  
    * Input image pixels are assigned `t = 0`  
    * Corresponding output image pixels are assigned `t = 1`  
       This explicitly encodes the before-and-after states.

This tokenization allows a solver to reason about pixel changes by comparing tokens with the same `(x, y, z)` but different `t` values, or to track the movement of “objects” (collections of pixel tokens) through changes in their `(x, y)` coordinates over `t`.

*(See example table in \[reference\].)*

---

## **6.3. Proposed Model Architecture: Leveraging MonSTER and Latent Attention**

To process this 4D-addressed, hierarchically tokenized data, I propose a transformer architecture incorporating MonSTER and inspired by recent advancements in efficient attention mechanisms.

### **6.3.1. Inspiration from DeepSeek: Multi-Head Latent Attention (MLA)**

DeepSeek models introduced **Multi-Head Latent Attention (MLA)**, a technique that improves efficiency by compressing keys and values into a smaller latent space before attention computation. MLA achieves this by factorizing key and value projections through a low-dimensional bottleneck, reducing the size of the KV cache significantly—especially beneficial for long sequences or large models. This allows for faster inference and longer context handling, often with little to no performance degradation, and sometimes even improves performance due to the regularizing effect of the low-rank constraint. I aim to adopt a similar “small-medium sized Multi-Head Latent Attention architecture” for efficiency.

---

### **6.3.2. Structured Input Token Processing: \[Name, Type, Value\]**

Our input tokens are `[Parent, Name, Type, Value, Position (t, x, y, z)]`. We will first process the content-bearing part: `[Name, Type, Value]`. Each of these sub-tokens (or a combined representation) will be embedded into a high-dimensional vector. This embedding serves as the initial representation for the “content” of the token. This high-dimensional embedding can be thought of as an “ID” that uniquely represents the essence of that specific `[Name, Type, Value]` combination.

---

### **6.3.3. Latent Space Projection and MonSTER Integration**

Following the MLA paradigm, this high-dimensional content-ID embedding will be down-projected into a lower-dimensional latent space to generate the queries (Q), keys (K), and values (V) for the attention mechanism.

Separately, the `Position (t, x, y, z)` component of each token will be processed by MonSTER. MonSTER takes the 4D coordinates and outputs the block-wise 4×4 Lorentz transformation matrices as described in Section 4\. These transformations are then applied to the queries and keys (or incorporated into their dot-product calculation within the attention mechanism) to make the attention scores sensitive to relative 4D spacetime positions. This effectively detaches the positional calculation from the content embedding until they are combined in the attention mechanism as outlined in the decoupled RoPE implementation in DeepSeek V2 and V3.

---

### **6.3.4. Concatenated Latent Representation (Conceptual)**

While the actual mechanics involve MonSTER modulating the Q–K interaction, conceptually, the information available to each attention head for a given token includes:

1. **A Parent ID Embedding**  
   * Possibly a summary embedding of its Parent token.  
2. **The Latent Q, K, V Derived from Its \[Name, Type, Value\] Content**  
3. **Its `Position (t, x, y, z)` as Processed by MonSTER**

Implied is a final latent representation concatenating these aspects:

\[Parent\_ID\_embedding, Name\_Type\_Value\_embedding (source of Q/K/V), Position\_ID\_embedding (post-MonSTER application)\]

More precisely, MonSTER-transformed positional information directly influences the Q⋅KTQ \\cdot K^T dot products within the attention heads. The `Parent_ID` might be incorporated either:

* As an additional feature to the content embedding before projection, or  
* Used in a separate attention mechanism to model hierarchical context.

---

### **6.3.5. Simultaneous, Structured Output Generation**

A key departure from standard autoregressive models is the envisioned output mechanism. After several layers of latent self-attention, the model is intended to **simultaneously output** all the tokens of a target document or composition, including their full 4D positions (t, x, y, z) and their parent–child relationships (which define the hierarchy).

This is a significant step beyond token-by-token generation. It implies that the model, having processed the input (e.g., an ARC input grid or a piece of p5.js code), directly predicts the entire structured set of output tokens that form the solution or corresponding representation.

This could be conceptualized as:

1. Predicting a set of `[Name, Type, Value]` tuples.  
2. For each tuple, predicting its `Position (t, x, y, z)`.  
3. For each tuple, predicting its `Parent` token from the set of contemporaneously generated tokens, thereby forming the output hierarchy.

This ambitious output strategy aligns with the idea that understanding often involves grasping a whole structure rather than just predicting the next element in a sequence. It presents challenges in terms of:

* Defining the loss function (e.g., set-prediction losses, alignment between predicted and target graphs)  
* Designing the decoding process

Yet, it offers the potential for generating globally coherent and complex structures in a single pass. This approach also relates to ideas from object detection—such as DETR, which uses a fixed set of object queries to predict a set of bounding boxes in parallel—and structured prediction.

# **7\. Summer Goals, Future Evaluation, and Vision**

The research outlined charts a course from identifying fundamental limitations in current AI to proposing novel solutions grounded in geometric principles and advanced data representation.

---

## **7.1. Immediate Focus: Dataset and Architecture Implementation**

The primary goals for the upcoming summer research period are twofold and highly synergistic:

1. **Dataset Creation**

   * Finalize the schema details for the Holarchic 4D Token model.

   * Populate a significant dataset of 4D-addressed code/image pairs.  
     * Develop parsers and tools to extract code structures (AST-like representations from p5.js or other languages).  
     * Extract image features (pixel grids, potentially segmented objects).  
     * Annotate both code and image elements with consistent `[Parent, Name, Type, Value, Position (t, x, y, z)]` tokens.

2. **Model Architecture Implementation**

   * Build a functional prototype of the proposed transformer architecture.  
     * Implement the Multi-Head Latent Attention (MLA) mechanism.  
     * Integrate MonSTER for 4D positional encoding within attention layers.  
     * Design the input processing pipeline for the Holarchic tokens.

   * Develop the output layer and loss functions capable of handling simultaneous generation of structured, 4D-addressed token sets.

---

## **7.2. Evaluation on ARC-AGI**

Upon achieving a stable implementation of the model and dataset, the system will be rigorously evaluated on the ARC-AGI challenge. Success will be measured not just by raw accuracy but also by the model’s ability to:

* **Generalize from Few Examples**

* **Demonstrate Interpretable Internal Representations** that reflect spatial and procedural understanding

* **Potentially Generate Solutions with Compositional Steps**, if the output mechanism supports hierarchical composition

It is anticipated that the combination of MonSTER’s native spacetime awareness and the hierarchically structured data representation will provide a significant advantage over systems relying on flattened inputs and 1D sequential processing.

---

## **7.3. Long-Term Vision: Advancing General AI**

This research program is driven by a long-term vision of creating AI systems with more human-like general intelligence. The perceived limitations of current LLMs on tasks like ARC highlight that true intelligence requires more than statistical pattern matching; it demands an understanding of structure, causality, geometry, and the ability to reason and adapt in novel situations.

* **Spacetime as a Foundational Inductive Bias**  
  Explicitly grounding AI representations and computations in a 4D spacetime framework (as MonSTER aims to do) provides a powerful and universal inductive bias, reflecting the fundamental arena in which real-world events unfold.

* **Compositionality and Hierarchy**  
  The Holarchic 4D Token model addresses the need for AI to understand and generate compositional structures—how parts combine to form wholes, and how these structures are nested at multiple scales. This is crucial for everything from understanding complex scenes to parsing code or generating coherent narratives.

* **Bridging Perception, Reasoning, and Action**  
  While this phase focuses on representation and generation for tasks like ARC, the underlying framework (spatiotemporal awareness, structured knowledge) is intended to be extensible to systems that perceive, reason, and act in dynamic environments.

The successful execution of this research could lead to a new class of transformer architectures that are not just “language” models or “vision” models in isolation, but rather “spacetime structure” models, capable of a more unified and general understanding of the world. The journey is complex, but the potential to overcome some of the most persistent limitations in current AI makes it a compelling endeavor. The ultimate report will summarize the results of these efforts, detailing performance on ARC-AGI and discussing the broader implications for the future of AI.