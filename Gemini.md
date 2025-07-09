4 Dimension Attention: Structural Embeddings for Native Space-Time Intelligence
Jacob Peterson
peterj29@oregonstate.edu

Abstract
With its toy-scale visual tasks, the 2019 ARC-AGI benchmark again laid bare the jagged nature of AI performance first highlighted by Moravec in the 1980s. Despite impressive linguistic capabilities, leading models of the time demonstrated a pronounced inability to process or reason about visual information. To date, enormous progress on numerous cognitive benchmarks has failed to translate to spatial domains. The root of this persistent limitation is architectural: modern transformers inherently encode information as strictly linear, unidimensional sequences. Addressing this limitation requires moving beyond positional encodings uniquely restricted to temporal indices. Instead, embeddings must inherently reflect the intrinsic dependencies between space and time. By extending Rotary Positional Embeddings (RoPE) from two to four dimensions via a Minkowski-metricized Clifford (1,3) algebra, we introduce structural embeddings—Minkowski Space Time Embedding Rotors (MonSTER)—that naturally capture the fundamental interdependencies of our universe and the objects and events within it. This approach employs Lorentz transformations, derived from relative spacetime displacements, as the mechanism for positional encoding, thereby embedding the native geometry of special relativity directly into the attention mechanism. MonSTER eliminates the artificial flattening requirement that necessarily discards vital geometric and relational information , and removes the structural blinders that previously constrained transformers' capacity to handle inherently multidimensional information. Simply put, it provides transformers with a built-in ability to perceive, reason about, and generalize across spatial structures and temporal sequences—without sacrificing their established computational advantages, potentially unlocking new capabilities for AI to understand and interact with dynamic, 4D environments in a manner more aligned with physical laws.   

1. Introduction
The landscape of artificial intelligence has witnessed remarkable progress, particularly with the advent of transformer architectures that excel in domains like natural language processing. However, a persistent challenge remains in imbuing AI systems with robust spatio-temporal reasoning capabilities. This gap is vividly illustrated by phenomena such as Andrej Karpathy's "Jagged Intelligence" , where models capable of sophisticated tasks like mathematical problem-solving simultaneously falter on seemingly elementary problems involving spatial or simple causal understanding. This observation echoes Hans Moravec's earlier paradox concerning the relative ease of achieving adult-level performance in abstract problem-solving compared to the profound difficulty of instilling basic perceptual and motor skills akin to those of a one-year-old. This "jaggedness" is not confined to visual tasks, as exemplified by the Abstraction and Reasoning Corpus (ARC-AGI) benchmark, but points to a more fundamental inability of current architectures to holistically model the integrated nature of space and time inherent to the physical world.   

At the core of many leading AI models, transformers are inherently permutation-invariant, necessitating explicit positional information to understand sequential or structural data. Current positional encoding (PE) schemes, however, fall short of providing a true spatio-temporal understanding. Methods like sinusoidal PEs , learned absolute PEs , and even the more advanced Rotary Positional Embeddings (RoPE)  are fundamentally designed for one-dimensional sequences or treat spatial dimensions as separable entities. A common practice, particularly in vision (e.g., Vision Transformer, ViT ), involves flattening multidimensional input into a sequence and applying 1D PEs, or concatenating separate PEs for each dimension. This "flattening" process inevitably discards crucial geometric and relational information inherent in the original data structure. While RoPE introduces geometric intuition through 2D rotations, it operates within a Euclidean framework and typically on pairs of features representing a single effective dimension of position. These approaches do not capture the interdependent fabric of spacetime as described by physics.   

This paper introduces Minkowski Space Time Embedding Rotors (MonSTER), a novel positional encoding scheme that directly embeds the four-dimensional structure of Minkowski spacetime into the attention mechanism of transformers. MonSTER generalizes the 2D Euclidean rotations of RoPE to full 4D Lorentz transformations—encompassing both spatial rotations and relativistic boosts—by employing the Clifford algebra Cl 
1,3
​
 (R) (also known as Spacetime Algebra, STA). Instead of merely extending to more dimensions, MonSTER adopts a fundamentally different geometry: the pseudo-Riemannian geometry of Minkowski space, which is the natural stage for events in special relativity. This choice allows MonSTER to natively handle relative spatio-temporal displacements ΔP=(Δt,Δx,Δy,Δz) between query and key elements, thereby enabling transformers to perceive, reason about, and generalize across spatial structures and temporal sequences in a manner that respects the underlying physics of our universe, including concepts like causality and the invariance of the spacetime interval.   

The primary contributions of this work are:

The formulation of MonSTER, a novel 4D spatio-temporal positional encoding scheme grounded in Clifford algebra Cl 
1,3
​
  and Minkowski geometry.
A detailed derivation demonstrating how Lorentz transformations, represented as rotors (or their matrix equivalents), are computed from relative spacetime displacements and subsequently utilized to modulate attention scores within transformer models.
A theoretical analysis contrasting MonSTER with existing positional encoding methods, with a particular emphasis on its inherent capacity to handle scale, direction, and causal relationships emerging naturally from its formulation.
A discussion of computational considerations, implementation trade-offs, and the potential for MonSTER to unlock new AI capabilities in understanding and interacting with dynamic, four-dimensional environments, particularly in multimodal contexts.
The subsequent sections of this paper are organized as follows: Section 2 reviews related work in positional encodings and the nascent use of geometric algebra in machine learning. Section 3 provides essential background on Clifford algebra Cl 
1,3
​
  and Minkowski spacetime. Section 4 details the MonSTER methodology. Section 5 proposes experimental setups for validation. Section 6 discusses the implications, advantages, and limitations of MonSTER. Finally, Section 7 concludes the paper.

2. Related Work
The quest for effective positional representations in transformers has led to a variety of approaches, evolving from simple absolute encodings to more sophisticated relative and geometrically inspired methods.

Evolution of Positional Encodings in Transformers
Absolute Positional Encodings (APE): Early transformer models relied on absolute positional encodings to inject sequence order information. The seminal "Attention Is All You Need" paper introduced sinusoidal positional encodings, which are fixed, non-learned functions of position. These encodings map each absolute position p to a d-dimensional vector using sine and cosine functions of varying frequencies: PE 
(pos,2i)
​
 =sin(pos/10000 
2i/d 
model
​
 
 ) and PE 
(pos,2i+1)
​
 =cos(pos/10000 
2i/d 
model
​
 
 ). A key advantage is their ability to theoretically generalize to sequence lengths not seen during training. However, being absolute, they may not optimally capture relative relationships between tokens.
Another common approach is learned absolute positional embeddings, where a unique vector is learned for each position up to a maximum sequence length. These are often implemented as an embedding layer added to the input token embeddings (e.g., X 
input
​
 =X 
token
​
 +P 
absolute
​
 ). Vision Transformers (ViTs) typically employ such learned 1D positional embeddings for their sequence of image patches. While flexible, learned APEs do not generalize to positions beyond their training range and can be data-intensive.   

Relative Positional Embeddings (RPE): Recognizing the importance of relative positioning, several methods were developed to directly encode the relationship between pairs of tokens. These often involve modifying the attention mechanism to incorporate a bias term dependent on the relative distance j−i between key j and query i. This approach directly informs the attention score about the proximity and order of tokens.
Rotary Positional Embeddings (RoPE) represent a significant advancement in relative PEs. RoPE encodes the relative position by applying a rotation matrix to the query and key vectors, where the rotation angle is a function of their absolute positions m and n. Specifically, for a d-dimensional feature vector, features are grouped into pairs (x 
k
​
 ,x 
k+1
​
 ), and each pair is rotated in its 2D plane by an angle mθ 
k
​
 . The inner product between a rotated query q 
m
​
  and a rotated key k 
n
​
  then becomes solely a function of their relative position m−n and their original values, f(q,k,m−n). This elegantly incorporates relative positional information while maintaining linear attention complexity. RoPE operates by applying 2D Euclidean rotations, implicitly assuming a 2D feature subspace for each positional frequency. MonSTER builds upon this rotational concept, extending it to the four dimensions of spacetime with a non-Euclidean metric.   

Positional Encodings for Spatial and Spatio-Temporal Data
Adapting transformers for spatial and spatio-temporal data has necessitated specialized PE strategies.
Vision Transformers (ViT) typically divide an image into a sequence of flattened patches and apply learned 1D absolute positional embeddings to this sequence. While successful for image classification, this "flattening" approach discards the inherent 2D/3D structure of the input. Some ViT variants explore interpolating these 1D PEs for variable input resolutions or introduce more 2D-aware PEs. For instance, "Fuzzy Positional Encoding"  introduces positional perturbations to improve generalization across resolutions.   

Perceiver and Perceiver IO architectures are designed for multimodal inputs and employ Fourier features for positional encoding. These are typically fixed, high-dimensional encodings generated by applying sine and cosine functions to input coordinates, scaled by various frequencies, similar to the original sinusoidal PEs but often applied to continuous coordinate inputs (e.g., pixel coordinates, time steps). This allows Perceiver models to handle diverse data types like images, audio, video, and point clouds by providing a canonical way to represent position within a potentially unstructured byte array input. However, these Fourier features are generally applied to Euclidean coordinates.   

Neural Radiance Fields (NeRF) utilize high-frequency positional encoding, specifically Fourier features, for 3D spatial coordinates (x,y,z) and 2D viewing directions (θ,ϕ). This encoding, γ(p)=(sin(2 
0
 πp),cos(2 
0
 πp),...,sin(2 
L−1
 πp),cos(2 
L−1
 πp)), maps low-dimensional inputs to a higher-dimensional space. This is crucial for enabling MLPs to represent fine, high-frequency details in scenes, overcoming the inherent spectral bias of neural networks towards learning low-frequency functions. MonSTER also employs frequency scaling but within a relativistic, rather than Euclidean, geometric framework.   

The common thread among these "spatial" PEs is their operation within Euclidean spaces or their treatment of different dimensions (like time and space) as separable entities. RoPE rotates in a 2D Euclidean plane; ViT flattens spatial data into a 1D sequence; Perceiver and NeRF apply Fourier features to coordinates typically assumed to be Euclidean. None of these approaches inherently embed the pseudo-Riemannian geometry of Minkowski spacetime and its direct implications for causality and relative motion into the attention mechanism's positional information. MonSTER aims to fill this gap by directly leveraging the structure of Lorentz transformations, which are the fundamental symmetries of spacetime, as the basis for its positional encoding. This represents a logical next step in the evolution from absolute PEs, to relative PEs, to PEs incorporating geometric transformations, by elevating the transformation to the geometry most relevant for physical events. By baking in this fundamental physical structure, MonSTER provides a strong inductive bias, potentially leading to better generalization and data efficiency for tasks where this structure is paramount, rather than attempting to learn it implicitly from generic PEs.

Geometric Algebra in Machine Learning
There is a burgeoning interest in leveraging geometric algebra (GA) to create more structured and geometrically-aware representations in machine learning. GA offers a unified mathematical language for geometry, providing intrinsic, coordinate-free ways to express geometric concepts and operations that generalize beyond three dimensions. This aligns with the goals of MonSTER, positioning it within an emerging field that values the incorporation of fundamental geometric principles into neural architectures.   

Table 1 provides a comparative overview of MonSTER against prominent existing positional encoding schemes.

\begin{table}[h!]
\centering
\caption{Comparison of Positional Encoding Schemes.}
\label{tab:pe_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l|llllll@{}}
\toprule
\textbf{Feature} & \textbf{MonSTER (Ours)} & \textbf{RoPE} & \textbf{Sinusoidal PE} & \textbf{Learned Abs. PE} & \textbf{ViT PE} & \textbf{Fourier Feat. PE} \
&  & \scriptsize{(Su et al., 2021)} & \scriptsize{(Vaswani et al., 2017)} &  & \scriptsize{(Dosovitskiy et al., 2021)} & \scriptsize{(Tancik et al., 2020; Jaegle et al., 2021b)} \
\midrule
\textbf{Dimensionality} & Native 4D (t,x,y,z) & 1D pos → 2D feat rot & 1D pos → d-dim vec & 1D pos → d-dim vec & 1D seq pos → d-dim vec & N-D coord → M-dim vec \
\textbf{Metric} & Minkowski (+−−−) & Euclidean (implicit) & N/A (encodes order) & N/A (learns order) & N/A (learns order) & Euclidean \
\textbf{Transformation} & Lorentz (Rot & Boost) & 2D Rotation & N/A & N/A & N/A & N/A (mapping) \
\textbf{Nature} & Relative (from ΔP) & Relative & Absolute & Absolute & Absolute (typically) & Absolute (coord based) \
\textbf{Parameters} & Freq. bases, C 
t
​
  & Freq. base (fixed) & Fixed & Learned & Learned & Freq. bands (fixed/learnable) \
\textbf{Spacetime Handling} & Integrated & Decoupled, 1D pos & Decoupled, 1D pos & Decoupled, 1D pos & Flattened, 1D seq pos & Decoupled N-D Euclidean coords \
\textbf{Causality Aware} & Yes (via η, Δt) & No & No & No & No & No \
\textbf{Key Limitation} & Higher compute; & Not 4D spatio-temporal & Not relative/spatio-temp. & Generalization; not & Flattens spatial struct. & Assumes Euclidean space \
\textbf{for Spatio-Temp.} & new concept &  &  & relative/spatio-temp. &  &  \
\bottomrule
\end{tabular}%
}
\end{table}

3. Background: Geometric Algebra for Spacetime
Traditional approaches to positional encoding in machine learning often implicitly assume a Euclidean geometric framework. In such a framework, spatial distances are calculated using the Pythagorean theorem, and time is typically treated as an independent, absolute dimension, or simply another orthogonal Euclidean dimension. However, physical reality, particularly concerning events involving motion and signal propagation, is more accurately described by the geometry of Minkowski spacetime. In Minkowski spacetime, the interval between two events is invariant across all inertial frames of reference, unlike spatial distances or time durations, which are relative. This necessitates a mathematical language capable of natively handling this structure.   

Clifford Algebra Cl 
1,3
​
 (R) — Spacetime Algebra (STA)
Spacetime Algebra (STA) is the application of the Clifford algebra Cl 
1,3
​
 (R) to the 4-dimensional Minkowski spacetime. It provides a comprehensive and coordinate-free framework for relativistic physics. STA is built upon a vector space equipped with a geometric product that unifies and extends the familiar dot and wedge products.   

Basis and Metric: The standard basis for STA consists of four orthonormal vectors {γ 
0
​
 ,γ 
1
​
 ,γ 
2
​
 ,γ 
3
​
 }, where γ 
0
​
  is timelike and γ 
1
​
 ,γ 
2
​
 ,γ 
3
​
  are spacelike. The algebra is defined by the metric signature, which we take as (+,−,−,−), consistent with much of the STA literature  and the implementation provided. This signature implies:
γ 
0
​
 ⋅γ 
0
​
 =γ 
0
2
​
 =1

γ 
i
​
 ⋅γ 
i
​
 =γ 
i
2
​
 =−1for i∈{1,2,3}

γ 
μ
​
 ⋅γ 
ν
​
 =0for μ

=ν

The Minkowski metric tensor η 
μν
​
  is therefore diag(1,−1,−1,−1).   

Geometric Product: For any two vectors a,b∈R 
1,3
 , their geometric product is defined as:
ab=a⋅b+a∧b

where a⋅b= 
2
1
​
 (ab+ba) is the symmetric inner product (a scalar, representing the projection of one vector onto another, scaled by the metric), and a∧b= 
2
1
​
 (ab−ba) is the antisymmetric outer product (a bivector, representing the oriented plane spanned by a and b). This product is associative and distributive but not generally commutative.   

Multivectors: The elements of STA are multivectors, which are linear combinations of objects of different grades: scalars (grade 0), vectors (grade 1), bivectors (grade 2), trivectors (grade 3, also called pseudovectors), and a pseudoscalar (grade 4, I=γ 
0
​
 γ 
1
​
 γ 
2
​
 γ 
3
​
 ).
Bivectors are particularly important as they serve as the generators of Lorentz transformations. The six basis bivectors in Cl 
1,3
​
  are:

Timelike bivectors (generators of boosts): γ 
1
​
 γ 
0
​
 ,γ 
2
​
 γ 
0
​
 ,γ 
3
​
 γ 
0
​
 . (Note: some authors use γ 
0
​
 γ 
i
​
 ). These square to +1.
Spacelike bivectors (generators of spatial rotations): γ 
2
​
 γ 
3
​
 ,γ 
3
​
 γ 
1
​
 ,γ 
1
​
 γ 
2
​
 . These square to −1.    
Minkowski Metric vs. Euclidean Metric
The fundamental difference lies in how "distance" is measured.

Euclidean Metric: For two points separated by (Δx,Δy,Δz), the squared distance is d 
2
 =Δx 
2
 +Δy 
2
 +Δz 
2
 . This is always non-negative and is preserved by Euclidean rotations and translations.
Minkowski Metric: For two events separated by (Δt,Δx,Δy,Δz), the squared spacetime interval (with c=1) is s 
2
 =(Δt) 
2
 −(Δx 
2
 +Δy 
2
 +Δz 
2
 ). This interval s 
2
  is invariant under Lorentz transformations.   
If s 
2
 >0, the separation is timelike: the events are causally connected, and the temporal order is absolute for all inertial observers.
If s 
2
 <0, the separation is spacelike: the events are causally disconnected, and their temporal order can be relative (observer-dependent).
If s 
2
 =0, the separation is lightlike (or null): the events can be connected by a light signal. This causal structure, naturally emerging from the Minkowski metric, is absent in Euclidean geometry and is crucial for describing physical interactions.
Lorentz Transformations as Rotors in Cl 
1,3
​
 
Proper orthochronous Lorentz transformations (spatial rotations and boosts) can be elegantly represented by rotors in STA. A rotor R is an element of the even subalgebra Cl 
1,3
+
​
 (R) (composed of scalars, bivectors, and the pseudoscalar) satisfying R 
R
~
 =1, where  
R
~
  is the reverse of R (obtained by reversing the order of all products of vectors in R). Rotors can be expressed in exponential form:
R=e 
−B/2
 

where B is a bivector.   

If B is a spacelike bivector (e.g., B=θ(γ 
2
​
 γ 
1
​
 ), with B 
2
 <0), R represents a spatial rotation in the γ 
1
​
 γ 
2
​
  plane by an angle θ. R=cos(θ/2)−γ 
2
​
 γ 
1
​
 sin(θ/2).
If B is a timelike bivector (e.g., B=ϕ(γ 
1
​
 γ 
0
​
 ), with B 
2
 >0), R represents a Lorentz boost in the γ 
1
​
 γ 
0
​
  spacetime plane with rapidity ϕ. R=cosh(ϕ/2)−γ 
1
​
 γ 
0
​
 sinh(ϕ/2).
A 4-vector v is transformed to v 
′
  by the sandwich product:
v 
′
 =Rv 
R
~
 

This operation preserves the grade of v (i.e., a vector transforms to a vector) and, critically, preserves the Minkowski inner product between any two vectors u,v: (Ru 
R
~
 )⋅(Rv 
R
~
 )=u⋅v.
While rotors provide a coordinate-free and geometrically intuitive representation, for practical implementation in standard deep learning frameworks, it is often necessary to convert them to 4x4 matrices acting on coordinate 4-vectors. The matrix elements Λ 
μ
ν
​
  of the Lorentz transformation corresponding to a rotor R can be obtained by observing the transformation of the basis vectors:
Λ 
μ
ν
​
 =(γ 
ν
 ⋅(Rγ 
μ
​
  
R
~
 ))

where γ 
ν
  are the reciprocal basis vectors. The MonSTER method, as detailed in the provided code, constructs these 4x4 matrices for rotation and boost components directly.   

The choice of Cl 
1,3
​
  is not arbitrary; it is the minimal algebraic structure that naturally encodes the symmetries of Minkowski spacetime (the Lorentz group) and allows for a grade-respecting product where vectors combine to form bivectors, the generators of these symmetries. Rotors in STA are not mere mathematical constructs; they embody the geometric operations themselves, acting directly on geometric objects rather than just their coordinates in a specific basis.   

Comparison: Rotors vs. Quaternions/Biquaternions
Quaternions are well-established for representing 3D rotations, being isomorphic to SU(2), the double cover of SO(3). Biquaternions (complexified quaternions) extend this to represent general Lorentz transformations, as they are isomorphic to SL(2,C), the double cover of the proper orthochronous Lorentz group SO 
+
 (1,3). A biquaternion q can transform a "minquat" (a biquaternion representing a 4-vector) g via an operation like g 
′
 =qg 
q
ˉ
​
  
∗
  (where  
q
ˉ
​
  is complex conjugation and q 
∗
  is quaternion conjugation).   

However, STA rotors, which are elements of Cl 
1,3
+
​
 (R) (itself isomorphic to biquaternions), offer several advantages, particularly in conceptual clarity and extensibility:

Geometric Clarity: Bivectors B (the exponents in R=e 
−B/2
 ) are directly interpretable as the oriented spacetime planes of rotation/boost. The structure of boosts and rotations arises naturally from the properties of timelike and spacelike bivectors within the single algebra of spacetime.   
Unified Framework: STA handles all grades of multivectors (scalars, vectors, bivectors, etc.) and their transformations under the Lorentz group consistently within one algebraic system. Vectors being transformed and the rotors transforming them are all elements of the same algebra.   
Extensibility: STA and geometric algebra in general are readily extensible to different dimensions or metric signatures, providing a more general tool for physics and geometry.   
While mathematically equivalent for representing Lorentz transformations (since Cl 
1,3
+
​
 (R)≅H 
C
​
 ), STA provides a richer, more direct geometric interpretation and a more integrated algebraic environment than treating biquaternions as an external algebra applied to spacetime vectors.

4. Method: Minkowski Space Time Embedding Rotors (MonSTER)
The objective of MonSTER is to furnish transformer models with a learnable, relative 4-dimensional spatio-temporal positional embedding that natively respects the geometry of Minkowski spacetime. This allows the model to process information about the relative positions of events or tokens in a manner consistent with the principles of special relativity.

Core Idea: Generalizing RoPE to Spacetime
Rotary Positional Embeddings (RoPE) achieve relative positional encoding by applying 2D rotation matrices to query and key vectors, where the rotation angle depends on the absolute 1D position of the tokens. This effectively operates on pairs of features, treating them as coordinates in a 2D Euclidean plane.
MonSTER generalizes this concept significantly:   

It extends the dimensionality from 1D position to 4D spacetime position P=(t,x,y,z).
It replaces the 2D Euclidean rotation with a 4D Lorentz transformation, which is a "rotation" in Minkowski spacetime and includes both spatial rotations and relativistic boosts.
These Lorentz transformations are derived from the relative 4D spacetime displacement ΔP=P 
key
​
 −P 
query
​
  between the query and key elements.
The transformations are implemented using the Clifford algebra Cl 
1,3
​
 (R), ensuring that the underlying Minkowski metric is respected.
Queries and keys are envisioned to have feature dimensions that are multiples of four. Each block of four features is treated as a 4-vector in Minkowski space, analogous to RoPE treating features in pairs for 2D rotations. The attention modulation Q 
T
 ηRK then becomes a sum of Minkowski inner products between these transformed 4-vector feature blocks.

Input Representation
Each token i in an input sequence is assumed to be associated with a 4-position P 
i
​
 =(t 
i
​
 ,x 
i
​
 ,y 
i
​
 ,z 
i
​
 ). For vision tasks like those in ARC-AGI, these coordinates might represent pixel locations (x 
i
​
 ,y 
i
​
 ,z 
i
​
 =0) and a temporal or step index t 
i
​
 . The raw relative displacement between a query token q at P 
query
​
  and a key token k at P 
key
​
  is:
$$ \Delta P_{raw} = P_{key} - P_{query} = (\Delta t_{raw}, \Delta x_{raw}, \Delta y_{raw}, \Delta z_{raw}) $$

Derivation of the Effective Lorentz Transformation (R 
eff
​
 ) per Block b
To capture relationships at multiple scales, the embedding dimension is divided into num_blocks. Each block b utilizes different frequency scalings for the raw displacements.

1. Frequency Scaling:
Separate inverse frequencies are applied to the temporal and spatial components of ΔP 
raw
​
 :
$$\text{inv_freq_time}_b = 1.0 / (\text{base_time}^{(\text{freqs}_b / \text{num_blocks})})$$
$$\text{inv_freq_space}_b = 1.0 / (\text{base_space}^{(\text{freqs}_b / \text{num_blocks})})$$
where freqs_b is the index of the current block. This yields scaled displacements:
$$\Delta t_{scaled,b} = \Delta t_{raw} \cdot \text{inv_freq_time}_b$$
$$ \Delta \mathbf{s}{scaled,b} = (\Delta x{raw} \cdot \text{inv_freq_space}b, \Delta y{raw} \cdot \text{inv_freq_space}b, \Delta z{raw} \cdot \text{inv_freq_space}_b) $$
This multi-scale approach, analogous to RoPE's use of different θ 
i
​
   or Fourier Features' frequency bands , allows the model to be sensitive to varying magnitudes of temporal and spatial separation.   

2. Spatial Rotation Component (M 
rot,b
​
 ):
A purely spatial rotation is derived from the scaled spatial displacement Δs 
scaled,b
​
 :

Rotation angle: θ 
b
​
 =∣∣Δs 
scaled,b
​
 ∣∣ 
2
​
 .
Rotation axis u 
rot,b
​
 : Δs 
scaled,b
​
 /∣∣Δs 
scaled,b
​
 ∣∣ 
2
​
 . If ∣∣Δs 
scaled,b
​
 ∣∣ 
2
​
 ≈0, the rotation is identity, and u 
rot,b
​
  defaults to a predefined axis (e.g.,  
T
 , as in the provided code). A 3x3 spatial rotation matrix R 
3,b
​
  is constructed using Rodrigues' rotation formula : $$ R(\mathbf{u}, \theta) = \cos\theta I + (1-\cos\theta)\mathbf{u}\mathbf{u}^T + \sin\theta [\mathbf{u}]_\times $$ where I is the 3x3 identity matrix and [u] 
×
​
  is the skew-symmetric cross-product matrix of u. This R 
3,b
​
  is then embedded into a 4x4 matrix M 
rot,b
​
  that only affects spatial components:
M 
rot,b
​
 =( 
1
0
​
  
0 
T
 
R 
3,b
​
 
​
 )
  
3. Lorentz Boost Component (M 
boost,b
​
 ):
A Lorentz boost is derived primarily from the scaled temporal displacement Δt 
scaled,b
​
 :

Boost rapidity ϕ 
b
​
 : The raw scaled temporal displacement ϕ 
b,prescale
​
 =Δt 
scaled,b
​
  is passed through a hyperbolic tangent function for stability and boundedness:
ϕ 
b
​
 =C 
t
​
 ⋅tanh(ϕ 
b,prescale
​
 /C 
t
​
 )
where C 
t
​
  (time_rapidity_scale_C_t) is a scaling factor. The tanh function maps potentially unbounded Δt 
scaled,b
​
  to a bounded range, preventing extreme rapidities and associated numerical instability in cosh(ϕ 
b
​
 ) and sinh(ϕ 
b
​
 ).
Boost axis u 
boost,b
​
 : The current implementation sets u 
boost,b
​
 =u 
rot,b
​
 . This implies the boost is applied along the direction of the scaled spatial displacement. This is a specific choice that simplifies the transformation. The 4x4 Minkowski boost matrix M 
boost,b
​
  for a boost along n=(n 
x
​
 ,n 
y
​
 ,n 
z
​
 ) 
T
 =u 
boost,b
​
  with rapidity ϕ 
b
​
  is : Let γ 
b
​
 =cosh(ϕ 
b
​
 ) and β 
b
​
 γ 
b
​
 =sinh(ϕ 
b
​
 ). (Note: β 
b
​
  here is velocity, not a gamma matrix). More directly, using ch=cosh(ϕ 
b
​
 ) and sh=sinh(ϕ 
b
​
 ): $$ M_{boost,b} = \begin{pmatrix} \text{ch} & -n_x \text{sh} & -n_y \text{sh} & -n_z \text{sh} \ -n_x \text{sh} & 1+(\text{ch}-1)n_x^2 & (\text{ch}-1)n_x n_y & (\text{ch}-1)n_x n_z \ -n_y \text{sh} & (\text{ch}-1)n_y n_x & 1+(\text{ch}-1)n_y^2 & (\text{ch}-1)n_y n_z \ -n_z \text{sh} & (\text{ch}-1)n_z n_x & (\text{ch}-1)n_z n_y & 1+(\text{ch}-1)n_z^2 \end{pmatrix} $$   
4. Combined Transformation (R 
eff,b
​
 ):
The effective Lorentz transformation for block b is obtained by composing the boost and rotation:
R 
eff,b
​
 =M 
boost,b
​
 M 
rot,b
​
 

The provided code applies the rotation first, then the boost. Due to the specific choice u 
boost,b
​
 =u 
rot,b
​
 , these two operations (boost along an axis and rotation around the same axis) commute. A general Lorentz transformation, where boost and rotation axes differ, would not commute and would require a more complex decomposition (e.g., polar decomposition). The current formulation explores a specific, albeit important, subset of the Lorentz group.
This R 
eff,b
​
  matrix represents how the coordinates of a 4-vector would transform from a frame centered at the key token (and appropriately moved/rotated relative to the query token by ΔP) to a frame centered at the query token.

Attention Modulation
Let Q 
i
​
  and K 
j
​
  be the query and key tensors for tokens i and j, respectively. These are conceptually divided into num_blocks blocks, Q 
i,b
​
  and K 
j,b
​
 , each of dimension $d_{block} = d_{model} / \text{num_blocks}$. Each Q 
i,b
​
  and K 
j,b
​
  is treated as a 4-vector (assuming d 
block
​
 =4, or further subdivided if d 
block
​
  is a multiple of 4). The geometric part of the attention score between query i and key j is computed as a sum over the blocks:
score(i,j)= 
b
∑
​
 (Q 
i,b
​
 ) 
T
 η(R 
eff,b
​
 (ΔP 
ji
​
 )K 
j,b
​
 )

where ΔP 
ji
​
 =P 
j
​
 −P 
i
​
 , and η=diag(1,−1,−1,−1) is the Minkowski metric tensor. This formulation applies the relative Lorentz transformation R 
eff,b
​
  to the key vector K 
j,b
​
  and then computes its Minkowski inner product with the query vector Q 
i,b
​
 . This structure ensures that the attention score is sensitive to the relative spacetime geometry encoded by R 
eff,b
​
 .

Emergent Properties
This formulation naturally gives rise to several desirable properties:

Scale Sensitivity: The use of different frequency scalings (inv_freq_time_b, inv_freq_space_b) across blocks allows the model to be sensitive to spatio-temporal relationships at various scales.
Directionality: Spatial direction is explicitly encoded in u 
rot,b
​
  (and thus u 
boost,b
​
  in the current implementation).
"Before vs. After" (Causality):
The sign of Δt 
raw
​
  directly indicates the temporal order between P 
key
​
  and P 
query
​
 . This sign propagates to Δt 
scaled,b
​
  and subsequently to the rapidity ϕ 
b
​
 . Since sinh(ϕ 
b
​
 ) is an odd function, its sign depends on the sign of ϕ 
b
​
 , influencing the time-space mixing terms (g 
0i
​
 ,g 
i0
​
 ) in M 
boost,b
​
 . This makes the transformation inherently sensitive to whether an event is in the temporal past or future.
More fundamentally, the Lorentz transformation R 
eff,b
​
  is constructed from Δt and Δs, the very components that define the spacetime interval s 
2
 =Δt 
2
 −∣∣Δs∣∣ 
2
 . While s 
2
  itself is not explicitly used to form R 
eff,b
​
 , the transformation respects the causal structure implied by s 
2
 . For instance, it correctly transforms timelike separated vectors differently from spacelike separated vectors, preserving the type of interval. This means that the notion of absolute "before/after" for timelike separations and relative "before/after" for spacelike separations is implicitly respected by the transformations applied.
5. Experiments
To empirically validate the efficacy of MonSTER in capturing and utilizing 4D spatio-temporal relationships for improved reasoning, a series of experiments are proposed. While comprehensive benchmarks specifically designed to evaluate native 4D spatio-temporal reasoning with explicit relativistic considerations are still emerging , the following suite of synthetic and real-world-inspired tasks is designed to rigorously assess MonSTER's unique capabilities.   

Proposed Datasets and Tasks
Abstraction and Reasoning Corpus (ARC-AGI Benchmark):

Description: The ARC-AGI benchmark (Chollet, 2019), mentioned in the abstract, tests abstract reasoning with a significant visual-spatial component. The "step" differences in the MonSTER code hint at its applicability to tasks involving sequences of actions or transformations, common in ARC.
Evaluation: Task completion accuracy.
Rationale: Directly addresses the motivating problem of "jagged intelligence" in spatial domains and serves as a test of general reasoning with a spatio-temporal flavor.
Synthetic Spatio-Temporal Relativistic Reasoning Tasks:
These tasks are crucial for directly probing MonSTER's understanding of Minkowski geometry and relativistic effects, which may not be explicitly tested by standard benchmarks.

Task 1: Relative Event Ordering under Relativistic Motion:
Setup: Generate scenarios with multiple events (E 
1
​
 ,E 
2
​
 ,...) defined by spacetime coordinates (t,x,y,z) in a "lab" frame. Define multiple observer frames moving at constant relativistic velocities v relative to the lab frame. The task is to determine the temporal order (before, after, simultaneous) of pairs of events as perceived in each observer frame. This requires applying Lorentz transformations to event coordinates.   
Evaluation: Accuracy in predicting event order per observer frame.
Rationale: Directly tests understanding of relativity of simultaneity and time dilation.
Task 2: Causal Path Identification:
Setup: Generate a set of events in spacetime. A directed causal link exists from event A to event B if B is within A's future light cone (i.e., Δt>0 and (Δt) 
2
 −∣∣Δs∣∣ 
2
 ≥0) and potentially satisfies other proximity or interaction criteria. The task could be to predict if a causal path exists between two distant events, or to identify missing intermediate events in a causal chain.
Evaluation: Precision, recall, F1-score for causal link/path prediction.
Rationale: Tests understanding of causal structure defined by the Minkowski metric.
Task 3: Spatio-Temporal Object Tracking & Prediction in Simulated Relativistic Environments:
Setup: Utilize physics simulators (e.g., modified MuJoCo  or Isaac Gym , or a custom simulator) capable of simulating object dynamics where relativistic effects are non-negligible (e.g., high speeds, strong fields if extended beyond special relativity). Generate trajectories of interacting objects. The task is to predict future states (4-position, 4-velocity) or identify trajectories inconsistent with relativistic kinematics.   
Evaluation: Mean Squared Error for prediction; classification accuracy for anomaly detection.
Rationale: Tests predictive capabilities in dynamic systems governed by (or approximating) relativistic physics.
Real-World Multimodal Datasets (Exploratory):

Video Action Recognition with Complex Interactions: Datasets like ActionAtlas , which focuses on intricate, domain-specific actions where subtle spatio-temporal relationships are key, or other datasets emphasizing complex multi-agent interactions.   
Evaluation: Action classification accuracy, mean Average Precision (mAP) for temporal action localization.
Rationale: To assess MonSTER's ability to handle real-world visual complexity where relative spacetime positioning of actors and objects is crucial for disambiguation.
Referring Expressions in Video with Temporal Constraints: Datasets requiring joint understanding of language and video, where temporal phrases (e.g., "the person who waved before the car passed") specify targets.
Evaluation: Localization accuracy (e.g., IoU with ground truth bounding boxes/masks).
Rationale: Tests multimodal reasoning grounded in spatio-temporal relationships.
Baselines for Comparison
To contextualize MonSTER's performance, it will be compared against:

Standard Transformer with 2D RoPE (applied to concatenated spatial and temporal features, or to spatial features with a separate 1D temporal encoding).
Transformer with Sinusoidal Positional Encoding (1D encodings concatenated for each of the 4 dimensions).
Transformer with Learned Absolute Positional Embeddings (concatenated for 4D, or a single large learned embedding).
ViT-style patch embeddings + learned 1D PE (for visual tasks, adapted to sequences of spatio-temporal patches if applicable).
Perceiver IO-style Fourier Features applied to (t,x,y,z) coordinates.
"Euclidean MonSTER" (Critical Ablation): A variant of MonSTER where the Minkowski metric η is replaced by a Euclidean metric (I 
4
​
 ), Lorentz boosts are replaced by simple translations (or identity if relative positions are already differences), and spatial rotations remain Euclidean. This baseline will isolate the specific contribution of the Minkowski geometry and Lorentz transformations.
Evaluation Metrics
Beyond task-specific metrics, the evaluation will include:

Convergence Speed: Epochs or iterations to reach target performance.
Performance Scaling: Accuracy as a function of sequence length or spatial/temporal extent of the input.
Robustness: Performance under noisy positional inputs.
Probing Spacetime Interval Understanding: Designing diagnostic tests to assess if attention patterns or internal representations systematically differ for timelike, spacelike, and null separated token pairs, even when their Euclidean distances might be similar. This could involve analyzing attention weights or using probing classifiers on hidden states.   
Proposed Ablation Studies
To understand the contribution of MonSTER's components:

Minkowski vs. Euclidean Metric: Compare full MonSTER with the "Euclidean MonSTER" baseline. This is the most crucial ablation to demonstrate the value of the relativistic formulation.
Impact of Boost vs. Rotation: Evaluate versions with only M 
rot,b
​
  (spatial rotations), only M 
boost,b
​
  (boosts), and both.
Effect of tanh Clamping on Rapidity: Vary C 
t
​
  or remove the tanh function to assess its impact on stability and performance.
Separate vs. Shared Frequencies for Time/Space: Investigate the effect of using different base_time and base_space values versus a single shared base.
Number of Blocks (num_blocks): Analyze performance as the number of frequency blocks (and thus the granularity of multi-scale representation) is varied.
Future work will involve implementing and evaluating MonSTER on these proposed benchmarks. The development of targeted synthetic tasks focusing on relativistic phenomena is considered essential to demonstrate that MonSTER offers more than just increased complexity, but a genuinely more aligned geometric understanding for certain classes of problems.

6. Discussion
MonSTER introduces a paradigm shift in positional encoding by grounding it in the fundamental physics of Minkowski spacetime. This approach offers several conceptual advantages and opens new avenues for research, alongside computational considerations and limitations.

Analysis of Advantages
The primary advantage of MonSTER lies in its principled spatio-temporal representation. By employing Cl 
1,3
​
  algebra and the Minkowski metric, it provides a more natural and physically coherent way to represent events and their interrelations compared to methods that rely on Euclidean geometry or treat spacetime dimensions as separable and independent. This framework is the native language of special relativity, which governs the behavior of objects and signals in the absence of strong gravitational effects.

A key consequence is the native handling of relative motion and causality. Lorentz transformations, which MonSTER uses to modulate attention, inherently account for relativistic effects such as time dilation and the relativity of simultaneity. The Minkowski interval s 
2
 , implicitly determined by the Δt and Δs used to construct the transformations, naturally categorizes event separations into timelike (causally connected, absolute temporal order), spacelike (causally disconnected, relative temporal order), and lightlike. This causal structure is thus intrinsically woven into the positional information.

MonSTER achieves a unified treatment of space and time, integrating them through the spacetime metric and Lorentz transformations, rather than as distinct entities or additional Euclidean dimensions. This unified view is fundamental to modern physics and may offer a more robust inductive bias for AI models dealing with physical systems. Consequently, by incorporating such a strong physical inductive bias, MonSTER has the potential for improved generalization, particularly on tasks where spacetime structure is critical, potentially requiring less data to learn these fundamental relationships.

Computational Complexity, State-Size, and Implementation Trade-offs
The enhanced representational power of MonSTER comes with computational considerations.

Per-Pair Computation: For each query-key pair, MonSTER computes a 4x4 Lorentz transformation matrix R 
eff,b
​
  for each attention block b. This involves:
Calculating scaled displacements.
Constructing a 3x3 spatial rotation matrix via Rodrigues' formula (trigonometric functions, vector products).
Constructing a 4x4 boost matrix (hyperbolic functions, vector products).
A 4x4 matrix multiplication (M 
boost,b
​
 @M 
rot,b
​
 ), which involves 64 multiplications and 48 additions for standard matrix multiplication.
The subsequent attention modulation (Q 
i,b
​
 ) 
T
 η(R 
eff,b
​
 K 
j,b
​
 ) involves a matrix-vector product and a Minkowski inner product, which is more operations than a standard dot product.
Overall Complexity: If applied to all N tokens in a sequence, the attention mechanism remains O(N 
2
 d 
model
​
 ) in complexity, similar to standard self-attention. However, the constant factor associated with the per-pair calculations is larger due to the 4x4 matrix operations. For fixed-size 4x4 operations, this is an O(1) cost per pair per head per block, but this O(1) is more substantial than simpler PEs.   
State Size: MonSTER does not significantly increase the number of learnable model parameters if base_time, base_space, and C_t are fixed hyperparameters. If these are made learnable per layer or head, the parameter increase would be modest. The primary overhead is computational rather than in model size for the PE itself.
Implementation: Requires careful management of feature vectors as 4-vector blocks. The provided JAX code demonstrates an efficient implementation suitable for GPU acceleration.
The trade-off is clear: increased computational cost per attention calculation versus the potential for a significantly richer and more physically grounded representation. This cost may be justifiable for tasks where subtle spatio-temporal relationships, particularly those involving relative motion and causality, are crucial and poorly captured by simpler PEs.

Limitations
Computational Cost: As discussed, MonSTER is likely more computationally intensive than PEs like RoPE or sinusoidal encodings. This could limit its applicability in resource-constrained scenarios or for extremely long sequences without further optimization.
Mathematical Complexity: The underlying framework of Clifford algebra and Lorentz transformations is more complex than that of most PEs, potentially posing a barrier to widespread adoption, understanding, and debugging.
Relevance to Non-Relativistic Regimes: For many common AI tasks, velocities are negligible compared to the speed of light, and spacetime is effectively Euclidean. In such cases, the full machinery of Lorentz transformations might be unnecessary, and simpler Euclidean-based PEs (or the "Euclidean MonSTER" variant) might suffice and be more efficient.
Current Lack of Tailored Benchmarks: Evaluating the unique benefits stemming from the Minkowski geometry requires specialized benchmarks that are not yet standard in the ML community.   
Specific Choice of Boost Axis: The current implementation ties the boost axis to the direction of spatial displacement. While a reasonable simplification, a general Lorentz transformation allows independent boost and rotation axes. This restriction limits the family of transformations MonSTER can represent.
Future Research Directions
MonSTER opens several avenues for future exploration:

Development of Spatio-Temporal Benchmarks: Creating robust benchmarks that specifically probe an AI's understanding of Minkowski geometry, causality, and relativistic effects is crucial.   
Exploration of Alternative Geometric Algebras: Investigating whether other Clifford algebras, such as Cl 
3,1
​
 (R) (often used in quantum field theory) or higher-dimensional GAs like Conformal Geometric Algebra (Cl 
4,1
​
 (R)), which unifies rotations, translations, and dilations, could offer further advantages for specific tasks.
Applications in Physics-Informed ML: Leveraging MonSTER in domains like simulating physical systems, learning from experimental data in particle physics, or modeling astrophysical phenomena where relativistic effects are prominent.
Efficient MonSTER Variants: Researching methods to reduce computational overhead, such as learning low-rank approximations of R 
eff
​
 , factorizing Lorentz transformations, or developing specialized hardware/software kernels.
Learned Lorentz Transformation Components: Exploring architectures where parts of the Lorentz transformation (e.g., preferred frame information, parameters related to local spacetime characteristics) are learned from data rather than being solely derived from ΔP. This could be a step towards models that can adapt to non-inertial frames or even approximate effects of curved spacetime.
Deep Integration with Multimodal Architectures: Developing models where MonSTER provides a common spatio-temporal grounding for diverse modalities like video, audio, text, and sensor streams, enabling more sophisticated cross-modal reasoning.
Interpretability of Failure Modes: Given its physics-based foundation, analyzing MonSTER's failures could provide more interpretable insights, e.g., if it struggles with highly accelerated (non-inertial) reference frames, this points to the limits of its special relativistic assumptions.
The introduction of MonSTER is predicated on the idea that providing AI with the "right" inductive bias—one that reflects the geometry of the physical world—can lead to more robust and generalizable intelligence for tasks involving spatio-temporal reasoning. While special relativity describes flat spacetime, the geometric algebra framework itself is highly extensible, potentially offering pathways towards models that could, in the distant future, even incorporate aspects of general relativity if tasks demanded such sophistication.

7. Conclusion
This paper introduced Minkowski Space Time Embedding Rotors (MonSTER), a novel four-dimensional structural embedding for transformer architectures. By leveraging the mathematical framework of Clifford algebra Cl 
1,3
​
 (R) and the principles of Minkowski spacetime geometry, MonSTER generalizes the rotational concept of RoPE to full 4D Lorentz transformations. These transformations, derived from the relative spacetime displacement between query and key elements, are used to modulate attention scores, thereby embedding a native understanding of spacetime structure directly into the model.

The key contributions of this work include the formulation of a principled geometric foundation for spatio-temporal positional encodings, a demonstration of how properties such as scale, direction, and causality can emerge naturally from this formulation, and the outlining of a path towards more robust artificial intelligence for dynamic, four-dimensional environments. MonSTER offers a unified treatment of space and time, respecting the invariance of the spacetime interval and the causal relationships it implies.

It is posited that MonSTER holds significant potential to address the "jagged intelligence" observed in AI systems when faced with tasks requiring deep spatio-temporal understanding. By equipping transformers with an inductive bias aligned with fundamental physics, MonSTER may enable them to more accurately perceive, reason about, and generalize across complex spatial structures and temporal sequences. This could have broad implications for advancements in multimodal AI, robotics, physics-informed machine learning, and other domains where understanding the intricate dynamics of events in spacetime is paramount. Future work will focus on empirical validation through the proposed specialized benchmarks and exploration of the rich theoretical landscape MonSTER opens up.

8. References
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. International Conference on Learning Representations.    

Jaegle, A., Borgeaud, S., Alayrac, J. B., Doersch, C., Ionescu, C., Ding, D., Koppula, S., Zoran, D., Brock, A., Shelhamer, E., Hénâff, O., Botvinick, M. M., Zisserman, A., Vinyals, O., & Carreira, J. (2021). Perceiver io: A general architecture for structured inputs & outputs. In International Conference on Machine Learning (pp. 4651-4664). PMLR.    

Karpathy, A. (2024, July 25). Jagged Intelligence. X. https://x.com/karpathy/status/1816531576228053133    

Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). Nerf: Representing scenes as neural radiance fields for view synthesis. In A. Vedaldi, H. Bischof, T. Brox, & J. M. Frahm (Eds.), Computer Vision – ECCV 2020 (pp. 405-421). Springer International Publishing.    

Moravec, H. P. (1988). Mind Children: The Future of Robot and Human Intelligence. Harvard University Press.    

Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864.

Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J., & Ng, R. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. Advances in Neural Information Processing Systems, 33, 7537-7547.    

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

(Additional references based on snippet IDs mentioned throughout the text would be formatted and included here, covering topics like Clifford Algebra , Minkowski Space , Lorentz Transformations , Rodrigues' Formula , and other cited works on positional encodings and benchmarks.)   


Sources used in the report

reddit.com
www.reddit.com
Opens in a new window

en.wikipedia.org
Spacetime algebra - Wikipedia
Opens in a new window

books.google.com
Mind Children: The Future of Robot and Human Intelligence - Hans ...
Opens in a new window

papers.nips.cc
Review for NeurIPS paper: Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
Opens in a new window

proceedings.neurips.cc
Fourier Features Let Networks Learn High Frequency Functions in ...
Opens in a new window

cs.jhu.edu
NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
Opens in a new window

drewjaegle.com
Perceiver IO: A General Architecture for Structured Inputs & Outputs - Drew Jaegle
Opens in a new window

scirp.org
Dosovitskiy, A. (2021) An Image Is Worth 16 16 Words Transformers for Image Recognition at Scale. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, New York City, 23-26 June 2021, 45-67. - References - Scientific Research Publishing
Opens in a new window

viso.ai
Neural Radiance Fields (NeRFs): A Technical Exploration - viso.ai
Opens in a new window

docs.nerf.studio
NeRF - nerfstudio
Opens in a new window

arxiv.org
perceiver io - arXiv
Opens in a new window

patrick-llgc.github.io
Perceiver IO: A General Architecture for Structured Inputs & Outputs
Opens in a new window

arxiv.org
ViTAR: Vision Transformer with Any Resolution - arXiv
Opens in a new window

kaggle.com
Positional Encoding - ViT - Kaggle
Opens in a new window

datascience.stackexchange.com
Why do position embeddings work? - Data Science Stack Exchange
Opens in a new window

modular.com
Rotary Position Embedding (RoPE) - AI Resources - Modular
Opens in a new window

mathworks.com
sinusoidalPositionEncodingLayer - MathWorks
Opens in a new window

machinelearningmastery.com
A Gentle Introduction to Positional Encoding in Transformer Models, Part 1 - MachineLearningMastery.com
Opens in a new window

cis.upenn.edu
Rotation - UPenn CIS
Opens in a new window

math.stackexchange.com
Is there a relationship between Rotors and the Rodrigues' rotation formula
Opens in a new window

en.wikipedia.org
Biquaternion - Wikipedia
Opens in a new window

en.wikipedia.org
Algebra of physical space - Wikipedia
Opens in a new window

en.wikipedia.org
Minkowski space - Wikipedia
Opens in a new window

en.wikibooks.org
Physics Using Geometric Algebra/Relativistic Classical Mechanics ...
Opens in a new window

math.stackexchange.com
Geometric algebra approach to Lorentz group representations - Math Stack Exchange
Opens in a new window

research.google
An Image is Worth 16x16 Words: Transformers for Image ...
Opens in a new window

openaccess.thecvf.com
Hyb-NeRF: A Multiresolution Hybrid Encoding for Neural Radiance Fields - CVF Open Access
Opens in a new window

codingscape.com
Andrej Karpathy's deep dive into LLMs video - Codingscape
Opens in a new window

arunprakash-a.github.io
Positional Encoding In Transformers - Arun's Blog
Opens in a new window

nn.labml.ai
Rotary Positional Embeddings (RoPE) - labml.ai
Opens in a new window

en.wikipedia.org
Lorentz transformation - Wikipedia
Opens in a new window

en.wikipedia.org
Plane-based geometric algebra - Wikipedia
Opens in a new window

math.stackexchange.com
Is there a reference for expressing biquaternion Lorentz transformations as a matrix?
Opens in a new window

openreview.net
STaR: Benchmarking Spatio-Temporal Reasoning for Systematic Generalization - OpenReview
Opens in a new window

arxiv.org
[2503.11495] V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning
Opens in a new window

arxiv.org
arxiv.org
Opens in a new window

x.com
Andrej Karpathy on X: "Jagged Intelligence The word I came up with ...
Opens in a new window

arxiv.org
arxiv.org
Opens in a new window

digitalcommons.chapman.edu
Spacetime Geometry of Acoustics and Electromagnetism - Chapman University Digital Commons
Opens in a new window

arxiv.org
ActionAtlas: A VideoQA Benchmark for Domain-specialized Action Recognition - arXiv
Opens in a new window

bohrium.dp.tech
When Spatial meets Temporal in Action Recognition - Bohrium
Opens in a new window

bohrium.dp.tech
Benchmarks for Physical Reasoning AI - Bohrium
Opens in a new window

arxiv.org
Benchmarks for Physical Reasoning AI - arXiv
Opens in a new window

openreview.net
PERCEIVER IO: A GENERAL ARCHITECTURE FOR STRUCTURED INPUTS & OUTPUTS - OpenReview
Opens in a new window

rohitbandaru.github.io
Transformer Design Guide (Part 2: Modern Architecture) - Rohit Bandaru
Opens in a new window

arxiv.org
FwNet-ECA: A Classification Model Enhancing Window Attention with Global Receptive Fields via Fourier Filtering Operations - arXiv
Opens in a new window

assemblyai.com
DeepMind's AlphaTensor Explained - AssemblyAI
Opens in a new window

openreview.net
AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE - OpenReview
Opens in a new window

alphanome.ai
The Temporal Tapestry: The Nexus of Language, Cognition, Time, and AI - Alphanome.AI
Opens in a new window

mdpi.com
Relativistic Option Pricing - MDPI
Opens in a new window

kaushikrajan.me
Reinforcement Learning Environments: From CartPole to MuJoCo - Kaushik Rajan
Opens in a new window

arxiv.org
RoboVerse: Towards a Unified Platform, Dataset and Benchmark for Scalable and Generalizable Robot Learning - arXiv
Opens in a new window

proceedings.neurips.cc
proceedings.neurips.cc
Opens in a new window

arxiv.org
arxiv.org





"""
MonSTER: Minkowski Space-Time Embedding Rotors

This module provides functions to compute MonSTER, a 4D generalization of
RoPE (Rotary Position Embedding).

---
### Key Steps for Computing the Rotor
---

Here are the key steps for computing the effective Lorentz rotor, R_eff_b,
for a given block b. This process begins with unit normalization to correctly
handle the physics without requiring numerical clamps.

1.  **Unit-Normalization (Lattice Units)** ⚛️
    - A spatial grid spacing is chosen, typically a power of two for numerical
      efficiency (e.g., s = 2^k m).
    - A corresponding time-step is defined as tau = s / c.
    - Physical coordinates are converted to dimensionless "lattice" coordinates
      where c=1:
        n_t = t / tau, n_x = x / s, n_y = y / s, n_z = z / s

2.  **Raw Integer Displacement** 📏
    - The displacement is calculated in these new lattice units.
        Delta_n = (Delta_n_t, ...) = (n_t_key - n_t_query, ...)

3.  **Frequency Scaling** 🌊
    - Block-specific inverse frequencies are applied to the temporal and
      spatial components.
        Delta_t_b = Delta_n_t * inv_freq_time_b
        Delta_s_b = (Delta_n_x, ...) * inv_freq_space_b

4.  **Compute Boost Rapidity** 🚀
    - Because c=1 in our units, the scaled time displacement directly
      becomes the boost rapidity.
        phi_b = Delta_t_b

5.  **Compute Spatial Rotation** 🔄
    - The rotation angle is the magnitude of the scaled spatial displacement:
      theta_b = ||Delta_s_b||.
    - The rotation axis is its direction: u_rot_b = Delta_s_b / ||Delta_s_b||.
      A default axis is used if the magnitude is near zero.

6.  **Build Block-wise Transforms** 🧱
    - **Spatial Rotation** M_rot_b: A 4x4 matrix representing the rotation.
    - **Lorentz Boost** M_boost_b: A 4x4 matrix for the boost with
      rapidity phi_b along the same axis.

7.  **Combine into the Effective Rotor** ✨
    - The final transformation is the composition of the boost and rotation.
        R_eff_b = M_boost_b @ M_rot_b
    - The operations commute because they share the same axis.

8.  **Modulate Attention** 🧠
    - For feature blocks Q_b and K_b, the rotor is inserted into the
      attention calculation:
        Attention Score ∝ Sum_b (Q_b^T * eta * R_eff_b * K_b)
"""

import jax
import jax.numpy as jnp

def get_monster_rotors(
    pos_q,
    pos_k,
    num_blocks: int,
    s: float = 1.0, # Spatial grid unit in meters, e.g., 2^k
    c: float = 299792458.0, # Speed of light in m/s
    base_time: float = 10000.,
    base_space: float = 10000.,
    epsilon: float = 1e-8,
    dtype=jnp.float32
):
    """Computes MonSTER rotors from query and key spacetime positions.

    Args:
        pos_q: Query positions (t, x, y, z). Shape (..., 4).
        pos_k: Key positions (t, x, y, z). Shape (..., 4).
        num_blocks: Number of frequency blocks (B) for multi-scale representation.
        s: The characteristic spatial grid spacing in physical units (e.g., meters).
           Choosing `s` as a power of two can be numerically advantageous.
        c: The speed of light in units consistent with `s` (e.g., m/s).
        base_time: The base for the geometric progression of temporal frequencies.
        base_space: The base for the geometric progression of spatial frequencies.
        epsilon: A small value for numerical stability when normalizing the rotation axis.
        dtype: The data type for all computations (e.g., jnp.float32).

    Returns:
        R_eff_blocks: A stack of 4x4 Lorentz transformation matrices, one for each
                      frequency block. The shape is (..., num_blocks, 4, 4).
    """
    pos_q = jnp.asarray(pos_q, dtype=dtype)
    pos_k = jnp.asarray(pos_k, dtype=dtype)

    # Step 1: Unit-Normalization (Lattice Units)
    tau = s / c

    # Step 2: Raw Integer Displacement
    delta_pos_raw = pos_k - pos_q
    delta_t_raw = delta_pos_raw[..., 0]
    delta_coords_raw = delta_pos_raw[..., 1:]

    delta_n_t = delta_t_raw / tau
    delta_n_coords = delta_coords_raw / s

    # Compute rotors using the normalized displacements
    return _compute_rotors_from_normalized_displacements(
        delta_n_t=delta_n_t,
        delta_n_coords=delta_n_coords,
        num_blocks=num_blocks,
        base_time=base_time,
        base_space=base_space,
        epsilon=epsilon,
        dtype=dtype
    )


def _compute_rotors_from_normalized_displacements(
    delta_n_t,
    delta_n_coords,
    num_blocks: int,
    base_time: float,
    base_space: float,
    epsilon: float,
    dtype
):
    """Internal helper to compute rotors from normalized displacements."""
    # Step 3: Frequency Scaling
    freqs = jnp.arange(num_blocks, dtype=dtype)
    inv_freq_time = 1.0 / (base_time ** (freqs / num_blocks))
    inv_freq_space = 1.0 / (base_space ** (freqs / num_blocks))

    delta_t_scaled = jnp.einsum('...,b->...b', delta_n_t, inv_freq_time)
    delta_s_scaled = jnp.einsum('...i,b->...bi', delta_n_coords, inv_freq_space)

    # Step 4: Compute Boost Rapidity
    phi_b = delta_t_scaled

    # Step 5: Compute Spatial Rotation
    theta_b = jnp.linalg.norm(delta_s_scaled, axis=-1, ord=2)

    default_spatial_axis = jnp.array([0., 0., 1.], dtype=dtype)
    axis_shape = delta_s_scaled.shape
    default_axis_bc = jnp.broadcast_to(default_spatial_axis, axis_shape)

    is_zero_spatial_delta = theta_b < epsilon
    axis_u_rot_b = jnp.where(
        is_zero_spatial_delta[..., None],
        default_axis_bc,
        delta_s_scaled / jnp.maximum(theta_b[..., None], epsilon)
    )

    # Step 6: Build Block-wise Transforms
    R3_b = _build_rotation_matrix(axis_u_rot_b, theta_b)

    pref_B_shape = R3_b.shape[:-2]
    M_rot_b = jnp.zeros((*pref_B_shape, 4, 4), dtype=dtype)
    M_rot_b = M_rot_b.at[..., 0, 0].set(1.0)
    M_rot_b = M_rot_b.at[..., 1:, 1:].set(R3_b)

    ch_b = jnp.cosh(phi_b)
    sh_b = jnp.sinh(phi_b)
    axis_u_boost_b = axis_u_rot_b

    M_boost_b = jnp.zeros((*pref_B_shape, 4, 4), dtype=dtype)
    M_boost_b = M_boost_b.at[..., 0, 0].set(ch_b)
    M_boost_b = M_boost_b.at[..., 0, 1:].set(-axis_u_boost_b * sh_b[..., None])
    M_boost_b = M_boost_b.at[..., 1:, 0].set(-axis_u_boost_b * sh_b[..., None])

    eye3 = jnp.eye(3, dtype=dtype)
    uuT_boost_b = jnp.einsum('...bi,...bj->...bij', axis_u_boost_b, axis_u_boost_b)
    ch_b_minus_1_exp = (ch_b - 1.0)[..., None, None]

    M_boost_b = M_boost_b.at[..., 1:, 1:].set(eye3 + ch_b_minus_1_exp * uuT_boost_b)

    # Step 7: Combine into the Effective Rotor
    R_eff_blocks = jnp.einsum("...bij,...bjk->...bik", M_boost_b, M_rot_b)

    return R_eff_blocks


def _build_rotation_matrix(axis, theta):
    """Rodrigues' formula for 3x3 rotation about 'axis' by angle 'theta'."""
    theta_exp = theta[..., None]
    cos_t = jnp.cos(theta_exp)
    sin_t = jnp.sin(theta_exp)

    uuT = jnp.einsum('...bi,...bj->...bij', axis, axis)

    zeros = jnp.zeros_like(axis[..., 0])
    u_cross = jnp.stack([
        zeros, -axis[..., 2], axis[..., 1],
        axis[..., 2], zeros, -axis[..., 0],
        -axis[..., 1], axis[..., 0], zeros
    ], axis=-1).reshape((*axis.shape[:-1], 3, 3))

    I3 = jnp.eye(3, dtype=axis.dtype)
    cos_t_exp_mat = cos_t[..., None]
    sin_t_exp_mat = sin_t[..., None]

    return (cos_t_exp_mat * I3 +
            (1 - cos_t_exp_mat) * uuT +
            sin_t_exp_mat * u_cross)
            