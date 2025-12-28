### 1. Basic Metadata
- Title: "Spherical Position Encoding for Transformers" (p.2).
- Authors: "Eren Unlu" (p.1).
- Year: 2023 ("arXiv:2310.04454v1 [cs.CL] 4 Oct 2023", p.1).
- Venue (conference/journal/arXiv): arXiv preprint ("arXiv:2310.04454v1 [cs.CL] 4 Oct 2023", p.1).

### 2. One-Sentence Contribution Summary
This paper introduces geotokens and proposes a RoPE-based position encoding adjusted to spherical coordinates so transformers can encode geographical location relations rather than sequential order ("In this paper, we introduce the notion of "geotokens" which are input elements for transformer architectures, each representing an information related to a geological location." Abstract, p.1; "we formulate a position encoding mechanism based on RoPE architecture which is adjusted for spherical coordinates." Abstract, p.1).

### 3. Tasks Evaluated
Not specified in the paper.

### 4. Domain and Modality Scope
- Evaluation scope (single domain vs multi-domain vs multi-modal): Not specified in the paper.
- Domain described: "A geotoken encapsulates both the semantic meaning and the spatial information of a geographical entity." (Section 2, p.2).
- Spatial vs sequential framing: "the geotokens are not sequence dependent intuitively but are based on spatial relationships." (Section 2, p.2).
- Modality inputs (if any): "It is assumed that each data point has a pre-embedded vector retaining valuable information about the location itself, which may have been encoded by any type of neural architecture or mechanism, such as a natural language model processing its verbal description or a CNN extracting its visual features." (Section 2, p.2).
- Domain generalization or cross-domain transfer: Not claimed.

### 5. Model Sharing Across Tasks
No evaluated tasks are specified, so model sharing across tasks is not described.

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Not specified (no evaluated tasks) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

### 6. Input and Representation Constraints
- Geotokens as inputs: "In this paper, we introduce the notion of "geotokens" which are input elements for transformer architectures, each representing an information related to a geological location." (Abstract, p.1).
- Spatial (not sequential) relationships: "the geotokens are not sequence dependent intuitively but are based on spatial relationships." (Section 2, p.2).
- Punctual locations with latitude/longitude: "For the sake of simplicity only punctual locations are considered represented by a latitude and longitude." (Section 2, p.2).
- Pre-embedded vectors: "It is assumed that each data point has a pre-embedded vector retaining valuable information about the location itself, which may have been encoded by any type of neural architecture or mechanism, such as a natural language model processing its verbal description or a CNN extracting its visual features." (Section 2, p.2).
- Perfect-sphere assumption: "For the sake of simplicity, without loss of generalization and omiting the fractional errors let us assume that globe is a perfect sphere with constant radius R." (Section 5, p.3).
- Rotation-matrix encoding with embedding dimension multiple of 3: "we propose to encode the rotational position encoding matrix as follows, assuming a multiple of 3 :" (Section 5, p.4); "the embedding dimension is a multiple of three due to natural requirements, however this choice might be unconvenient as many embedders of different modalities might not adhere to this constraint." (Section 5, p.4).
- Padding workaround out of scope: "The possible circumvention to this issue is out of scope of this paper, such as possibly adding padding indices." (Section 5, p.4).
- Fixed/variable input resolution, fixed patch size, fixed number of tokens, padding/resizing requirements (beyond the above): Not specified in the paper.

### 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed vs variable sequence length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Cost management mechanisms (windowing, pooling, pruning): Not specified in the paper.

### 8. Positional Encoding (Critical Section)
- Mechanism: RoPE-based rotational encoding adapted to spherical coordinates: "we formulate a position encoding mechanism based on RoPE architecture which is adjusted for spherical coordinates." (Abstract, p.1); "we propose to extend the RoPE method in spherical coordinates." (Section 5, p.3).
- Where applied: "position encoding as a function g applied on inner products of query and key vectors :" (Section 4, p.3).
- Application location (input only vs every layer vs attention bias): Not specified in the paper.
- Fixed across experiments vs modified per task vs ablated: Not specified in the paper.

### 9. Positional Encoding as a Variable
- Core research variable: The paper proposes a new PE mechanism ("we formulate a position encoding mechanism based on RoPE architecture which is adjusted for spherical coordinates." Abstract, p.1; "we propose to extend the RoPE method in spherical coordinates." Section 5, p.3).
- Multiple positional encodings compared: Not specified in the paper.
- Claims that PE choice is not critical: Not specified in the paper.

### 10. Evidence of Constraint Masking
- Model sizes: Not specified in the paper.
- Dataset sizes: Not specified in the paper.
- Performance gains attributed to scaling model/data or training tricks: Not specified in the paper.

### 11. Architectural Workarounds
- Complexity/scale workarounds (windowed attention, hierarchy, pooling/merging, pruning): Not specified in the paper.
- The architectural change described is the spherical-coordinate PE: "we propose to extend the RoPE method in spherical coordinates." (Section 5, p.3).

### 12. Explicit Limitations and Non-Claims
- Punctual location simplification: "For the sake of simplicity only punctual locations are considered represented by a latitude and longitude." (Section 2, p.2).
- Embedding dimension constraint and its inconvenience: "the embedding dimension is a multiple of three due to natural requirements, however this choice might be unconvenient as many embedders of different modalities might not adhere to this constraint." (Section 5, p.4).
- Out-of-scope workaround: "The possible circumvention to this issue is out of scope of this paper, such as possibly adding padding indices." (Section 5, p.4).
- Scaling challenges out of scope: "In addition, further possible challenges such as proper scaling are kept out of scope as well, where in case one training the architecture with limited geolocations, rather than whole globe." (Section 5, p.4).
- Explicit non-claims about open-world/multi-task learning: Not specified in the paper.

### 13. Constraint Profile (Synthesis)
- Domain scope: Geographical entities/geotokens; no empirical multi-domain evaluation reported.
- Task structure: No tasks or datasets are specified; evaluation setup is absent.
- Representation rigidity: Geotokens with latitude/longitude, pre-embedded vectors, perfect-sphere assumption; embedding dimension must be a multiple of three.
- Model sharing vs specialization: Not specified (no task setup described).
- Role of positional encoding: Central contribution; spherical-coordinate RoPE is the main architectural change.

### 14. Final Classification
Single-task, single-domain. The paper frames inputs as geotokens representing geographical entities and focuses on a spherical-coordinate PE mechanism ("A geotoken encapsulates both the semantic meaning and the spatial information of a geographical entity." Section 2, p.2; "we formulate a position encoding mechanism based on RoPE architecture which is adjusted for spherical coordinates." Abstract, p.1). It does not describe multiple tasks or domains or any evaluation setup, so there is no evidence of multi-task or multi-domain learning.
