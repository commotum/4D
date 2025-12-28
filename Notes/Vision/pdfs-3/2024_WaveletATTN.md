## 1. Basic Metadata
Title: Beyond Position: the emergence of wavelet-like properties in Transformers.
Evidence (Title page, p.1):
```
Beyond Position: the emergence of wavelet-like properties in Transformers
```

Authors: Valeria Ruscio, Umberto Nanni, Fabrizio Silvestri.
Evidence (Title page, p.1):
```
Valeria Ruscio, Umberto Nanni, Fabrizio Silvestri
```

Year: 2025.
Evidence (Title page, p.1):
```
4
Jun
2025
```

Venue: arXiv (cs.LG).
Evidence (Title page, p.1):
```
arXiv:2410.18067v4
[cs.LG]
```

## 2. One-Sentence Contribution Summary
The paper argues that RoPE-equipped Transformers develop emergent, wavelet-like multi-resolution processing that compensates for RoPE's positional-encoding limitations.
Evidence (Abstract, p.1):
```
This paper studies how Transformer models
with Rotary Position Embeddings (RoPE) de-
velop emergent, wavelet-like properties that
compensate for the positional encoding’s theo-
retical limitations.
```

## 3. Tasks Evaluated

Task 1: Frequency-domain analysis of attention distributions
Task type: Other (frequency-domain analysis of attention)
Dataset(s): Curated sample of 500 Wikipedia sequences
Domain: Natural language text
Evidence (Section 3.1, p.2):
```
To probe the spectral properties of attention distri-
butions, we employed a frequency-domain analysis
using the Discrete Fourier Transform (DFT), we
used the Hann window and zero padding.
```
Dataset evidence (Section 4 Implementation Details, p.4):
```
All models were evaluated on a curated sam-
ple of 500 sequences drawn from wikipedia. The
selected sequences varied in length to expose scale-
dependent behavior and stress-test the models’ posi-
tional encoding strategies under diverse conditions.
```

Task 2: Wavelet decomposition analysis
Task type: Other (wavelet decomposition analysis)
Dataset(s): Curated sample of 500 Wikipedia sequences
Domain: Natural language text
Evidence (Section 3.2, p.2):
```
While frequency-domain analysis captures global
spectral properties, it lacks explicit positional lo-
calization. To address this, we employed wavelet
decompositions using the Daubechies-2 (db2)
wavelet 1. Wavelets offer a time-frequency (or
position-frequency) representation that enables si-
multaneous assessment of spatial localization and
scale-dependent behaviors.
```
Dataset evidence (Section 4 Implementation Details, p.4):
```
All models were evaluated on a curated sam-
ple of 500 sequences drawn from wikipedia. The
selected sequences varied in length to expose scale-
dependent behavior and stress-test the models’ posi-
tional encoding strategies under diverse conditions.
```

Task 3: Uncertainty (positional vs spectral entropy) analysis
Task type: Other (entropy/uncertainty analysis)
Dataset(s): Curated sample of 500 Wikipedia sequences
Domain: Natural language text
Evidence (Section 3.3, p.3):
```
To evaluate the theoretical trade-off between posi-
tional precision and spectral organization, we com-
puted entropy measures for both the positional and
spectral domains.
```
Dataset evidence (Section 4 Implementation Details, p.4):
```
All models were evaluated on a curated sam-
ple of 500 sequences drawn from wikipedia. The
selected sequences varied in length to expose scale-
dependent behavior and stress-test the models’ posi-
tional encoding strategies under diverse conditions.
```

## 4. Domain and Modality Scope

Domain/modality scope:
- Single domain, single modality (English natural language text).
Evidence (Limitations, p.9):
```
our findings are based on a spe-
cific set of open-source language models trained
predominantly on English text.
```

Multiple domains within the same modality: Not specified in the paper.
Cross-domain transfer/domain generalization claim: Not claimed.
Evidence (Limitations, p.10):
```
Further research is
needed to determine if these principles generalize
across other modalities (e.g., vision, audio), data
types (e.g., code, multilingual text), and proprietary
architectures.
```

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Frequency-domain analysis of attention distributions | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We selected five pre-trained Transformer-based lan-" (Section 4, p.4) |
| Wavelet decomposition analysis | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We selected five pre-trained Transformer-based lan-" (Section 4, p.4) |
| Uncertainty (positional vs spectral entropy) analysis | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We selected five pre-trained Transformer-based lan-" (Section 4, p.4) |

## 6. Input and Representation Constraints

- Variable sequence length (no fixed length stated).
Evidence (Section 4 Implementation Details, p.4):
```
selected sequences varied in length to expose scale-
dependent behavior and stress-test the models’ posi-
tional encoding strategies under diverse conditions.
```
- Decomposition level tied to shortest sequence length.
Evidence (Section 3.2, p.2):
```
We selected a maximum decomposi-
tion level suitable for the shortest sequence length
to ensure consistent comparisons across models
and scales.
```
- Fixed/variable input resolution, fixed patch size, fixed number of tokens, fixed dimensionality, padding/resizing: Not specified in the paper.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified in the paper.
- Sequence length fixed or variable: Variable.
Evidence (Section 4 Implementation Details, p.4):
```
selected sequences varied in length to expose scale-
dependent behavior and stress-test the models’ posi-
tional encoding strategies under diverse conditions.
```
- Attention type (global/windowed/hierarchical/sparse): Not explicitly specified. The paper reports emergent local/global specialization.
Evidence (Section 5 Experiments and Analysis, p.4):
```
attention heads specialize
into either local or global processors, evidenced by
the pronounced vertical striping in visualizations of
local-to-global attention ratios.
```
- Mechanisms introduced to manage computational cost (windowing, pooling, token pruning, etc.): Not specified in the paper.

## 8. Positional Encoding (Critical Section)

Positional encoding mechanism: RoPE (rotary position embeddings with rotation matrices).
Evidence (Section 6, p.7):
```
Rotary Position Embeddings (RoPE) encode po-
sitional information through position-dependent
rotation matrices defined over the complex plane.
```

Where applied: The paper states that the embedding is rotated; no layer-wide placement details are given.
Evidence (Section 6, p.7):
```
At position m, the embedding applies a rotation
```

Fixed vs modified per task / ablated: RoPE is the main focus, and alternative PEs are compared in ablation.
Evidence (Section 5.3 Ablation Study, p.5):
```
we conducted an ablation
study comparing it against three alternative posi-
tional encoding schemes: T5’s relative position
biases, BERT’s absolute positional embeddings,
and a GPT-2 model with no explicit positional en-
coding (No PE).
```

## 9. Positional Encoding as a Variable

- Core research variable: Yes (explicit ablation across PE types).
Evidence (Section 5.3 Ablation Study, p.5):
```
we conducted an ablation
study comparing it against three alternative posi-
tional encoding schemes: T5’s relative position
biases, BERT’s absolute positional embeddings,
and a GPT-2 model with no explicit positional en-
coding (No PE).
```
- Multiple positional encodings compared: Yes (RoPE vs T5 vs BERT vs No PE).
- PE choice claimed as "not critical": No. The paper highlights RoPE as distinctive.
Evidence (Abstract, p.1):
```
We demonstrate that the emergence of robust,
wavelet-like, scale-invariant properties is a
distinctive feature of the RoPE architecture
compared to other common position encoding
schemes.
```

## 10. Evidence of Constraint Masking

Model size(s) / scaling evidence:
Evidence (Section 4 Implementation Details, p.4):
```
We selected five pre-trained Transformer-based lan-
guage models that vary in size, architecture, and
training regimen to ensure the generality of our
findings. Specifically, we analyzed Gemma 2 2B,
Pythia 2.8B and 12B, LLaMA-3-2 1B, Mistral 7B,
and Qwen 2.5 5B. These models encompass a wide
parameter range (1B–12B), capturing different rep-
resentational capacities and training protocols.
```

Dataset size(s):
Evidence (Section 4 Implementation Details, p.4):
```
All models were evaluated on a curated sam-
ple of 500 sequences drawn from wikipedia.
```

Compensation/constraint masking (architectural compensation for RoPE limitations):
Evidence (Abstract, p.1):
```
This paper studies how Transformer models
with Rotary Position Embeddings (RoPE) de-
velop emergent, wavelet-like properties that
compensate for the positional encoding’s theo-
retical limitations.
```

Additional evidence of compensatory behavior:
Evidence (Section 5 Experiments and Analysis, p.4):
```
Our analysis shows that Transformer models with
RoPE spontaneously develop a sophisticated, multi-
resolution processing strategy, similar to wavelet
decomposition, to overcome the theoretical limita-
tions of their position embeddings.
```

Attribution of gains:
- Architectural/positional-encoding mechanism emphasized (RoPE is distinctive).
Evidence (Abstract, p.1):
```
We demonstrate that the emergence of robust,
wavelet-like, scale-invariant properties is a
distinctive feature of the RoPE architecture
compared to other common position encoding
schemes.
```
- Scaling data or training tricks: Not specified in the paper.

## 11. Architectural Workarounds

- No explicit architectural modifications (e.g., windowed attention, token pooling/merging) are introduced; the paper reports emergent specialization patterns.
Evidence (Section 5 Experiments and Analysis, p.4):
```
attention heads specialize
into either local or global processors, evidenced by
the pronounced vertical striping in visualizations of
local-to-global attention ratios.
```

## 12. Explicit Limitations and Non-Claims

Inference-time analysis focus:
Evidence (Limitations, p.9):
```
Our primary analysis, like much of the field, ex-
amines model properties at inference time after
training is complete.
```

Limited to open-source English language models:
Evidence (Limitations, p.9):
```
our findings are based on a spe-
cific set of open-source language models trained
predominantly on English text.
```

No claim of generalization across modalities/data types:
Evidence (Limitations, p.10):
```
Further research is
needed to determine if these principles generalize
across other modalities (e.g., vision, audio), data
types (e.g., code, multilingual text), and proprietary
architectures.
```

## 13. Constraint Profile (Synthesis)

- Domain scope: Single modality/domain (English language text); evaluation uses Wikipedia sequences; no cross-domain claims.
- Task structure: Analysis tasks only (frequency, wavelet, entropy analyses) on pre-trained LMs; no downstream task evaluation.
- Representation rigidity: Variable-length sequences; decomposition level tied to shortest sequence length; other constraints not specified.
- Model sharing vs specialization: Uses pre-trained models; no fine-tuning described; specialization appears at the head level (local/global).
- Positional encoding role: RoPE is central and explicitly compared against other PE types; PE is a core variable.

## 14. Final Classification

Classification: Single-task, single-domain.
Justification: The evaluation is confined to analysis of attention patterns on a single text domain (Wikipedia sequences) and is based on English-language models, with no multi-domain or cross-modal evaluation claimed. The paper explicitly notes its scope is limited to English-language models and calls for future work to test other modalities.
