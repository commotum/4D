## 1. Basic Metadata
- Title: Core knowledge
- Authors: Elizabeth S. Spelke and Katherine D. Kinzler
- Year: 2007
- Venue (conference/journal/arXiv): Developmental Science (journal), 10:1, pp. 89–96

Evidence:
- "Developmental Science 10:1 (2007), pp 89– 96" (p. 89).
- "Core knowledge" (p. 89).
- "Elizabeth S. Spelke and Katherine D. Kinzler" (p. 89).

---

## 2. One-Sentence Contribution Summary
The paper argues that human cognition is grounded in a small set of core knowledge systems (objects, actions, number, space, and possibly social partners) and synthesizes cross-species and cross-cultural evidence about their properties and limits.

---

## 3. Tasks Evaluated
Not specified in the paper. The article is a conceptual review and does not report new experimental task evaluations or datasets.

Evidence:
- "Converging research on human infants, non-human primates, children and adults in diverse cultures can aid both understanding of these systems and attempts to overcome their limits." (Abstract, p. 89).
- "Studies of human infants and non-human animals, focused on the ontogenetic and phylogenetic origins of knowledge, provide evidence for four core knowledge systems (Spelke, 2004)." (Introduction, p. 89).

---

## 4. Domain and Modality Scope
- Domain scope: Multiple domains. Evidence: "Human cognition is founded, in part, on four systems for representing objects, actions, number, and space. It may be based, as well, on a fifth system for representing social partners." (Abstract, p. 89).
- Modality scope: Multiple modalities are explicitly discussed for number representations. Evidence: "number representations are abstract: they apply to diverse entities encountered through multiple sensory modalities, including arrays of objects, sequences of sounds, and perceived or produced sequences of actions." (p. 91).
- Domain generalization or cross-domain transfer: Not claimed.

---

## 5. Model Sharing Across Tasks
Not applicable; the paper does not describe model training or task-specific model variants.

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| N/A (no task evaluations reported) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

---

## 6. Input and Representation Constraints
- Fixed or variable input resolution: Not specified in the paper.
- Fixed patch size: Not specified in the paper.
- Fixed number of tokens: Not specified in the paper.
- Fixed dimensionality (e.g., strictly 2D): Not specified in the paper.
- Padding/resizing requirements: Not specified in the paper.

Explicit representational constraints mentioned:
- "It centers on the spatio-temporal principles of cohesion (objects move as connected and bounded wholes), continuity (objects move on connected, unobstructed paths), and contact (objects do not interact at a distance)" (p. 89).
- "infants are able to represent only a small number of objects at a time (about three; Feigenson & Carey, 2003)." (p. 90).
- "monkeys’ object representations obey the continuity and contact constraints (Santos, 2004) and show a set size limit (of four; Hauser & Carey, 2003)." (p. 90).
- "number representations are abstract: they apply to diverse entities encountered through multiple sensory modalities, including arrays of objects, sequences of sounds, and perceived or produced sequences of actions." (p. 91).
- "The last system of core knowledge captures the geometry of the environment: the distance, angle, and sense relations among extended surfaces in the surrounding layout." (p. 92).

---

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms to manage computational cost (e.g., windowing, pooling, token pruning): Not specified in the paper.

---

## 8. Positional Encoding (Critical Section)
Not specified in the paper.

---

## 9. Positional Encoding as a Variable
Not specified in the paper.

---

## 10. Evidence of Constraint Masking
Not specified in the paper (no model sizes, dataset sizes, or scaling claims reported).

---

## 11. Architectural Workarounds
Not specified in the paper.

---

## 12. Explicit Limitations and Non-Claims
Limitations:
- "Nevertheless, core systems of representation also can lead humans into cognitive errors and maladaptive actions." (p. 93).
- "At the smallest and largest scales that science can probe, objects are not cohesive or continuous, and space is not Euclidean or three-dimensional." (p. 93).
- "The gaps and inaccuracies in core representations cause problems for adults and children alike, who are prone to errors in reasoning about properties of object mechanics, non-Euclidean geometry, or numbers that violate the principles of core knowledge" (p. 93).

Explicit non-claims:
- "humans are endowed neither with a single, general-purpose learning system nor with myriad special-purpose systems and predispositions." (p. 89).
- "This research suggests that the human mind is not a single, general-purpose device that adapts itself to whatever structures and challenges the environment affords." (p. 91).

Future work statements: Not specified in the paper.

---

## 13. Constraint Profile (Synthesis)
- Domain scope: Multiple domains (objects, actions, number, space, social partners) across populations; the paper frames these as distinct core systems.
- Task structure: No explicit task evaluations or datasets; the paper is a synthesis of prior studies.
- Representation rigidity: Explicit constraints include spatio-temporal principles and set-size limits in object representations, plus abstract, cross-modal number representations.
- Model sharing vs specialization: Not specified; no model training described.
- Role of positional encoding: Not specified.

---

## 14. Final Classification
Multi-task, multi-domain (constrained). The paper explicitly frames cognition around multiple core systems for different domains—"objects, actions, number, and space" plus a possible "fifth system for representing social partners"—but confines discussion to these core domains rather than open-ended tasks. The evidence base spans multiple populations and modalities, yet it remains bounded to these specific representational systems. (Abstract, p. 89)
