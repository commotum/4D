1. Basic Metadata

- Title: Rapid detection of porcine reproductive and respiratory syndrome viral nucleic acid in blood using a fluorimeter based PCR method
- Authors: M Spagnuolo-Weaver; I W Walker; S T Campbell; F McNeilly; B M Adair; G M Allan
- Year: 2000
- Venue: Vet Microbiol (Veterinary microbiology)

2. One-Sentence Contribution Summary

The paper reports a closed-tube, fluorimeter-based PCR assay to detect PRRSV RNA in porcine blood/serum samples (and differentiate American vs European strains) without gel electrophoresis, with sensitivity/specificity comparable to conventional RT-PCR (Abstract).

3. Tasks Evaluated

Task 1
- Task name: Detection of PRRSV RNA in serum samples and blood-impregnated filter disks from experimentally inoculated pigs
- Task type: Detection; Other (diagnosis)
- Dataset(s) used: Serum samples and blood impregnated filter disks (FDs) from experimentally inoculated pigs
- Domain: Porcine blood/serum samples (veterinary/virology)
- Evidence: "We describe the detection of PRRSV RNA in serum samples and in blood impregnated filter disks (FDs), obtained from experimentally inoculated pigs, using a closed-tube, fluorimeter-based PCR assay." (Abstract)

Task 2
- Task name: Differentiation of American and European PRRSV strains
- Task type: Classification (strain differentiation)
- Dataset(s) used: Not specified in the paper.
- Domain: PRRSV strains (veterinary virology)
- Evidence: "We also report a rapid fluorimeter based PCR method for differentiating American and European strains of PRRSV." (Abstract)

4. Domain and Modality Scope

- Single domain vs multiple: Single domain (porcine blood/serum samples from experimentally inoculated pigs).
  - Evidence: "serum samples and in blood impregnated filter disks (FDs), obtained from experimentally inoculated pigs" (Abstract)
- Multiple domains within same modality: Not specified in the paper.
- Multiple modalities: Not specified in the paper.
- Domain generalization or cross-domain transfer claim: Not claimed.

5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Detection of PRRSV RNA in serum/FDs | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Differentiation of American vs European PRRSV strains | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

6. Input and Representation Constraints

- Input type explicitly stated: "serum samples" and "blood impregnated filter disks (FDs)" from pigs. (Abstract)
- Fixed or variable input resolution: Not specified in the paper.
- Fixed patch size: Not specified in the paper.
- Fixed number of tokens: Not specified in the paper.
- Fixed dimensionality (e.g., strictly 2D): Not specified in the paper.
- Padding or resizing requirements: Not specified in the paper.

7. Context Window and Attention Structure

- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms to manage computational cost: Not specified in the paper.

8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Not specified in the paper.
- Where it is applied: Not specified in the paper.
- Fixed vs modified per task vs ablated: Not specified in the paper.

9. Positional Encoding as a Variable

- Treated as core variable or fixed assumption: Not specified in the paper.
- Multiple positional encodings compared: Not specified in the paper.
- Claims PE choice is not critical/secondary: Not specified in the paper.

10. Evidence of Constraint Masking

- Model size(s): Not specified in the paper.
- Dataset size(s): Not specified in the paper.
- Performance gains attributed to scaling model/data/architecture/training tricks: Not specified in the paper.

11. Architectural Workarounds

- Closed-tube, fluorimeter-based PCR assay to avoid gel electrophoresis and reduce contamination risk.
  - Evidence: "using a closed-tube, fluorimeter-based PCR assay." (Abstract)
  - Evidence: "The assay eliminates the use of gel electrophoresis" (Abstract)
  - Evidence: "present high risk of DNA carry-over contamination between the samples tested." (Abstract)

12. Explicit Limitations and Non-Claims

- Limitations or future work: Not specified in the paper.
- Explicit non-claims (e.g., open-world, unrestrained multi-task): Not specified in the paper.

13. Constraint Profile (Synthesis)

Constraint Profile:
- Domain scope: Single domain focused on porcine blood/serum samples from experimentally inoculated pigs.
- Task structure: Diagnostic detection plus strain differentiation within PRRSV.
- Representation rigidity: Inputs are biological samples (serum and blood-impregnated filter disks); no other representation details given.
- Model sharing vs specialization: Not specified; method-level description only.
- Role of positional encoding: Not specified.

14. Final Classification

Multi-task, single-domain.

Justification: The paper evaluates a diagnostic detection task on porcine blood/serum samples ("serum samples and ... blood impregnated filter disks (FDs)") and additionally reports a method for "differentiating American and European strains of PRRSV." Both tasks are within the same veterinary/virology domain, with no evidence of cross-domain evaluation (Abstract).
