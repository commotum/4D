## 1. Basic Metadata
- Title: WALRUS : A CROSS-DOMAIN FOUNDATION MODEL FOR CONTINUUM DYNAMICS. Evidence: "WALRUS : A C ROSS - DOMAIN F OUNDATION M ODEL FOR C ONTINUUM DYNAMICS" (p.1, title).
- Authors: Michael McCabe; Payel Mukhopadhyay; Tanya Marwah; Bruno Régaldo-Saint Blancard; François Rozet; Cristiana Diaconu; Lucas Meyer; Kaze W. K. Wong; Hadi Sotoudeh; Alberto Bietti; Irina Espejo; Rio Fear; Siavash Golkar; Tom Hehir; Keiya Hirashima; Geraud Krawezik; François Lanusse; Rudy Morel; Ruben Ohana; Liam Parker; Mariel Pettee; Jeff Shen; Kyunghyun Cho; Miles Cranmer; Shirley Ho; The Polymathic AI Collaboration. Evidence: "Michael McCabe∗,1,2 , Payel Mukhopadhyay 3 , Tanya Marwah 1 , Bruno Régaldo-Saint Blancard 1 , François Rozet 1,4 , Cristiana Diaconu 3 , Lucas Meyer 1 , Kaze W. K. Wong, Hadi Sotoudeh 3 , Alberto Bietti 1 , Irina Espejo 1,2 , Rio Fear 3 , Siavash Golkar 1,2 , Tom Hehir 3 , Keiya Hirashima 1,5 , Geraud Krawezik 1 , François Lanusse 1,4 , Rudy Morel 1 , Ruben Ohana 1 , Liam Parker 1 , Mariel Pettee 8 , Jeff Shen 9 , Kyunghyun Cho 2,10,11 , Miles Cranmer 3 , Shirley Ho 1,2,9" and "The Polymathic AI Collaboration" (p.1, author line).
- Year: 2025. Evidence: "arXiv:2511.15684v1 [cs.LG] 19 Nov 2025" (p.1).
- Venue: arXiv (preprint). Evidence: "arXiv:2511.15684v1 [cs.LG] 19 Nov 2025" (p.1).

## 2. One-Sentence Contribution Summary
The paper introduces Walrus, a transformer-based foundation model for data-driven emulation of fluid-like continuum dynamics across diverse physical scenarios, using stabilization and compute-adaptive tokenization to improve long-horizon prediction.
- Evidence: "Using these tools, we develop Walrus, a transformer-based foundation model developed primarily for fluid-like continuum dynamics." (Abstract)
- Evidence: "we incorporate new approaches to mitigate these obstacles, including a harmonic-analysis–based stabilization method, load-balanced distributed 2D-3D training strategies, and compute-adaptive tokenization." (Abstract)

## 3. Tasks Evaluated
General task framing (applies to all tasks below): "Walrus takes as input a short sequence of snapshots and predicts the next step in the sequence." (Figure 1 caption)

### Pretraining datasets (Table 5)
- acoustic scattering discontinuous
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Acoustics and wave propagation
  - Evidence: "acoustic scattering discontinuous     Discontinuous          The Well      (x, y)           256 × 256              102     2 000" (Table 5, Data)
  - Evidence: "Acoustics and wave propagation" (Table 1)
- acoustic scattering inclusions
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Acoustics and wave propagation
  - Evidence: "acoustic scattering inclusions        Inclusions             The Well      (x, y)           256 × 256              102     4 000" (Table 5, Data)
  - Evidence: "Acoustics and wave propagation" (Table 1)
- acoustic scattering maze
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Acoustics and wave propagation
  - Evidence: "acoustic scattering maze              Maze                   The Well      (x, y)           256 × 256              202     2 000" (Table 5, Data)
  - Evidence: "Acoustics and wave propagation" (Table 1)
- active matter
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Biological and chemical behavior
  - Evidence: "active matter                         Active Matter          The Well      (x, y)           256 × 256               81       360" (Table 5, Data)
  - Evidence: "Biological and chemical behavior" (Table 1)
- euler multiquadrants periodicBC
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Inviscid fluids
  - Evidence: "euler multiquadrants periodicBC       MultiQuadrantsP        The Well      (x, y)           512 × 512              101     5 000" (Table 5, Data)
  - Evidence: "Inviscid fluids" (Table 1)
- euler multiquadrants openBC
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Inviscid fluids
  - Evidence: "euler multiquadrants openBC           MultiQuadrantsO        The Well      (x, y)           512 × 512              101     5 000" (Table 5, Data)
  - Evidence: "Inviscid fluids" (Table 1)
- gray scott reaction diffusion
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Biological and chemical behavior
  - Evidence: "gray scott reaction diffusion         Gray-Scott             The Well      (x, y)           128 × 128            1 001     1 200" (Table 5, Data)
  - Evidence: "Biological and chemical behavior" (Table 1)
- helmholtz staircase
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Acoustics and wave propagation
  - Evidence: "helmholtz staircase                   Staircase              The Well      (x, y)         1 024 × 256               50       512" (Table 5, Data)
  - Evidence: "Acoustics and wave propagation" (Table 1)
- MHD (3D)
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Plasmas
  - Evidence: "MHD                                   MHD (3D)               The Well    (x, y, z)                643              100       100" (Table 5, Data)
  - Evidence: "Plasmas" (Table 1)
- planetswe (PlanetSWE)
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Astrophysical and geoscience applications
  - Evidence: "planetswe                             PlanetSWE              The Well      (θ, ϕ)           256 × 512            1 008       120" (Table 5, Data)
  - Evidence: "Astrophysical and geoscience applications" (Table 1)
- rayleigh benard
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Viscous fluids
  - Evidence: "rayleigh benard                       Rayleigh-Benard        The Well      (x, y)           512 × 128              200     1 750" (Table 5, Data)
  - Evidence: "Viscous fluids" (Table 1)
- rayleigh taylor instability
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Viscous fluids
  - Evidence: "rayleigh taylor instability           RT Instability (3D)    The Well    (x, y, z)    128 × 128 × 128              120        45" (Table 5, Data)
  - Evidence: "Viscous fluids" (Table 1)
- shear flow
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Viscous fluids
  - Evidence: "shear flow                            Shear Flow             The Well      (x, y)           256 × 512              200     1 120" (Table 5, Data)
  - Evidence: "Viscous fluids" (Table 1)
- supernova explosion
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Astrophysical and geoscience applications
  - Evidence: "supernova explosion                   Supernova (3D)         The Well    (x, y, z)               1283               59     1 000" (Table 5, Data)
  - Evidence: "Astrophysical and geoscience applications" (Table 1)
- turbulence gravity cooling
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Astrophysical and geoscience applications
  - Evidence: "turbulence gravity cooling            TGC (3D)               The Well    (x, y, z)       64 × 64 × 64               50     2 700" (Table 5, Data)
  - Evidence: "Astrophysical and geoscience applications" (Table 1)
- turbulent radiative layer 2D
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Inviscid fluids
  - Evidence: "turbulent radiative layer 2D          TRL (2D)               The Well      (x, y)           128 × 384              101        90" (Table 5, Data)
  - Evidence: "Inviscid fluids" (Table 1)
- turbulent radiative layer 3D
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Inviscid fluids
  - Evidence: "turbulent radiative layer 3D          TRL (3D)               The Well    (x, y, z)    128 × 128 × 256              101        90" (Table 5, Data)
  - Evidence: "Inviscid fluids" (Table 1)
- viscoelastic instability
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Non-newtonian fluids
  - Evidence: "viscoelastic instability              Viscoelastics          The Well      (x, y)           512 × 512          variable      260" (Table 5, Data)
  - Evidence: "Non-newtonian fluids" (Table 1)
- FPOHarmonics (FBHarmonics)
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Viscous fluids
  - Evidence: "FPOHarmonics                          FBHarmonics            The Well      (x, y)           512 × 128              242     400∗" (Table 5, Data)
  - Evidence: "Viscous fluids" (Table 1)

### Downstream / evaluation datasets (Table 6)
- FPOSkelenton
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: Flowbench
  - Domain: Not specified in the paper.
  - Evidence: "FPOSkelenton                         Flowbench            (x, y)           512 × 128            242         262" (Table 6, Data)
- PoolBoil Subcooled
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: BubbleML 2.0
  - Domain: Not specified in the paper.
  - Evidence: "PoolBoil Subcooled                BubbleML 2.0            (x, y)           512 × 512           2001          44" (Table 6, Data)
- Conditioned Incompressible NS
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: PDEArena
  - Domain: Not specified in the paper.
  - Evidence: "Conditioned Incompressible NS        PDEArena             (x, y)           128 × 128             56        6816" (Table 6, Data)
- CE-RM
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: PDEGym
  - Domain: Not specified in the paper.
  - Evidence: "CE-RM                                 PDEGym              (x, y)           128 × 128             21        1260" (Table 6, Data)
- CNS Turbulent
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: PDEBench
  - Domain: Not specified in the paper.
  - Evidence: "CNS Turbulent                        PDEBench           (x, y, z)                643              21        600" (Table 6, Data)
- CNS Random
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: PDEBench
  - Domain: Not specified in the paper.
  - Evidence: "CNS Random                           PDEBench           (x, y, z)               1283              21        100" (Table 6, Data)
- post neutron star merger
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Not specified in the paper.
  - Evidence: "post neutron star merger              The Well      (log r, θ, ϕ)     192 × 128 × 66             181          8" (Table 6, Data)
- convective envelope rsg
  - Task type: Other (spatiotemporal prediction / emulation of continuum dynamics)
  - Dataset(s) used: The Well
  - Domain: Not specified in the paper.
  - Evidence: "convective envelope rsg               The Well          (r, θ, ϕ)    256 × 128 × 256             100         29" (Table 6, Data)

## 4. Domain and Modality Scope
- Single domain vs multiple domains (same modality): Multiple domains within the same modality (continuum dynamics / physical simulation) are evaluated. Evidence: "Walrus is pretrained on nineteen diverse scenarios spanning astrophysics, geoscience, rheology, plasma physics, acoustics, and classical fluids." (Abstract)
- Multiple modalities: Not specified in the paper.
- Domain generalization or cross-domain transfer: The paper explicitly frames and evaluates Walrus as a cross-domain foundation model. Evidence: "Is Walrus truly a cross-domain foundation model?" (Section 5, Experiments); "We evaluate the claim that Walrus is a cross-domain foundation model" (Section 5.2, Cross-Domain Analysis).

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| acoustic scattering discontinuous | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| acoustic scattering inclusions | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| acoustic scattering maze | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| active matter | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| euler multiquadrants periodicBC | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| euler multiquadrants openBC | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| gray scott reaction diffusion | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| helmholtz staircase | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| MHD (3D) | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| planetswe (PlanetSWE) | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| rayleigh benard | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| rayleigh taylor instability | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| shear flow | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| supernova explosion | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| turbulence gravity cooling | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| turbulent radiative layer 2D | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| turbulent radiative layer 3D | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| viscoelastic instability | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| FPOHarmonics (FBHarmonics) | Yes (joint pretraining across datasets; shared encoder/decoder by dimensionality). | Yes (finetuned per dataset in cross-domain analysis). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1); "We specialize Walrus to each dataset through an additional 500K samples of finetuning." (Section 5.2) |
| FPOSkelenton | Yes (pretrained model shared across tasks). | Yes (finetuned per task). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "For all finetuning tasks, Walrus was trained for an additional 50 pseudo-epochs (500K samples) on the new dataset exclusively" (Appendix B.2) |
| PoolBoil Subcooled | Yes (pretrained model shared across tasks). | Yes (finetuned per task). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "For all finetuning tasks, Walrus was trained for an additional 50 pseudo-epochs (500K samples) on the new dataset exclusively" (Appendix B.2) |
| Conditioned Incompressible NS | Yes (pretrained model shared across tasks). | Yes (finetuned per task). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "For all finetuning tasks, Walrus was trained for an additional 50 pseudo-epochs (500K samples) on the new dataset exclusively" (Appendix B.2) |
| CE-RM | Yes (pretrained model shared across tasks). | Yes (finetuned per task). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "For all finetuning tasks, Walrus was trained for an additional 50 pseudo-epochs (500K samples) on the new dataset exclusively" (Appendix B.2) |
| CNS Turbulent | Yes (pretrained model shared across tasks). | Yes (finetuned per task). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "For all finetuning tasks, Walrus was trained for an additional 50 pseudo-epochs (500K samples) on the new dataset exclusively" (Appendix B.2) |
| CNS Random | Yes (pretrained model shared across tasks). | Yes (finetuned per task). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "For all finetuning tasks, Walrus was trained for an additional 50 pseudo-epochs (500K samples) on the new dataset exclusively" (Appendix B.2) |
| post neutron star merger | Yes (pretrained model shared across tasks). | Yes (finetuned per task). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "For all finetuning tasks, Walrus was trained for an additional 50 pseudo-epochs (500K samples) on the new dataset exclusively" (Appendix B.2) |
| convective envelope rsg | Yes (pretrained model shared across tasks). | Yes (finetuned per task). | Not specified in the paper. | "Walrus was pretrained using the 19 datasets in Table 5." (Appendix B.1); "For all finetuning tasks, Walrus was trained for an additional 50 pseudo-epochs (500K samples) on the new dataset exclusively" (Appendix B.2) |

## 6. Input and Representation Constraints
- Variable input resolution handled via adaptive compression: "CSM allows us to alter the stride of convolutions performing downsampling options letting us choose a spatial compression level appropriate to the task." (Section 3.1, Architecture)
- Fixed number of tokens per axis during pretraining (adaptive downsampling to match token count): "During pretraining, to maximize device utilization, we choose a fixed number of tokens per axis and adjust the downsampling factor accordingly such that data of the same dimensionality produces roughly the same number of tokens per frame across datasets." (Section 3.1, Architecture)
- Token count targets per dimension: "For 2D data, we select this size as 32 per dimension without padding. For 3D, this is set to 16 without padding." (Section 4.2)
- Fixed dimensionality (2D/3D only in this study): "Since this study only features 2D and 3D data" (Section 3.1, Shared Encoder-Decoder); "This includes both 2D and 3D data" (Introduction)
- 2D inputs embedded in 3D with padding: "treating 2D data as a plane randomly embedded in 3D space." (Introduction contributions); "The data is first projected into 3D by appending a singleton dimension and zero-padding the tensor-valued fields." (Section 4.1, Dimension padding)
- Padding / boundary handling constraints: "For non-periodic boundaries, padding is implemented with additional channels containing binary masks for each topological boundary type." (Section C.2, Boundary Handling)
- Patch/stride constraints (kernel/stride choices fixed to match internal resolution): "we choose the kernel sizes (p1 , p2 ) and strides (s1 , s2 ) used in the convolutional down-sampling layers to ensure the internal model resolution lines up with the settings in Table 2." (Appendix A.2)
- Temporal sampling constraint: "We sample the time stride from 1-5 during pretraining." (Section 4.1, Variable time striding)

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Sequence length fixed or variable:
  - Fixed time history lengths are specified: "Time History (2D)                                 6" and "Time History (3D)                                 3" (Table 2)
  - Temporal stride is variable during pretraining: "We sample the time stride from 1-5 during pretraining." (Section 4.1)
- Attention type:
  - Space-time factorized attention: "Walrus employs a space-time factorized transformer architecture ... where alternating operations within a block attend along the space and time axes of space-time tensor-structured data." (Section 3.1)
  - Spatial attention uses axial RoPE: "Spatial processing uses the parallelized attention ... using axial RoPE ... for position encoding." (Section 3.1)
  - Temporal attention is causal with relative position encoding: "Along the time axis, Walrus uses causal attention with T5-style relative position encoding." (Section 3.1)
  - Table summary: "Space block                              Parallel Attention" and "Time block                                Causal Attention" (Table 2)
- Mechanisms to manage computational cost:
  - Adaptive-compute compression: "Convolutional Stride Modulation ... to natively handle data at varying resolutions by adapting the level of downsampling/upsampling" (Section 3.1)
  - Adaptive-compute tokenization: "Adaptive-compute Tokenization: integrating recent adaptive-compute tokenization methods to allocate compute dynamically based on resolution or problem complexity." (Introduction contributions)
  - Topology-aware sampling for throughput: "Topology-aware Sampling: increasing training throughput by 262% by tying sampling scheme across ranks to minimize task variance within sharding groups." (Introduction contributions)

## 8. Positional Encoding (Critical Section)
- Mechanism(s) used:
  - Spatial: axial RoPE (rotary/relative) in spatial attention. Evidence: "Spatial processing uses the parallelized attention ... using axial RoPE ... for position encoding." (Section 3.1)
  - Temporal: T5-style relative position encoding. Evidence: "Along the time axis, Walrus uses causal attention with T5-style relative position encoding." (Section 3.1)
- Where applied:
  - Spatial block: "Space positional embedding                  AxialRoPE" (Table 2)
  - Time block: "Time positional embedding                   LearnedRPE" (Table 2)
- Fixed vs modified per task / ablated:
  - Modified during finetuning (learnable RoPE): "During finetuning, we replicate the RoPE frequency parameters in each dimension and make these parameters trainable." (Appendix B.2.1)
  - Optional learned absolute position embeddings (APE) in finetuning: "During finetuning, we therefore initialize a learned APE embedding layer." (Appendix B.2.1)
  - APE compared/ablated: "Removing APE consistently improves performance." (Table 3 caption)

## 9. Positional Encoding as a Variable
- Core research variable vs fixed assumption: Positional encoding is treated as a configurable finetuning component rather than the main research focus.
- Multiple positional encodings compared: Yes (APE vs no APE, and learnable RoPE). Evidence: "During finetuning, we replicate the RoPE frequency parameters in each dimension and make these parameters trainable." and "During finetuning, we therefore initialize a learned APE embedding layer." (Appendix B.2.1); "Removing APE consistently improves performance." (Table 3 caption)
- Claim that PE choice is “not critical” or secondary: Not stated explicitly; the paper notes APE is not universally beneficial. Evidence: "Table 3 shows that the effect of APE is not universally beneficial." (Appendix B.2.1)

## 10. Evidence of Constraint Masking (Scale vs Structure)
- Model size(s): "Walrus, a 1.3B parameter transformer model" (Introduction); "Parameters                                   1.3 × 109                         6.4 × 108" (Table 2).
- Dataset size(s): "Walrus is pretrained on nineteen diverse scenarios" (Abstract); "both 2D and 3D data and 63 distinct state variables drawn from 19 physical scenarios" (Introduction); "this works out to about 4M examples per datasets for 2D data and 2M per dataset for 3D data." (Appendix B.1)
- Evidence of gains attributed to architecture/training tricks rather than scale alone:
  - Stabilization (patch jittering): "Patch jittering: a lightweight procedure to improve the stability of autoregressive rollouts ... reduce long-horizon error in 89% of the pretraining scenarios." (Introduction contributions)
  - Adaptive compute: "Adaptive-compute Tokenization: integrating recent adaptive-compute tokenization methods to allocate compute dynamically based on resolution or problem complexity." (Introduction contributions)
  - Training throughput: "Topology-aware Sampling: increasing training throughput by 262%" (Introduction contributions)
  - Diversity vs scale: "through controlled experiments on fixed architectures, we show the importance of our diversity-first approach to pretraining ... diversity-first approach leads to stronger downstream performance across a range of tasks." (Introduction)
- Claims that scaling model size or data is the primary driver: Not claimed; improvements are explicitly attributed to stabilization, adaptive tokenization, and diversity-first pretraining (see evidence above).

## 11. Architectural Workarounds
- Patch jittering for stability: "Patch jittering: a lightweight procedure to improve the stability of autoregressive rollouts" (Introduction contributions)
- 2D-to-3D augmentation: "Augmentation of 2D into 3D data: jointly handling 2D and 3D data in a single pipeline by treating 2D data as a plane randomly embedded in 3D space." (Introduction contributions)
- Adaptive-compute tokenization: "Adaptive-compute Tokenization: integrating recent adaptive-compute tokenization methods to allocate compute dynamically based on resolution or problem complexity." (Introduction contributions)
- Topology-aware sampling for throughput: "Topology-aware Sampling: increasing training throughput by 262%" (Introduction contributions)
- Compute-adaptive compression for varying resolutions: "Convolutional Stride Modulation ... to natively handle data at varying resolutions by adapting the level of downsampling/upsampling in each encoder/decoder block." (Section 3.1)
- Shared encoder/decoder per dimensionality: "All physical systems S of a given dimensionality d share a single encoder and decoder block" (Section 3.1, Shared Encoder-Decoder)

## 12. Explicit Limitations and Non-Claims
- Limitations on scientific validity in some datasets: "We found that for RSG, while interior layers are represented faithfully, the exterior often develops artifacts. In PNS, while the bulk dynamics are well captured, the true physical processes are highly sensitive such that the emulated system results in incorrect estimates of physically important quantities" (Section 5.1, Limitations)
- Remaining obstacles/future work: "Training on non-uniform geometries while maintaining efficiencies is a natural next exploration direction." (Conclusion, Limitations)
- Limits of current training objective for stochastic systems: "Walrus while having stochastic elements, is trained deterministically with full reconstruction loss. This could prove to be a representational bottleneck for poorly observed or stochastic systems." (Conclusion, Limitations)
