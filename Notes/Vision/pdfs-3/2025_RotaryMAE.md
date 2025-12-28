## 1. Basic Metadata

- Title: Rotary Masked Autoencoders are Versatile Learners
- Authors (verbatim from PDF header): "Uros Zivanovic1 , Serafina Di Gioia2, 3 , Andre Scaffidi3 , Martín de los Rios3 , Gabriella Contardo7, 3 , and Roberto Trotta3, 4, 5, 6"
- Year: 2025
- Venue: NeurIPS 2025 (conference); arXiv:2505.20535v2

Evidence (header): "Rotary Masked Autoencoders are Versatile Learners"; "Uros Zivanovic1 , Serafina Di Gioia2, 3 , Andre Scaffidi3 , Martín de los Rios3 , Gabriella Contardo7, 3 , and Roberto Trotta3, 4, 5, 6"; "arXiv:2505.20535v2 [cs.LG] 8 Nov 2025"; "39th Conference on Neural Information Processing Systems (NeurIPS 2025)."

## 2. One-Sentence Contribution Summary

The paper introduces RoMAE, an MAE-based model that works natively with irregular multivariate time-series while preserving performance on standard modalities such as images and audio.

Evidence (Contributions): "RoMAE: An expansion of MAE that works natively with irregular multivariate time-series without sacrificing any performance on standard modalities such as images and audio."

## 3. Tasks Evaluated

Task 1:
- Task name: Absolute position reconstruction
- Task type: Reconstruction; Other: position regression
- Dataset(s): Synthetic positions (sequence of identical values with positions sampled between 0 and 50)
- Domain: Synthetic 1D positional data

Task evidence:
- Section 5.1: "we give the model a sequence of 10 identical values as input. Each embedding is then given a 1D position sampled uniformly between 0 and 50. We then use the same linear head to predict the position for all tokens."

Task 2:
- Task name: Image classification
- Task type: Classification
- Dataset(s): Tiny ImageNet
- Domain: Natural images

Task evidence:
- Section 5.2: "we train three versions of RoMAE on Tiny ImageNet [31]"
- Contributions: "(ii) image classification"

Task 3:
- Task name: Audio classification
- Task type: Classification
- Dataset(s): ESC-50 (finetuning); AudioSet-20k and Librispeech (pretraining)
- Domain: Audio (environmental sounds; speech)

Task evidence:
- Section 5.3: "We thus pretrain RoMAE using two different data sets: AudioSet-20k and the Librispeech dataset."
- Section 5.3: "For the finetuning audio classification benchmark, we used the ESC-50 dataset [39], consisting of 2000 5-second environmental audio recordings classified into 50 classes."

Task 4:
- Task name: Irregular multivariate time-series classification
- Task type: Classification
- Dataset(s): DESC ELAsTiCC Challenge; UEA Multivariate Time-series Archive
- Domain: Astronomical light-curve time-series; multivariate time-series

Task evidence:
- Section 5.4: "The DESC ELAsTiCC Challenge is a multi-variate irregular time-series dataset"
- Section 5.4: "We evaluate RoMAE on a variety of datasets from the UEA Multivariate Time-series Archive [3]."
- Contributions: "(i) irregularly sampled multi-variate time-series classification"

Task 5:
- Task name: Irregular time-series regression (Pendulum)
- Task type: Other: regression
- Dataset(s): Pendulum dataset
- Domain: Irregularly sampled images (pendulum time-series)

Task evidence:
- Section 5.4: "Irregular Time-series Regression: Pendulum Dataset The Pendulum dataset [51] is an irregular time-series dataset consisting of irregularly sampled images of a pendulum."
- Section 5.4: "RoMAE is trained directly on regression without any pre-training, predicting the sine and cosine of the angle of the pendulum"

Task 6:
- Task name: Irregular time-series interpolation
- Task type: Reconstruction; Other: interpolation
- Dataset(s): Spiral, Synthetic, PhysioNet
- Domain: Synthetic 2D trajectories; synthetic univariate time-series; medical ICU time-series

Task evidence:
- Section 5.5: "We evaluate RoMAE on three interpolation tasks with increasing dimensionality and sampling irregularity. (i) Spiral: A 2D synthetic benchmark of 300 noisy Archimedean spirals as in Ref. [12]; (ii) Synthetic: The 50-step univariate task from Ref. [48] and (iii) PhysioNet: 48-hour ICU records containing 41 clinical variables [49]."
- Contributions: "(iii) irregularly sampled time-series interpolation"

## 4. Domain and Modality Scope

- Single domain: No.
  Evidence (Contributions): "RoMAE: An expansion of MAE that works natively with irregular multivariate time-series without sacrificing any performance on standard modalities such as images and audio."
- Multiple domains within the same modality: Yes, multiple time-series domains/datasets are evaluated.
  Evidence (Section 5.4): "We evaluate RoMAE on a variety of datasets from the UEA Multivariate Time-series Archive [3]."; "Irregular Time-series Regression: Pendulum Dataset"
  Evidence (Section 5.5): "We evaluate RoMAE on three interpolation tasks with increasing dimensionality and sampling irregularity."
- Multiple modalities: Yes (time-series, images, audio).
  Evidence (Contributions): "RoMAE: An expansion of MAE that works natively with irregular multivariate time-series without sacrificing any performance on standard modalities such as images and audio."
- Domain generalization or cross-domain transfer claims: Not claimed.

## 5. Model Sharing Across Tasks

The paper describes training per dataset/task; no explicit joint multi-task training is stated.

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Absolute position reconstruction | Not specified | Not specified | Yes (linear head) | "We then use the same linear head to predict the position for all tokens." (Section 5.1) |
| Image classification (Tiny ImageNet) | Not specified | Yes | Yes (classification head) | "After pre-training each model for 200 epochs, we fine-tune for another 15."; "we place the classification head on top of the mean of the output embeddings, otherwise we place the head on top of the [CLS] token." (Section 5.2) |
| Audio classification (ESC-50) | Not specified | Yes | Yes (classification head) | "We thus pretrain RoMAE using two different data sets: AudioSet-20k and the Librispeech dataset."; "For the finetuning audio classification benchmark, we used the ESC-50 dataset [39]"; "This token becomes useful during fine-tuning, when an MLP head can be placed on top of it to conduct classification." (Section 5.3; Section 4.1) |
| Irregular time-series classification (ELAsTiCC/UEA) | Not specified | Yes | Yes (classification head) | "We train RoMAE-tiny by conducting full pre-training for 200 epochs with a masking ratio of 75%, then fine-tuning for 25 epochs."; "For each dataset we conduct pre-training for 400 epochs. When fine-tuning, we found it necessary to change hyper-parameters between different datasets."; "This token becomes useful during fine-tuning, when an MLP head can be placed on top of it to conduct classification." (Section 5.4; Section 4.1) |
| Irregular time-series regression (Pendulum) | Not specified | No (trained directly) | Not specified | "RoMAE is trained directly on regression without any pre-training, predicting the sine and cosine of the angle of the pendulum" (Section 5.4) |
| Irregular time-series interpolation (Spiral/Synthetic/PhysioNet) | Not specified (single pre-trained model mentioned) | Not specified | Yes (reconstruction head) | "one tiny/small RoMAE model, pre-trained once with a generic masked-autoencoder objective and no task-specific architectural tuning, matches or surpasses specialised baselines across three increasingly difficult interpolation datasets."; "the model head predicts np values for each patch that was masked out." (Section 6; Section 4.1) |

## 6. Input and Representation Constraints

Explicit constraints and assumptions stated in the paper:
- ND patchification with fixed patch sizes per dimension and non-overlapping segments.
  Evidence (Section 4): "we define a patch size (p1 , · · · , pD ) and divide each dimension into Ni = di /pi non-overlapping segments"
- Irregular dimensions must use patch size 1.
  Evidence (Section 4): "Proposition 4.1. For any irregular dimension di in x, the corresponding patch size for that dimension pi must be equal to 1."
- Embedding dimension constraints due to Axial RoPE.
  Evidence (Section 3.1): "since RoPE requires that embeddings be even and Axial RoPE requires that embeddings be divisible by D, this puts constraints on the possible values that dmodel can take."
- Example fixed patch sizes for specific modalities/tasks.
  Evidence (Section 5.2): "We use a patch size of (16, 16)"
  Evidence (Section 5.4): "we use a patch size of (1, 24, 24) for (time, height, width)."
- Variable-length inputs are padded with a pad mask (ELAsTiCC).
  Evidence (Appendix D.6): "In order to handle the variable number of points per sample we utilize padding, applying a pad mask to the attention scores."

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Variable; sequence length depends on patchification and padding is used for variable-length samples.
  Evidence (Section 4): "we define a patch size (p1 , · · · , pD ) and divide each dimension into Ni = di /pi non-overlapping segments"
  Evidence (Appendix D.6): "we utilize padding, applying a pad mask to the attention scores."
- Attention type: Standard global scaled dot-product attention (SDPA).
  Evidence (Section 3): "The matrices Q, K, V containing q,k,v are then passed through Scaled Dot-Product Attention (SDPA)"
- Mechanisms to manage computational cost (windowing, pooling, pruning, etc.): Not described; standard attention is used and noted as a limitation.
  Evidence (Section 6): "RoMAE is also not well suited for very long sequences, as it uses standard Attention which has O(n2 ) memory complexity with regards to sequence length."

## 8. Positional Encoding (Critical Section)

- Mechanism used: RoPE (rotary) with continuous positions; Axial RoPE and p-RoPE are used for multi-dimensional positions.
  Evidence (Contributions): "Continuous Positional Embedding with RoPE: We investigate how RoPE can be used to embed continuous positions"
  Evidence (Section 3.1): "RoPE is applied directly to the queries and keys before they enter SDPA."; "Axial RoPE: To encode multi-dimensional position, we utilize Axial RoPE [17]."; "p-RoPE: In this work we make use of p-RoPE [4]"
- Where applied: Queries and keys before SDPA.
  Evidence (Section 3.1): "RoPE is applied directly to the queries and keys before they enter SDPA."
- Fixed vs modified across experiments: Positional encoding is varied in ablations/comparisons (RoPE with/without [CLS], absolute sin/cos, quantized vs continuous RoPE).
  Evidence (Section 5.2): "we train three versions of RoMAE on Tiny ImageNet [31]; RoPE with the [CLS] token, RoPE without the [CLS] token, and absolute sinusoidal positional embeddings [58]"
  Evidence (Appendix B.2): "We evaluate the performance of RoMAE when using different positional encoding methods, specifically: absolute sin/cos [58], RoPE with integer (quantized) positions, and RoPE with continuous positions."

## 9. Positional Encoding as a Variable

- Core research variable: Yes.
  Evidence (Contributions): "Continuous Positional Embedding with RoPE: We investigate how RoPE can be used to embed continuous positions"
- Multiple positional encodings compared: Yes.
  Evidence (Section 5.2): "RoPE with the [CLS] token, RoPE without the [CLS] token, and absolute sinusoidal positional embeddings [58]"
  Evidence (Appendix B.2): "absolute sin/cos [58], RoPE with integer (quantized) positions, and RoPE with continuous positions"
- Claim that PE choice is “not critical” or secondary: Not specified in the paper.

## 10. Evidence of Constraint Masking

Model sizes:
- "Throughout the experiments we make use of different sizes of RoMAE: RoMAE-tiny, RoMAE-small, and RoMAE-base" (Section 5)
- "We use a custom size for RoMAE which contains only 2 layers and an MLP hidden size of 30." (Section 5.4, Pendulum)

Dataset sizes (explicit counts reported):
- "in 2 million 10-second segments of YouTube videos" (Section 5.3, AudioSet description)
- "ESC-50 dataset [39], consisting of 2000 5-second environmental audio recordings classified into 50 classes." (Section 5.3)
- "Our generated training set has 20 000 samples" (Section 5.1)
- "All the datasets trained on are relatively small, with some being on the order of hundreds of samples." (Section 5.4, UEA)

Attribution of performance gains (scaling, data, architecture, training tricks):
- Pretraining vs baselines: "A key reason for RoMAE’s better performance might be that ATAT does not conduct any pre-training." (Section 5.4)
- Model scale: "The larger RoMAE-tiny achieves an improvement of .18 F-score." (Section 5.4)
- Data scale: "showing that the size and richness of the pretraining dataset impacts, in a non-negligible way, the performance of that model on the finetuning tasks." (Section 5.3)
- Training trick attribution: "which we attribute to MAE tubelet-masking enforcing long-range reasoning." (Section 6)

## 11. Architectural Workarounds

- ND patchification to convert N-D inputs into a token sequence.
  Evidence (Section 4): "we define a patch size (p1 , · · · , pD ) and divide each dimension into Ni = di /pi non-overlapping segments"
- Constraint for irregular dimensions (patch size fixed to 1).
  Evidence (Section 4): "For any irregular dimension di in x, the corresponding patch size for that dimension pi must be equal to 1."
- Asymmetric encoder/decoder to reduce compute in MAE-style pretraining.
  Evidence (Section 4.1): "RoMAE’s structure follows MAE’s, using an asymmetric encoder/decoder, with the encoder being much larger than the decoder."
- p-RoPE truncation to improve robustness to sequence length variation.
  Evidence (Section 3.1): "The unchanged region in the embedding space provides the model with a data channel it can use to pass information into SDPA without any modifications by RoPE, making the model more robust to varying sequence length."
- Axial RoPE and dimensional index trick to handle many positional dimensions.
  Evidence (Section 4.2): "we optionally reserve a dimension in Axial RoPE that is used to store the dimensional index i"; "allows us to reduce the number of positional dimensions from 6 to 2."
- Task-specific heads and masking for pretraining/classification.
  Evidence (Section 4.1): "This token becomes useful during fine-tuning, when an MLP head can be placed on top of it to conduct classification."; "the model head predicts np values for each patch that was masked out."
- Padding with attention mask for variable-length samples.
  Evidence (Appendix D.6): "we utilize padding, applying a pad mask to the attention scores."

## 12. Explicit Limitations and Non-Claims

Explicit limitations:
- Additional overhead for continuous positions: "RoPE in RoMAE has some additional computational overhead if the positions are different with each forward pass, e.g., with any continuous irregular time-series." (Section 6)
- Long-sequence limitation: "RoMAE is also not well suited for very long sequences, as it uses standard Attention which has O(n2 ) memory complexity with regards to sequence length." (Section 6)
- Extrapolation limitation: "RoMAE’s ability to perform on extrapolation tasks is limited" (Section 6); "Being a BERT-style model, RoMAE is not well suited for extrapolation." (Appendix B.3)
- Position reconstruction generalization: "the model does not find solutions that generalize to out-of-distribution positions." (Appendix B.1)

Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not specified in the paper.
