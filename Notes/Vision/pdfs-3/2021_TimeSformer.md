## 1. Basic Metadata
- Title: Is Space-Time Attention All You Need for Video Understanding?
  - Evidence: "Is Space-Time Attention All You Need for Video Understanding?" (Title page)
- Authors: Gedas Bertasius; Heng Wang; Lorenzo Torresani.
  - Evidence: "Gedas Bertasius 1 Heng Wang 1 Lorenzo Torresani 1 2" (Title page)
- Year: 2021.
  - Evidence: "arXiv:2102.05095v4 [cs.CV] 9 Jun 2021" (Title page)
- Venue: arXiv (cs.CV).
  - Evidence: "arXiv:2102.05095v4 [cs.CV] 9 Jun 2021" (Title page)

## 2. One-Sentence Contribution Summary
TimeSformer is presented as "a convolution-free approach to video classification built exclusively on self-attention over space and time" by adapting Transformers to spatiotemporal patches. (Abstract)

## 3. Tasks Evaluated
General modality evidence: "The TimeSformer takes as input a clip X ∈ RH×W ×3×F consisting of F RGB frames of size H × W sampled from the original video." (3. The TimeSformer Model)

Task: Action recognition / video classification (Kinetics-400)
- Task type: Classification.
- Dataset(s): Kinetics-400.
- Domain: Video (human action categories).
- Evidence (task): "We evaluate TimeSformer on four popular action recognition datasets: Kinetics-400 (Carreira & Zisserman, 2017), Kinetics-600 (Carreira et al., 2018), Something-SomethingV2 (Goyal et al., 2017b), and Diving-48 (Li et al., 2018)." (4. Experiments)
- Evidence (dataset/domain): "Kinetics-400 (Carreira & Zisserman, 2017) consists of 240K training videos and 20K validation videos that span 400 human action categories." (Appendix: Datasets)

Task: Action recognition / video classification (Kinetics-600)
- Task type: Classification.
- Dataset(s): Kinetics-600.
- Domain: Video (human action categories).
- Evidence (task): "We evaluate TimeSformer on four popular action recognition datasets: Kinetics-400 (Carreira & Zisserman, 2017), Kinetics-600 (Carreira et al., 2018), Something-SomethingV2 (Goyal et al., 2017b), and Diving-48 (Li et al., 2018)." (4. Experiments)
- Evidence (dataset/domain): "Kinetics-600 (Carreira et al., 2018) has 392K training videos and 30K validation videos spanning 600 action categories." (Appendix: Datasets)

Task: Action recognition / video classification (Something-Something-V2)
- Task type: Classification.
- Dataset(s): Something-Something-V2.
- Domain: Video (action categories).
- Evidence (task): "We evaluate TimeSformer on four popular action recognition datasets: Kinetics-400 (Carreira & Zisserman, 2017), Kinetics-600 (Carreira et al., 2018), Something-SomethingV2 (Goyal et al., 2017b), and Diving-48 (Li et al., 2018)." (4. Experiments)
- Evidence (dataset/domain): "SomethingSomething-V2 (Goyal et al., 2017b) contains 170K training videos and 25K validation videos that span 174 action categories." (Appendix: Datasets)

Task: Action recognition / video classification (Diving-48)
- Task type: Classification.
- Dataset(s): Diving-48.
- Domain: Video (fine-grained diving categories).
- Evidence (task): "We evaluate TimeSformer on four popular action recognition datasets: Kinetics-400 (Carreira & Zisserman, 2017), Kinetics-600 (Carreira et al., 2018), Something-SomethingV2 (Goyal et al., 2017b), and Diving-48 (Li et al., 2018)." (4. Experiments)
- Evidence (dataset/domain): "Lastly, Diving-48 (Li et al., 2018) has 16K training videos and 3K testing videos spanning 48 fine-grained diving categories." (Appendix: Datasets)

Task: Long-term task classification / long-term video modeling (HowTo100M)
- Task type: Classification.
- Dataset(s): HowTo100M.
- Domain: Instructional videos (long videos, multi-task).
- Evidence (task): "Lastly, we evaluate TimeSformer on the task of long-term video modeling using HowTo100M (Miech et al., 2019)." (4.6. Long-Term Video Modeling)
- Evidence (dataset/domain): "HowTo100M is an instructional video dataset that contains around 1M instructional Web videos showing humans performing over 23K different tasks, such as cooking, repairing, making arts, etc." (4.6. Long-Term Video Modeling)
- Evidence (task definition): "Table 8. Long-term task classification on HowTo100M. Given a video spanning several minutes, the goal is to predict the long-term task demonstrated in the video (e.g., cooking breakfast, cleaning house, etc)." (Table 8 caption)

## 4. Domain and Modality Scope
- Domain/modality scope: Multiple domains within a single modality (RGB video), spanning action recognition datasets and instructional video.
  - Evidence (multiple datasets): "We evaluate TimeSformer on four popular action recognition datasets: Kinetics-400 (Carreira & Zisserman, 2017), Kinetics-600 (Carreira et al., 2018), Something-SomethingV2 (Goyal et al., 2017b), and Diving-48 (Li et al., 2018)." (4. Experiments)
  - Evidence (instructional domain): "HowTo100M is an instructional video dataset that contains around 1M instructional Web videos showing humans performing over 23K different tasks, such as cooking, repairing, making arts, etc." (4.6. Long-Term Video Modeling)
  - Evidence (single modality): "The TimeSformer takes as input a clip X ∈ RH×W ×3×F consisting of F RGB frames of size H × W sampled from the original video." (3. The TimeSformer Model)
- Domain generalization / cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Action recognition (Kinetics-400) | Not specified in the paper. | Initialized from ImageNet before training on video data. | Yes, 1-hidden-layer MLP for video classes. | "We adopt the “Base” ViT architecture (Dosovitskiy et al., 2020) pretrained on either ImageNet-1K or ImageNet-21K (Deng et al., 2009), as specified for each experiment." (4. Experiments) + "Thus, before training TimeSformer on video data, we initialize it with weights learned from ImageNet." (4.2. Comparison to 3D CNNs) + "On top of this representation we append a 1-hidden-layer MLP, which is used to predict the final video classes." (3. The TimeSformer Model) |
| Action recognition (Kinetics-600) | Not specified in the paper. | Initialized from ImageNet before training on video data. | Yes, 1-hidden-layer MLP for video classes. | "We adopt the “Base” ViT architecture (Dosovitskiy et al., 2020) pretrained on either ImageNet-1K or ImageNet-21K (Deng et al., 2009), as specified for each experiment." (4. Experiments) + "Thus, before training TimeSformer on video data, we initialize it with weights learned from ImageNet." (4.2. Comparison to 3D CNNs) + "On top of this representation we append a 1-hidden-layer MLP, which is used to predict the final video classes." (3. The TimeSformer Model) |
| Action recognition (Something-Something-V2) | Not specified in the paper. | Initialized from ImageNet before training on video data. | Yes, 1-hidden-layer MLP for video classes. | "We adopt the “Base” ViT architecture (Dosovitskiy et al., 2020) pretrained on either ImageNet-1K or ImageNet-21K (Deng et al., 2009), as specified for each experiment." (4. Experiments) + "Thus, before training TimeSformer on video data, we initialize it with weights learned from ImageNet." (4.2. Comparison to 3D CNNs) + "On top of this representation we append a 1-hidden-layer MLP, which is used to predict the final video classes." (3. The TimeSformer Model) |
| Action recognition (Diving-48) | Not specified in the paper. | Initialized from ImageNet before training on video data. | Yes, 1-hidden-layer MLP for video classes. | "We adopt the “Base” ViT architecture (Dosovitskiy et al., 2020) pretrained on either ImageNet-1K or ImageNet-21K (Deng et al., 2009), as specified for each experiment." (4. Experiments) + "Thus, before training TimeSformer on video data, we initialize it with weights learned from ImageNet." (4.2. Comparison to 3D CNNs) + "On top of this representation we append a 1-hidden-layer MLP, which is used to predict the final video classes." (3. The TimeSformer Model) |
| Long-term task classification (HowTo100M) | Not specified in the paper. | Yes; pretrained on Kinetics-400 and fine-tuned on HowTo100M. | Yes, 1-hidden-layer MLP for video classes. | "All models in this comparison are pretrained on Kinetics-400 before finetuning on HowTo100M." (4.6. Long-Term Video Modeling) + "On top of this representation we append a 1-hidden-layer MLP, which is used to predict the final video classes." (3. The TimeSformer Model) |

## 6. Input and Representation Constraints
- Input modality/shape: "The TimeSformer takes as input a clip X ∈ RH×W ×3×F consisting of F RGB frames of size H × W sampled from the original video." (3. The TimeSformer Model)
- Patchification: "we decompose each frame into N non-overlapping patches, each of size P × P , such that the N patches span the entire frame, i.e., N = HW/P 2 ." (3. The TimeSformer Model)
- Patch size (default experiments): "The patch size is 16 × 16 pixels." (4. Experiments)
- Default clip size and sampling: "Unless differently indicated, we use clips of size 8 × 224 × 224, with frames sampled at a rate of 1/32." (4. Experiments)
- Alternative clip sizes / resolutions: "TimeSformer, which is the default version of our model operating on 8 × 224 × 224 video clips, (2) TimeSformer-HR, a high spatial resolution variant that operates on 16 × 448 × 448 video clips, and lastly (3) TimeSformer-L, a long-range configuration of our model that operates on 96 × 224 × 224 video clips with frames sampled at a rate of 1/4." (4.2. Comparison to 3D CNNs)
- Resizing and cropping (training): "During training, we first resize the shorter side of the video to a random value in [256, 320]. We then randomly sample a 224 × 224 crop from the resized video. For our high-resolution model, TimeSformer-HR, we resize the shorter side of the video to a random value in [448, 512], and then randomly sample a 448 × 448 crop." (Appendix: Implementation Details)
- Resizing and cropping (inference): "We scale the shorter spatial side of a video to 224 pixels (or 448 for TimeSformer-HR) and take 3 crops of size 224×224 (448 × 448 for TimeSformer-HR) to cover a larger spatial extent within the clip." (Appendix: Implementation Details)
- Fixed number of tokens / fixed resolution: Not specified in the paper. (Token count varies with H, W, and F via N = HW/P 2 and the multiple clip sizes above.)
- Padding requirements: Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper. The maximum tested clip length is constrained by: "Due to GPU memory constraints, we are not able to test our model on clips longer than 96 frames." (4.3. Varying the Number of Tokens)
- Fixed vs variable sequence length: Variable in experiments (different clip sizes and frame counts).
  - Evidence: "Unless differently indicated, we use clips of size 8 × 224 × 224, with frames sampled at a rate of 1/32." (4. Experiments)
  - Evidence: "TimeSformer-HR, a high spatial resolution variant that operates on 16 × 448 × 448 video clips, and lastly (3) TimeSformer-L, a long-range configuration of our model that operates on 96 × 224 × 224 video clips with frames sampled at a rate of 1/4." (4.2. Comparison to 3D CNNs)
- Attention types used:
  - Global (full) self-attention is implied by: "it requires computing a similarity measure for all pairs of tokens." (1. Introduction)
  - Space-only attention: "We can reduce the computational cost by replacing the spatiotemporal attention of Eq. 5 with spatial attention within each frame only (Eq. 6)." (3. The TimeSformer Model)
  - Divided space-time attention: "We propose a more efficient architecture for spatiotemporal attention, named “Divided Space-Time Attention” (denoted with T+S), where temporal attention and spatial attention are separately applied one after the other." (3. The TimeSformer Model)
  - Sparse local-global attention: "(L+G) first computes a local attention by considering the neighboring F × H/2 × W/2 patches and then calculates a sparse global attention over the entire clip using a stride of 2 patches along the temporal dimension and also the two spatial dimensions." (3. The TimeSformer Model)
  - Axial attention: "Finally, “Axial” attention decomposes the attention computation in three distinct steps: over time, width and height." (3. The TimeSformer Model)
- Cost-management mechanisms:
  - Divided attention reduces comparisons: "Note that compared to the (N F + 1) comparisons per patch needed by the joint spatiotemporal attention model of Eq. 5, Divided Attention performs only (N +F +2) comparisons per patch." (3. The TimeSformer Model)
  - Sparse local-global and axial factorize or sparsify attention (see quotes above).

## 8. Positional Encoding (Critical Section)
- Mechanism: Learned positional embeddings added to patch embeddings.
  - Evidence: "z(p,t) = Ex(p,t) + epos
(p,t)" and "where epos
(p,t) ∈ R represents a learnable positional embedding added to encode the spatiotemporal position of each patch." (3. The TimeSformer Model)
- Where applied: At input embedding stage (added to patch embeddings).
  - Evidence: "We linearly map each patch x(p,t) into an embedding vector z(p,t) ∈ RD ... z(p,t) = Ex(p,t) + epos
(p,t)." (3. The TimeSformer Model)
- Fixed vs modified / ablated: Positional embedding is compared across variants.
  - Evidence: "we also conduct experiments with a few variants of TimeSformer that use: (1) no positional embedding, (2) space-only positional embedding, and (3) space-time positional embedding." (4.4. The Importance of Positional Embeddings)

## 9. Positional Encoding as a Variable
- Treated as a research variable: Yes (explicit ablations).
  - Evidence: "we also conduct experiments with a few variants of TimeSformer that use: (1) no positional embedding, (2) space-only positional embedding, and (3) space-time positional embedding." (4.4. The Importance of Positional Embeddings)
- Multiple positional encodings compared: Yes (none vs space-only vs space-time).
  - Evidence: same quote as above.
- Claim that PE choice is not critical: Not specified in the paper.

## 10. Evidence of Constraint Masking
- Model size(s): "TimeSformer has a large learning capacity (the number of parameters is 121.4M )." (4.2. Comparison to 3D CNNs)
- Dataset size(s):
  - "Kinetics-400 ... consists of 240K training videos and 20K validation videos" and "Kinetics-600 ... has 392K training videos and 30K validation videos" and "SomethingSomething-V2 ... contains 170K training videos and 25K validation videos" and "Diving-48 ... has 16K training videos and 3K testing videos" (Appendix: Datasets)
  - "HowTo100M is an instructional video dataset that contains around 1M instructional Web videos showing humans performing over 23K different tasks" (4.6. Long-Term Video Modeling)
- Performance gains tied to scaling data: "TimeSformer outperforms the other models only when using enough training videos." (Figure 4 caption) and "we trained TimeSformer on different subsets of K400 and SSv2: {25%, 50%, 75%, 100%} of the full datasets." (4.3. Varying the Number of Tokens / Impact of Video-Data Scale)
- Performance gains tied to scaling tokens/resolution: "We see that increasing the spatial resolution (up to a certain point) leads to a boost in performance. Similarly, we observe that increasing the length of the input clip leads to consistent accuracy gains." (4.3. Varying the Number of Tokens)
- Performance gains tied to pretraining data scale: "ImageNet-21K pretraining is beneficial for K400, where it leads to a consistently higher accuracy compared to ImageNet-1K pretraining." (4.2. Comparison to 3D CNNs)
- Architectural factorization vs full attention: "Divided Attention performs only (N +F +2) comparisons per patch" and "this space-time factorization is not only more efficient but it also leads to improved classification accuracy." (3. The TimeSformer Model)

## 11. Architectural Workarounds
- Patchification of frames: "we decompose each frame into N non-overlapping patches, each of size P × P" (3. The TimeSformer Model) — tokenizes video into patches to enable Transformer processing.
- Space-only attention (cost reduction): "We can reduce the computational cost by replacing the spatiotemporal attention of Eq. 5 with spatial attention within each frame only (Eq. 6)." (3. The TimeSformer Model)
- Divided space-time attention (factorized attention): "temporal attention and spatial attention are separately applied one after the other" and "Divided Attention performs only (N +F +2) comparisons per patch." (3. The TimeSformer Model)
- Sparse local-global attention (sparsity pattern): "(L+G) first computes a local attention ... then calculates a sparse global attention over the entire clip using a stride of 2 patches along the temporal dimension and also the two spatial dimensions." (3. The TimeSformer Model)
- Axial attention (factorized axes): "“Axial” attention decomposes the attention computation in three distinct steps: over time, width and height." (3. The TimeSformer Model)

## 12. Explicit Limitations and Non-Claims
- Training from scratch is difficult: "Due to a large number of parameters, training our model from scratch is difficult." (4.2. Comparison to 3D CNNs)
- Maximum clip length tested: "Due to GPU memory constraints, we are not able to test our model on clips longer than 96 frames." (4.3. Varying the Number of Tokens)
- Performance limitation on SSv2: "Our results suggest that TimeSformer achieves lower accuracy than the best models on this dataset." (4.5. Comparison to the State-of-the-Art)
- Future work / non-claims: "In the future, we plan to extend our method to other video analysis tasks such as action localization, video captioning and question-answering." (5. Conclusion)

## 13. Constraint Profile (Synthesis)
Constraint Profile:
- Domain scope: Multiple video datasets (action recognition and instructional video), single modality (RGB video).
- Task structure: Classification-centric (action recognition and long-term task classification).
- Representation rigidity: Patchified frames with fixed patch size (16 × 16) and fixed clip sizes per experiment; resizing/cropping prescribed.
- Model sharing vs specialization: Pretraining on ImageNet (or Kinetics-400 for HowTo100M) followed by dataset-specific training; no joint multi-task training described.
- Role of positional encoding: Learned spatiotemporal positional embeddings are explicitly ablated (none vs space-only vs space-time).

## 14. Final Classification
Classification: Multi-task, multi-domain (constrained).
Justification: The paper evaluates multiple tasks/datasets within video classification, including action recognition datasets and long-term task classification on instructional videos ("We evaluate TimeSformer on four popular action recognition datasets..." and "Lastly, we evaluate TimeSformer on the task of long-term video modeling using HowTo100M"). All evaluations are within a single modality (RGB video input clips), and there is no claim of unrestrained multi-modal or open-world learning.
