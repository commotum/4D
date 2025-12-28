### 1. Basic Metadata
- Title: VRoPE: Rotary Position Embedding for Video Large Language Models.
  Evidence: "VRoPE: Rotary Position Embedding for Video Large Language Models" (Title page, p.1)
- Authors: Zikang Liu; Longteng Guo; Yepeng Tang; Tongtian Yue; Junxian Cai; Kai Ma; Qingbin Liu; Xi Chen; Jing Liu.
  Evidence: "Zikang Liu1,2 * , Longteng Guo1 * , Yepeng Tang3 * , Tongtian Yue1,2" and "Junxian Cai4 , Kai Ma4 , Qingbin Liu4 , Xi Chen4 , Jing Liu1,2† ," (Title page, p.1)
- Year: 2025.
  Evidence: "arXiv:2502.11664v4 [cs.AI] 31 Oct 2025" (Title page, p.1)
- Venue: arXiv preprint.
  Evidence: "arXiv:2502.11664v4 [cs.AI] 31 Oct 2025" (Title page, p.1)

### 2. One-Sentence Contribution Summary
- The paper introduces VRoPE, a video-specific rotary positional encoding for Video-LLMs to address positional bias and video-text discontinuity and improve video understanding, temporal reasoning, and retrieval.
  Evidence: "In this section, we introduce Video Rotary Position Embedding (VRoPE), a novel positional encoding method tailored for Video-LLMs." (Section 4 Method: VRoPE) and "Extensive experiments on different model scales validate its superior performance in video understanding, temporal reasoning, and retrieval tasks." (Section 6 Conclusion)

### 3. Tasks Evaluated
- Task: General video understanding.
  Task type: Other (specify: general video understanding).
  Dataset(s): Video-MME.
  Domain: video.
  Evidence: "Results across tasks, including general video understanding (Video-MME), video temporal understanding (MVBench, TempCompass), and long video understanding (MLVU, LongVideoBench, EgoSchema)." (Table 2 caption, p.6)
- Task: Video temporal understanding.
  Task type: Other (specify: video temporal understanding).
  Dataset(s): MVBench; TempCompass.
  Domain: video.
  Evidence: "Results across tasks, including general video understanding (Video-MME), video temporal understanding (MVBench, TempCompass), and long video understanding (MLVU, LongVideoBench, EgoSchema)." (Table 2 caption, p.6)
- Task: Long video understanding.
  Task type: Other (specify: long video understanding).
  Dataset(s): MLVU; LongVideoBench; EgoSchema.
  Domain: video.
  Evidence: "Results across tasks, including general video understanding (Video-MME), video temporal understanding (MVBench, TempCompass), and long video understanding (MLVU, LongVideoBench, EgoSchema)." (Table 2 caption, p.6)
- Task: Long video retrieval (Video-NIAH / V-NIAH).
  Task type: Other (specify: retrieval).
  Dataset(s): Video-NIAH (Video Needle-In-A-Haystack).
  Domain: video.
  Evidence: "We compare our method with RoPE (Su et al., 2024) and RoPE-3D (Wang et al., 2024) on the long video retrieval task" and "Following the setup in Video-NIAH (Zhao et al., 2024), we conduct Video Needle-In-A-Haystack (V-NIAH) experiments, where a target \"needle\" frame is in- serted into a sequence of background frames, with the total frame count varying between 256 and 1216." (Section 5.3 Results on Long Video Retrieval, p.7)
- Task: Event-based temporal reasoning tasks.
  Task type: Other (specify: event-based tasks / temporal reasoning).
  Dataset(s): EventBench.
  Domain: video.
  Evidence: "additional evaluations focusing on event-based tasks involving complex temporal dependencies." (Appendix B.1 Results on EventBench, p.11)

### 4. Domain and Modality Scope
- Domain scope: Evaluation is within video benchmarks.
  Evidence: "We evaluated VRoPE across diverse video benchmarks" (Section 5.1 Evaluation Benchmarks).
- Modality scope: Video + text tokens are jointly modeled (multimodal).
  Evidence: "These visual to- kens are then concatenated with text tokens and fed into an LLM backbone." (Section 3.2 RoPE for Video-LLMs)
- Domain generalization or cross-domain transfer: Not claimed in the paper.

### 5. Model Sharing Across Tasks
Note: The paper describes a shared training pipeline but does not explicitly state per-task fine-tuning, shared weights, or separate heads for each benchmark.

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| General video understanding (Video-MME) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We evaluate the performance of RoPE, RoPE-3D, and our proposed VRoPE across six video un- derstanding benchmarks." (Section 5.2 Main Results) |
| Video temporal understanding (MVBench, TempCompass) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We evaluate the performance of RoPE, RoPE-3D, and our proposed VRoPE across six video un- derstanding benchmarks." (Section 5.2 Main Results) |
| Long video understanding (MLVU, LongVideoBench, EgoSchema) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We evaluate the performance of RoPE, RoPE-3D, and our proposed VRoPE across six video un- derstanding benchmarks." (Section 5.2 Main Results) |
| Long video retrieval (Video-NIAH / V-NIAH) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We compare our method with RoPE (Su et al., 2024) and RoPE-3D (Wang et al., 2024) on the long video retrieval task" (Section 5.3 Results on Long Video Retrieval) |
| Event-based temporal reasoning (EventBench) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "additional evaluations focusing on event-based tasks involving complex temporal dependencies." (Appendix B.1 Results on EventBench) |

### 6. Input and Representation Constraints
- Video-to-text tokenization and concatenation: Video frames become visual tokens that are concatenated with text tokens.
  Evidence: "video frames are typically pro- cessed by vision encoders (e.g., ViTs (Alexey, 2020) or CNNs (He et al., 2016)) and transformed into a sequence of visual tokens. These visual to- kens are then concatenated with text tokens and fed into an LLM backbone." (Section 3.2 RoPE for Video-LLMs)
- Spatial resolution and tokenization (main setup): "We use a 224 × 224 resolution for both image and video in- puts. For video input, the number of input frames is 16 and the frames are tokenized using a 2 × 2 pool- ing kernel with a stride of 2, i.e., each frame has 64 tokens as input." (Section 5.1 Experimental Setup)
- Spatiotemporal dimensionality: "for a video of size (W, H, T) with an initial posi- tion index pstart" (Section 4.2 Temporal Centered Arrangement)
- Variable length: "arbitrary length of video input does not affect the continuity between video and text." (Section 4.2 Temporal Centered Arrangement)
- Scaled setting (appendix): "We expand the num- ber of input frames to 32 and the resolution is set to 384 × 384." (Appendix B.4 Results of Larger Models and Datasets)
- Fixed patch size / padding / resizing requirements: Not specified in the paper.

### 7. Context Window and Attention Structure
- Maximum sequence length: Not specified; evaluation for retrieval uses long sequences.
  Evidence: "the total frame count varying between 256 and 1216." (Section 5.3 Results on Long Video Retrieval)
- Fixed vs variable sequence length: Variable-length video inputs are supported.
  Evidence: "arbitrary length of video input does not affect the continuity between video and text." (Section 4.2 Temporal Centered Arrangement)
- Attention type (global/windowed/sparse/hierarchical): Not specified in the paper.
- Cost-management mechanisms: Token pooling reduces per-frame tokens.
  Evidence: "frames are tokenized using a 2 × 2 pool- ing kernel with a stride of 2, i.e., each frame has 64 tokens as input." (Section 5.1 Experimental Setup)

### 8. Positional Encoding (Critical Section)
- Mechanism: Rotary position embedding variant (VRoPE).
  Evidence: "In this section, we introduce Video Rotary Position Embedding (VRoPE), a novel positional encoding method tailored for Video-LLMs." (Section 4 Method: VRoPE)
- Components: Symmetric Bias Mitigation + Temporal Centered Arrangement.
  Evidence: "by leveraging a combina- tion of Symmetric Bias Mitigation and Temporal Centered Arrangement." (Section 4 Method: VRoPE)
- Where applied: Positional embedding component of the LLM; text tokens keep RoPE.
  Evidence: "our improvements primarily target the positional embedding component of the LLM" (Figure 3 caption) and "For text tokens, we retain the original RoPE encoding structure (Eq. 5) to en- sure compatibility with LLMs." (Section 5.1)
- Applied in which layers: Not specified in the paper.
- Fixed vs varied across experiments: Varied; multiple positional encodings are compared (see Section 5.4).

### 9. Positional Encoding as a Variable
- Core research variable: Yes, multiple RoPE variants are compared.
  Evidence: "We compare our method with RoPE (Su et al., 2024) and RoPE-3D (Wang et al., 2024) on the long video retrieval task" (Section 5.3) and "We conduct ex- periments to assess the impact of three key proper- ties... We first compare RoPE-2D (Agrawal et al., 2024) and RoPE-3D (Wang et al., 2024) with the baseline RoPE... Next, we evaluate two additional variants, RoPE-Share and RoPE-Compact" (Section 5.4 Ablation Studies).
- Multiple positional encodings compared: Yes (RoPE, RoPE-2D, RoPE-3D, RoPE-Share, RoPE-Compact, VRoPE).
  Evidence: same as above.
- Claim that positional encoding choice is "not critical": Not specified in the paper.

### 10. Evidence of Constraint Masking
- Model sizes: "experiments were limited to models with 1.5B, 7B and 8B" (Section 8 Limitations).
- Dataset sizes: "This configura- tion results in approximately 1 million samples for pre-training and 3 million samples for instruction tuning." (Appendix B.4 Results of Larger Models and Datasets) and "pre-train the models on a randomly sam- pled 1M caption dataset" (Section 5.1 Training Data).
- Scaling settings (frames/resolution): "We expand the num- ber of input frames to 32 and the resolution is set to 384 × 384." (Appendix B.4 Results of Larger Models and Datasets)
- Performance gains attributed to architecture (not scaling): "VRoPE introduces no new learnable parameters and does not increase computational complexity, making it a cost-free performance enhancement for Video-LLMs." (Section 5.2/5.3)

### 11. Architectural Workarounds
- Symmetric Bias Mitigation to reduce positional bias.
  Evidence: "we propose Symmetric Bias Mitigation" (Section 4.1 Symmetric Bias Mitigation).
- Temporal Centered Arrangement to align video-text positions and allow length variability.
  Evidence: "we propose the Tempo- ral Centered Arrangement for positioning video frames." (Section 4.2 Temporal Centered Arrangement) and "arbitrary length of video input does not affect the continuity between video and text." (Section 4.2)
- Token pooling to reduce tokens per frame.
  Evidence: "frames are tokenized using a 2 × 2 pool- ing kernel with a stride of 2, i.e., each frame has 64 tokens as input." (Section 5.1)
- Vision encoder + MLP connector with two-stage training (freezing vision encoder).
  Evidence: "connect the Vision Encoder to the LLM using a Multi-Layer Perceptron (MLP)" and "in the instruction-tuning stage, both the MLP and LLM backbones are fine- tuned, with the Vision Encoder frozen throughout." (Section 5.1)
- Windowed/hierarchical/sparse attention or token pruning: Not specified in the paper.

### 12. Explicit Limitations and Non-Claims
- Limitations on model scale: "Due to computational resource constraints, our experiments were limited to models with 1.5B, 7B and 8B (shown in Ap- pendix B) parameters. Larger-scale models could potentially yield further performance gains." (Section 8 Limitations)
- Modality extension not addressed: "its extension to other modalities (e.g., audio, 3D point clouds, Electroencephalogra- phy (EEG)) and higher-dimensional data (e.g., 4D spatiotemporal or medical imaging data) remains an area for future research and validation." (Section 8 Limitations)

### 13. Constraint Profile (Synthesis)
- Domain scope: Evaluation is on "diverse video benchmarks" (video domain), with video + text token fusion.
- Task structure: Multiple video tasks (general understanding, temporal understanding, long video understanding, retrieval, event-based tasks).
- Representation rigidity: Fixed resolutions/frames in main setup (224 × 224, 16 frames, 64 tokens per frame), with explicit spatiotemporal coordinates (W, H, T); variable-length inputs are supported.
- Model sharing vs specialization: Shared training pipeline described, but per-task sharing/fine-tuning is not specified.
- Positional encoding role: Central research variable with multiple RoPE variants compared and ablated.

### 14. Final Classification
- Classification: Multi-task, single-domain.
- Justification: The paper evaluates multiple tasks across video benchmarks, e.g., "general video understanding (Video-MME), video temporal understanding (MVBench, TempCompass), and long video understanding (MLVU, LongVideoBench, EgoSchema)" (Table 2 caption) plus a "long video retrieval task" (Section 5.3). All evaluations are within video benchmarks ("We evaluated VRoPE across diverse video benchmarks"), and there is no claim of cross-domain transfer.
