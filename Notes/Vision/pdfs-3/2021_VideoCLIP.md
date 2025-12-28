## 1. Basic Metadata
- Title: VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding
  Evidence (Title block):
  "VideoCLIP: Contrastive Pre-training for
  Zero-shot Video-Text Understanding"
- Authors: Hu Xu; Gargi Ghosh; Po-Yao Huang; Dmytro Okhonko; Armen Aghajanyan; Florian Metze; Luke Zettlemoyer; Christoph Feichtenhofer
  Evidence (Title block):
  "Hu Xu1 , Gargi Ghosh1 , Po-Yao Huang12 , Dmytro Okhonko1 , Armen Aghajanyan1
  Florian Metze,1 Luke Zettlemoyer1 and Christoph Feichtenhofer1"
- Year: 2021
  Evidence (Front matter):
  "arXiv:2109.14084v2 [cs.CV] 1 Oct 2021"
- Venue: arXiv (arXiv:2109.14084v2)
  Evidence (Front matter):
  "arXiv:2109.14084v2 [cs.CV] 1 Oct 2021"

## 2. One-Sentence Contribution Summary
VideoCLIP proposes a contrastive pre-training method for a unified video-text Transformer to enable zero-shot transfer to video-text understanding tasks without downstream labels.

## 3. Tasks Evaluated
### Task: Text-Video Retrieval
- Task type: Other (retrieval/ranking)
- Dataset(s): Youcook2, MSR-VTT, DiDeMo
- Domain: video (cooking videos; open-domain videos; Flicker videos)
- Evidence (task + datasets):
  "Text-Video Retrieval. We use Youcook2 and
  MSR-VTT to evaluate text-video retrieval."
  "We use Youcook2,
  MSR-VTT and DiDeMo to evaluate zero-shot
  transfer to text-video retrieval."
  "Youcook2 (Zhou et al., 2017) is a collection of
  2K cooking videos with a total duration of 176
  hours and 5.26 minutes on average per video."
  "MSR-VTT (Xu et al., 2016) is a widely-
  compared benchmark dataset for text-video re-
  trieval and video question answering. It contains
  open-domain videos where each video clips is
  around 10 seconds."
  "DiDeMo (Anne Hendricks et al., 2017) has 10,000
  videos annotated with 40,000 sentences on Flicker
  videos. We evaluate video-paragraph retrieval on
  4021 available testing examples."

### Task: VideoQA (multiple-choice)
- Task type: Other (multiple-choice QA/ranking)
- Dataset(s): MSR-VTT (QA test data)
- Domain: video (open-domain videos)
- Evidence (task + dataset):
  "In multiple-choice
  VideoQA (Yu et al., 2018), the model aligns each
  video with one out of several text candidate an-
  swers."
  "VideoQA. We further use the QA test data (Yu
  et al., 2018) for MSR-VTT to evaluate multiple-
  choice VideoQA."
  "MSR-VTT (Xu et al., 2016) is a widely-
  compared benchmark dataset for text-video re-
  trieval and video question answering. It contains
  open-domain videos where each video clips is
  around 10 seconds."

### Task: Action Segmentation
- Task type: Segmentation
- Dataset(s): COIN
- Domain: video
- Evidence (task + dataset):
  "Action Segmentation. Action segmentation as-
  signs each token (or frame) of a video with one of
  the pre-defined labels to separate meaningful seg-
  ments of videos from the rest tokens (or frames)."
  "Action Segmentation. We use COIN (Tang
  et al., 2019) to evaluate action segmentation. It has
  11,827 videos (476 hours) in total and the testing
  set has 2797 videos, where each video is labeled
  with 3.91 segments per video on average."

### Task: Action Step Localization (action localization)
- Task type: Segmentation; Other (localization)
- Dataset(s): CrossTask
- Domain: video
- Evidence (task + dataset):
  "Action Step Localization. We use CrossTask
  (Zhukov et al., 2019) to evaluate action localiza-
  tion. It contains 83 different tasks and 4.7K videos.
  Each task has a set of steps in the form of text
  descriptions and each frame of video is an-
  notated with one or multiple steps as a distribu-
  tion."

## 4. Domain and Modality Scope
- Single domain? No. Multiple domains within the same modality (video) are used.
  Evidence (Section 5.2 End Task Setups and Appendix A.1):
  "Youcook2 (Zhou et al., 2017) is a collection of
  2K cooking videos with a total duration of 176
  hours and 5.26 minutes on average per video."
  "MSR-VTT (Xu et al., 2016) is a widely-
  compared benchmark dataset for text-video re-
  trieval and video question answering. It contains
  open-domain videos where each video clips is
  around 10 seconds."
  "DiDeMo (Anne Hendricks et al., 2017) has 10,000
  videos annotated with 40,000 sentences on Flicker
  videos."
- Multiple modalities? Yes (video + text).
  Evidence (Introduction):
  "transfer to video-text understanding tasks."
- Domain generalization or cross-domain transfer claimed? Not claimed.
  Evidence of domain shift discussion (Section 5.4 Main Results):
  "The ma-
jor reason could be domain shift from HowTo100M
to MSR-VTT."

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Text-Video Retrieval | Yes (pretrained once; shared for zero-shot across tasks) | Yes (reported) | Not specified in the paper. | "After pre-training, we apply our model for zero-\nshot transfer without any fine-tuning on target\ndataset labels."; "VideoCLIP (Fine-tuned)             32.2 62.6 75.0" |
| VideoQA | Yes (pretrained once; shared for zero-shot across tasks) | Yes (reported) | Not specified in the paper. | "After pre-training, we apply our model for zero-\nshot transfer without any fine-tuning on target\ndataset labels."; "VideoCLIP (Fine-tuned)                    92.1" |
| Action Segmentation | Yes (pretrained once; shared for zero-shot across tasks) | Yes (reported) | Not specified in the paper. | "After pre-training, we apply our model for zero-\nshot transfer without any fine-tuning on target\ndataset labels."; "VideoCLIP (Fine-tuned)                   68.7" |
| Action Step Localization | Yes (pretrained once; shared for zero-shot across tasks) | Yes (reported) | Not specified in the paper. | "After pre-training, we apply our model for zero-\nshot transfer without any fine-tuning on target\ndataset labels."; "VideoCLIP (Fine-tuned)                47.3" |

## 6. Input and Representation Constraints
- Fixed or variable input resolution? Not specified in the paper.
- Fixed patch size? Not specified in the paper.
- Fixed number of tokens? Max video tokens fixed; text tokens bounded and variable.
  Evidence (Section 5.3 Implementation Details):
  "We limit the maximum number of video tokens
to be 32. For video transformer, its input sequence
is 34 with [CLS] and [SEP] tokens. For text
transformer, we have 61 text tokens plus [CLS]
and [SEP] tokens (63 in total)."
  "A text clip has a random length between 8 and 61
tokens, whereas a video clip has 3 to 32 seconds."
- Fixed dimensionality? Video token and model input dimensions are fixed.
  Evidence (Section 5.3 Implementation Details):
  "It is pre-trained on HowTo100M (Miech et al., 2020) to extract video
tokens of dimension 512."
  "to map the S3D outputs to the 768-dimensional inputs of the video
Transformer."
- Padding or resizing requirements? Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: video input 34 tokens; text input 63 tokens.
  Evidence (Section 5.3 Implementation Details):
  "We limit the maximum number of video tokens
to be 32. For video transformer, its input sequence
is 34 with [CLS] and [SEP] tokens. For text
transformer, we have 61 text tokens plus [CLS]
and [SEP] tokens (63 in total)."
- Sequence length fixed or variable? Variable within bounds for text and video clips.
  Evidence (Section 5.3 Implementation Details):
  "A text clip has a random length between 8 and 61
tokens, whereas a video clip has 3 to 32 seconds."
- Attention type (global/windowed/hierarchical/sparse)? Not specified in the paper.
- Mechanisms to manage computational cost: token limits; sliding window for long videos.
  Evidence (Section 5.3 Implementation Details and Section 5.2 End Task Setups):
  "We limit the maximum number of video tokens
to be 32."
  "we apply a sliding window with a step size of 16 seconds
and a window size of 32 seconds."

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism used: Not specified in the paper.
- Where it is applied: Not specified in the paper.
- Fixed across experiments / modified per task / ablated: Not specified in the paper.

## 9. Positional Encoding as a Variable
- Treated as a core research variable or fixed assumption? Not specified in the paper.
- Multiple positional encodings compared? Not specified in the paper.
- Claims PE choice is not critical or secondary? Not specified in the paper.

## 10. Evidence of Constraint Masking (Scale vs. Structure)
- Model size(s): BERT base initialization; 6-layer video encoder, 12-layer text encoder.
  Evidence (Section 5.3 Implementation Details):
  "we initialize their weights with the pre-trained BERTBASE-uncased (Devlin et al., 2019)."
  "We only use the first 6 Transformer layers for the video input and
all 12 layers for the text input."
- Dataset size(s):
  Evidence (Section 5.1 VideoCLIP Pre-training and Section 5.2 / Appendix A.1):
  "We use
1.1M videos after filtering out videos which are not
available or cannot be decoded."
  "Youcook2 (Zhou et al., 2017) is a collection of
2K cooking videos with a total duration of 176
hours and 5.26 minutes on average per video."
  "In total,
there are 200K clip-text pairs from 10K videos."
  "DiDeMo (Anne Hendricks et al., 2017) has 10,000
videos annotated with 40,000 sentences on Flicker
videos."
  "It has
11,827 videos (476 hours) in total and the testing
set has 2797 videos, where each video is labeled
with 3.91 segments per video on average."
  "It contains 83 different tasks and 4.7K videos."
- Performance gains attributed to scaling model size or data? Not claimed; gains are attributed to training techniques (overlapping positives and retrieval-augmented negatives).
  Evidence (Sections 3.3, 3.4, 5.6):
  "we pre-train with temporally overlapped pairs of
video and text clips (of varying length), thereby
greatly increasing the quality and quantity of the
video-text alignment."
  "We propose a retrieval aug-
mented pre-training approach to retrieve a cluster
of videos that are similar to each other for each
training batch."
  "VideoCLIP without retrieval augmented training significantly drops
performance by over 4% in R@1 and addition-
ally using exact alignment positives, i.e., the same
start/end timestamp for a pair of video and text
clips, has another 4% drop in R@1."
  "Different from CLIP
that scales pre-training data for zero-shot transfer
to image classification on an explicitly assembled
dataset using a simple contrastive objective (Chen
et al., 2020a), this paper uses a publicly established
pre-training dataset, HowTo100M (Miech et al.,
2019)."

## 11. Architectural Workarounds
- Temporally overlapped positives to improve alignment.
  Evidence (Section 3.3 Overlapped Video-Text Clips):
  "we pre-train with temporally overlapped pairs of
video and text clips (of varying length), thereby
greatly increasing the quality and quantity of the
video-text alignment."
- Retrieval-augmented negative sampling (hard negatives).
  Evidence (Section 3.4 Retrieval Augmented Training):
  "We propose a retrieval aug-
mented pre-training approach to retrieve a cluster
of videos that are similar to each other for each
training batch."
- Pooling choice to support token-level tasks.
  Evidence (Section 3.1 Video and Text Encoding):
  "We use average pooling (instead of using the
[CLS] token)"
- Token and window constraints for long videos.
  Evidence (Section 5.3 Implementation Details and Section 5.2 End Task Setups):
  "We limit the maximum number of video tokens
to be 32."
  "we apply a sliding window with a step size of 16 seconds
and a window size of 32 seconds."
- Text-encoder as hyper network for segmentation labels.
  Evidence (Section 4 Zero-shot Transfer to End Tasks):
  "the text en-
 coder of VideoCLIP can serve as self-supervision
 for videos during pre-training and as a hyper net-
 work to provide hidden states of segment textual
 labels for a video token."

## 12. Explicit Limitations and Non-Claims
- No-label claim for downstream tasks.
  Evidence (Introduction):
  "Video-
CLIP outperforms all existing zero-shot methods
and even outperforms fully supervised pre-training
+ fine-tuning methods, but without using any labels."
- No access to validation data in zero-shot thresholding.
  Evidence (Section 4 Zero-shot Transfer to End Tasks):
  "Note that in zero-shot trans-
fer, there is no access to training or validation data
to decide a threshold as a hyper-parameter."
- Outside label not modeled explicitly (action segmentation).
  Evidence (Section 5.2 End Task Setups):
  "we do not model the
Outside label explicitly and determine an Outside
label only when all other 778 labels reject a video
token."
- Training data scope limitation (HowTo100M only).
  Evidence (Section 5.5 Discussion on Work that Fine-tunes CLIP Model):
  "is not fair to compare to our approach which only
trains on HowTo100M instructional videos."
- Future work statement.
  Evidence (Section 5.7 Qualitative Analysis):
  "We leave incorporating such type of knowledge
into pre-training to future work."
