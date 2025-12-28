## 1. Basic Metadata
Title: VideoBERT: A Joint Model for Video and Language Representation Learning
Authors: Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid
Year: 2019
Venue: arXiv preprint (arXiv:1904.01766v2 [cs.CV], 11 Sep 2019)

Evidence (Front matter, page 1):
"VideoBERT: A Joint Model for Video and Language Representation Learning"
"Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid"
"arXiv:1904.01766v2 [cs.CV] 11 Sep 2019"

## 2. One-Sentence Contribution Summary
VideoBERT adapts BERT to jointly model discretized video tokens and ASR text for self-supervised video-language representation learning, enabling downstream tasks like action classification and video captioning without manual labels.

## 3. Tasks Evaluated
Task: Text-to-video prediction (text-to-video generation)
Task type: Generation
Dataset(s): Not specified in the paper (examples shown in Figures 1 and 2)
Domain: Not specified in the paper (examples use recipe text)
Evidence (Section 1. Introduction):
"tasks. For example, we can perform text-to-video predic-
tion, which can be used to automatically illustrate a set of
instructions (such as a recipe), as shown in the top examples
of Figure 1 and 2."

Task: Future forecasting (video-to-video prediction of future tokens)
Task type: Generation / forecasting
Dataset(s): Not specified in the paper (illustrated in Figures 1 and 2)
Domain: Not specified in the paper
Evidence (Section 1. Introduction):
"We can also use our model in a “unimodal” fashion. For
example, the implied marginal distribution p(x) is a lan-
guage model for visual words, which we can use for long-
range forecasting. This is illustrated in the bottom examples
of Figure 1 and 2."

Task: Zero-shot action classification (verb/noun prediction)
Task type: Classification
Dataset(s): YouCook II
Domain: instructional cooking videos (YouCook II)
Evidence (Section 4.4. Zero-shot action classification):
"Once pretrained, the VideoBERT model can be used
for “zero-shot” classification on novel datasets, such as
YouCook II (By “zero-shot” we mean the model is not
trained on YouCook II data nor with the same label ontol-
ogy used in YouCook II)."
Evidence (Section 4.4. Zero-shot action classification):
"More precisely, we want to com-
pute p(y|x) where x is the sequence visual tokens, and y is
a sequence of words. Since the model is trained to predict
sentences, we define y to be the fixed sentence, “now let
me show you how to [MASK] the [MASK],” and ex-
tract the verb and noun labels from the tokens predicted in
the first and second masked slots, respectively."
Evidence (Section 4.1. Dataset):
"We evaluate VideoBERT on the YouCook II dataset [38],
which contains 2000 YouTube videos averaging 5.26 min-
utes in duration, for a total of 176 hours."

Task: Video captioning (dense video captioning)
Task type: Generation
Dataset(s): YouCook II
Domain: instructional cooking videos (YouCook II)
Evidence (Section 4.6. Transfer learning for captioning):
"We evaluate the extracted features on video captioning,
following the setup from [39], where the ground truth video
segmentations are used to train a supervised model map-
ping video segments to captions."
Evidence (Table 3 caption):
"Table 3: Video captioning performance on YouCook II. We follow the setup from [39] and report captioning performance on
the validation set, given ground truth video segments. Higher numbers are better."

## 4. Domain and Modality Scope
Single domain evaluation: Yes, cooking instructional videos (YouCook II).
Evidence (Section 4.1. Dataset):
"the case for instructional videos, and we focus on cooking
videos specifically, since it is a well studied domain with
existing annotated datasets available for evaluation."
Evidence (Section 4.1. Dataset):
"We evaluate VideoBERT on the YouCook II dataset [38],
which contains 2000 YouTube videos averaging 5.26 min-
utes in duration, for a total of 176 hours."

Multiple domains within the same modality: Not specified in the paper.

Multiple modalities: Yes, video + language (ASR text).
Evidence (Section 1. Introduction):
"More precisely, our approach is to apply BERT to learn a
model of the form p(x, y), where x is a sequence of “visual
words”, and y is a sequence of spoken words."

Domain generalization / cross-domain transfer: Not claimed.
Evidence (Section 5. Discussion and conclusion):
"Beyond improving the model, we plan to assess our ap-
proach on other video understanding tasks, and on other do-
mains besides cooking."

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Text-to-video prediction | Yes (pretrained VideoBERT used directly) | Not specified in the paper | Not specified in the paper | "we can perform text-to-video predic- tion, which can be used to automatically illustrate a set of instructions (such as a recipe)" (Section 1. Introduction) |
| Future forecasting | Yes (pretrained VideoBERT used directly) | Not specified in the paper | Not specified in the paper | "the implied marginal distribution p(x) is a lan- guage model for visual words, which we can use for long- range forecasting." (Section 1. Introduction) |
| Zero-shot action classification | Yes (pretrained VideoBERT used) | No (not trained on YouCook II) | Not specified in the paper | "the model is not trained on YouCook II data nor with the same label ontol- ogy used in YouCook II)." (Section 4.4) |
| Video captioning | VideoBERT used as feature extractor; separate supervised model trained | Not specified in the paper (VideoBERT fine-tuning not stated) | Yes (transformer encoder-decoder) | "We evaluate the extracted features on video captioning, following the setup from [39], where the ground truth video segmentations are used to train a supervised model map- ping video segments to captions." (Section 4.6) |

## 6. Input and Representation Constraints
Frame sampling and clip length: 20 fps; 30-frame (1.5 seconds) non-overlapping windows.
Evidence (Section 4.2. Video and Language Preprocessing):
"For each input video, we sample frames at 20 fps, and
create clips from 30-frame (1.5 seconds) non-overlapping
windows over the video."

Visual features and dimensionality: S3D features, 1024-dimension vector after 3D average pooling.
Evidence (Section 4.2. Video and Language Preprocessing):
"In this
work, we use the S3D [34] which adds separable temporal
convolutions to an Inception network [25] backbone. We
take the feature activations before the final linear classifier
and apply 3D average pooling to obtain a 1024-dimension
feature vector."

Visual tokenization: hierarchical k-means with d=4, k=12 (20,736 clusters).
Evidence (Section 4.2. Video and Language Preprocessing):
"We tokenize the visual features using hierarchical k-
means. We adjust the number of hierarchy levels d and the
number of clusters per level k by visually inspecting the co-
herence and representativeness of the clusters. We set d=4
and k = 12, which yields 124
= 20736 clusters in total."

Text tokenization and vocabulary: WordPieces; 30,000 tokens; ASR punctuation via LSTM.
Evidence (Section 4.2. Video and Language Preprocessing):
"For each ASR word sequence, we break the stream
of words into sentences by adding punctuation using an
off-the-shelf LSTM-based language model. For each sen-
tence, we follow the standard text preprocessing steps from
BERT [6] and tokenize the text into WordPieces [33]. We
use the same vocabulary provided by the authors of BERT,
which contains 30,000 tokens."

Video segmentation: ASR timestamps define segments; 16 tokens per segment when ASR is missing.
Evidence (Section 4.2. Video and Language Preprocessing):
"Unlike language which can be naturally broken into sen-
tences, it is unclear how to break videos into semantically
coherent segments. We use a simple heuristic to address
this problem: when an ASR sentence is available, it is as-
sociated with starting and ending timestamps, and we treat
video tokens that fall into that time period as a segment.
When ASR is not available, we simply treat 16 tokens as a
segment."

Video token vocabulary size: 20,736 visual words appended to embedding lookup table.
Evidence (Section 4.3. Model Pre-training):
"We add support for video tokens by appending 20,736
entries to the word embedding lookup table for each of
our new “visual words”."

Input resolution: Not specified in the paper.
Fixed patch size: Not specified in the paper.
Fixed number of tokens: Not specified in the paper (segments are defined by ASR timestamps or 16-token heuristic).
Fixed dimensionality (strictly 2D): Not specified in the paper (features are 1024-dimension vectors).
Padding/resizing requirements: Not specified in the paper.

## 7. Context Window and Attention Structure
Maximum sequence length: Not specified in the paper.
Evidence (Section 3.1. The BERT model):
"let x = {x1, . . . , xL} be a set of discrete to-
kens, xl ∈ X."

Sequence length fixed or variable: Not specified in the paper (sequence length denoted as L; segmentation uses ASR timestamps or 16-token heuristic).
Evidence (Section 4.2. Video and Language Preprocessing):
"When ASR is not available, we simply treat 16 tokens as a
segment."

Attention type (global/windowed/hierarchical/sparse): Not specified in the paper (Transformer-based).
Evidence (Section 3.1. The BERT model):
"The function f(x\l) is a multi-layer bidirectional trans-
former model [28] that takes an L × D1 tensor, contain-
ing the D1-dimensional embedding vectors corresponding
to x\l, and returns an L × D2 tensor, where D2 is the size
of the output of each transformer node."

Mechanisms to manage computational cost: Subsampling video tokens (1 to 5 steps) and frame/clip tokenization.
Evidence (Section 3.2. The VideoBERT model):
"we randomly pick a subsampling
rate of 1 to 5 steps for the video tokens. This not only helps
the model be more robust to variations in video speeds, but
also allows the model to capture temporal dynamics over
greater time horizons and learn longer-term state transi-
tions."

## 8. Positional Encoding (Critical Section)
Positional encoding mechanism: absolute position tags with learned embeddings summed with token embeddings (input-level).
Evidence (Section 3.1. The BERT model):
"The above model is permutation invariant. In order to
capture order information, we can “tag” each word with its
position in the sentence. The BERT model learns an embed-
ding for each of the word tokens, as well as for these tags,
and then sums the embedding vectors to get a continuous
representation for each token."

Where applied: Input embeddings (token + position tag embeddings) as described above.
Evidence: same as above.

Fixed across all experiments / modified per task / ablated: Not specified in the paper.

## 9. Positional Encoding as a Variable
Core research variable vs fixed assumption: Not specified in the paper (only described as part of BERT; no experimental variation stated).
Multiple positional encodings compared: Not specified in the paper.
Claims that PE choice is not critical: Not specified in the paper.
Evidence (Section 3.1. The BERT model):
"The above model is permutation invariant. In order to
capture order information, we can “tag” each word with its
position in the sentence. The BERT model learns an embed-
ding for each of the word tokens, as well as for these tags,
and then sums the embedding vectors to get a continuous
representation for each token."

## 10. Evidence of Constraint Masking (Scale vs Structure)
Model size(s):
Evidence (Section 4.3. Model Pre-training):
"we use the BERTLARGE model re-
leased by the authors of [6], using the same backbone archi-
tecture: it has 24 layers of Transformer blocks, where each
block has 1024 hidden units and 16 self-attention heads."

Dataset size(s):
Evidence (Section 4.1. Dataset):
"resulting in a set of 312K
videos. The total duration of this dataset is 23,186 hours, or
roughly 966 days."
Evidence (Section 4.5. Benefits of large training sets):
"we take random subsets
of 10K, 50K and 100K videos from the pretraining set,
and pretrain VideoBERT using the same setup as above,
for the same number of epochs."

Performance gains attributed to scaling data:
Evidence (Section 4.5. Benefits of large training sets):
"We can see that the accuracy grows monotonically
as the amount of data increases, showing no signs of satura-
tion. This indicates that VideoBERT may benefit from even
larger pretraining datasets."

Performance gains attributed to cross-modal information:
Evidence (Abstract):
"confirm that large amounts
of training data and cross-modal information are critical to
performance."
Evidence (Section 4.6. Transfer learning for captioning):
"We can also see that cross-modal pre-
training outperforms the video-only version."

Attribution to architectural hierarchy or training tricks: Not specified in the paper.

## 11. Architectural Workarounds
Discrete visual tokenization (vector quantization into “visual words”):
Evidence (Section 3.2. The VideoBERT model):
"transform the raw visual data
into a discrete sequence of tokens. To this end, we propose
to generate a sequence of “visual words” by applying hi-
erarchical vector quantization to features derived from the
video using a pretrained model."

Special token to combine text and video sentences ([>]):
Evidence (Section 3.2. The VideoBERT model):
"such as this: [CLS] orange chicken with [MASK]
sauce [>] v01 [MASK] v08 v72 [SEP], where v01
and v08 are visual tokens, and [>] is a special token we in-
troduce to combine text and video sentences."

Random concatenation of neighboring sentences to handle misalignment:
Evidence (Section 3.2. The VideoBERT model):
"we first randomly concatenate neighbor-
ing sentences into a single long sentence, to allow the model
to learn semantic correspondence even if the two are not
well aligned temporally."

Subsampling video tokens to handle variable speeds / longer horizons:
Evidence (Section 3.2. The VideoBERT model):
"we randomly pick a subsampling
rate of 1 to 5 steps for the video tokens. This not only helps
the model be more robust to variations in video speeds, but
also allows the model to capture temporal dynamics over
greater time horizons and learn longer-term state transi-
tions."

Segmentation heuristic (ASR timestamps or fixed 16 tokens):
Evidence (Section 4.2. Video and Language Preprocessing):
"when an ASR sentence is available, it is as-
sociated with starting and ending timestamps, and we treat
video tokens that fall into that time period as a segment.
When ASR is not available, we simply treat 16 tokens as a
segment."

Multi-objective training across modalities (text-only, video-only, video-text) with alignment classification:
Evidence (Section 3.2. The VideoBERT model):
"Overall, we have three training regimes corresponding
to the different input data modalities: text-only, video-only
and video-text. For text-only and video-only, the standard
mask-completion objectives are used for training the model.
For text-video, we use the linguistic-visual alignment clas-
sification objective described above."

## 12. Explicit Limitations and Non-Claims
Future work on other ways of combining video and text:
Evidence (Section 3.2. The VideoBERT model):
"We leave investigation into other ways of combining
video and text to future work."

Future work on evaluation techniques:
Evidence (Section 4.4. Zero-shot action classification):
"we leave more sophisticated evaluation
techniques for future work."

Need for spatially fine-grained visual representations (limitation of current clip-level approach):
Evidence (Section 5. Discussion and conclusion):
"For many applications, includ-
ing cooking, it is important to use spatially fine-grained vi-
sual representations, instead of just working at the frame or
clip level, so that we can distinguish individual objects and
their attributes. We envision either using pretrained object
detection and semantic segmentation models, or using unsu-
pervised techniques for broader coverage."

Need for multi-scale temporal modeling beyond current frame-skipping vocabulary:
Evidence (Section 5. Discussion and conclusion):
"We also want to
explicitly model visual patterns at multiple temporal scales,
instead of our current approach, that skips frames but builds
a single vocabulary."

Future work on other domains (not claimed as current capability):
Evidence (Section 5. Discussion and conclusion):
"Beyond improving the model, we plan to assess our ap-
proach on other video understanding tasks, and on other do-
mains besides cooking."

Explicit statement about not using manual labeling:
Evidence (Section 2. Related Work - Instructional videos):
"We differ from this work in that we do not
use any manual labeling, and we learn a large-scale genera-
tive model of both words and (discretized) visual signals."

Other explicit non-claims (open-world/multi-domain/meta-learning): Not specified in the paper.
