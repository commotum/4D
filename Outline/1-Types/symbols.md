I'm thinking about just how difficult it is to work with symbols and representations. For example, I'm training a new AI model on Sudoku games, and tbh there's no real reason why we fill the places in with the numbers/digits 1-9 and not the letters A-I, or 9 ascending hues, 9 musical notes, or 9 distinct game pieces. As far as the neural network is concerned, they're just 9 high dimensional embedding tokens. If it's trained only on sudokus there's no way for it to recognize that A-I and 1-9 are symbolic parallels. And I'm wondering. I've been thinking about tokenization, and the possibility of using type tokens. You see there are two large problems with the SOTA in AI, and I think they are highly related.

The first is the vocab/tokenization problem. Standard token-based approaches in large language models (LLMs) typically rely on constrained vocabularies due to computational and representational limitations. Common data types, such as RGB colors or 64-bit integers (int64), present an immense combinatorial challenge. For instance, the RGB color space alone contains over 16 million unique values, vastly exceeding the vocabularies used by popular models like LLaMA-3.2 (\~128K tokens), DeepSeek-V3 (\~129K tokens), Claude (\~65K tokens), GPT-4 (\~100K tokens), GPT-4o (\~200K tokens), and Gemma (\~256K tokens). Likewise, the int64 data type spans approximately $9.22 \times 10^{18}$ distinct values, rendering explicit tokenization computationally infeasible.

The second problem is persistent spatial generalization deficits in SOTA NNs, a widely acknowledged flaw with a proven, though not broadly recognized, cause: real numbers. Real-valued neural networks rely on high-dimensional feature embeddings with independent weights, which inevitably result in feature disentanglement failures. With no forcing mechanism real-valued NNs fail to naturally bundle and encode hidden inter-dependencies in multidimensional data, such as RGB channels in pixels or time-correlated derivatives in acoustic signals. 

Fortunately for us, the solution to the first problem can be found by extending the solution to our second problem.


PUT A WORD HERE LIKE STANDARD UP PROJECTIONS with the proven solution for the second problem: quaternions. While it may not be immediately obvious how...

Quaternion-valued neural networks networks overcome spatial generalization deficits by accurately encoding and preserving multidimensional data (e.g., RGB channels or geometric affine transformations) that real-valued networks mishandle. 

1. Quaternions retain interdependency of features
2. Quaternions reduce parameter counts by 1/4 which is better than before, but if we combine it with quaternion based projections we don't just reduce our parameter count by 1/4, we reduce it by 99.9999 percent


They also yield a fourfold reduction in parameters, enhancing efficiency.

While this reduction alone cannot handle vast spaces like int64 ($9.22 \times 10^{18}$ values, quartered to $2.305 \times 10^{18}$), the key lies in quaternion-based feature encoding. To address tokenization challenges, we propose a structured approach: represent tokens via a **type** (e.g., RGB, int64) and a **value**, then apply a learned linear transformation—an *up-projection*—to map the low-dimensional value into a high-dimensional embedding space.

---

For RGB colors, map each value from hexadecimal `[00, FF]` to `[-1.0, 1.0]` and encode as a purely imaginary quaternion, yielding four float values. If we were employing a real-valued up-projection matrix for a 512-dimensional embedding, this would require only a $512 \times 4$ matrix (2,048 parameters, 16 for every 4 dimensions)—a drastic reduction of over 99.999999% from the more than 8.5 billion parameters needed to explicitly store embeddings for over 16 million distinct values. 

However, by using quaternion-valued weight matrices, we reduce the parameter count even further, quartering our 2,048 real-valued matrix to just 512 real-valued parameters representing just 128 quaternion weights.

Furthermore, by using quaternion weights, high dimensional embedding decoding becomes efficient and exact, as quaternion inversion recovers original values in constant time $O(1)$. 

Map each token to a **(type, value)** and up-project the low-D value with a single learned matrix per type. For RGB, map `[00,FF]` to `[-1,1]` and pack it as a purely imaginary quaternion `(0, r, g, b)`. That one matrix replaces the whole giant table—about a **99.99999%** cut—so for a 512-d embedding you’re down to a simple **512×4 = 2,048**-parameter real matrix instead of billions.

Then make the lift quaternion-native: split 512-d into 128 quaternion channels and compute each output as a quaternion product `w ⊗ q` (multiply two quaternions) instead of a 4×4 real matmul. Per 4-d block that drops ops and learned weights **from 16 to 4**, shrinking the 2,048 real params to **128 quaternions = 512 reals**. Net effect: the massive table goes away. You keep a tiny type lookup plus one small up-projection per type, and decoding is easy—pick the type and invert the quaternion to recover the value in O(1). That’s how this cleanly solves the table-embedding problem.

For RGB colors, map each value from hexadecimal `[00, FF]` to `[-1.0, 1.0]` and encode as a purely imaginary quaternion, yielding four float values. 

While traditionally projecting four real dimensions would require 16 real weights, quaternion arithmetic allows the same projection using just four quaternion weights. Thus, the original real-valued matrix of size $512 \times 4 = 2048$ parameters becomes just 128 quaternion weights (512 real-valued parameters)

If we were employing a real-valued up-projection matrix for a 512-dimensional embedding, this would require only a $512 \times 4$ matrix (2,048 parameters)—a drastic reduction of over 99.999999% from the more than 8.5 billion parameters needed to explicitly store embeddings for over 16 million distinct values. 


For RGB colors, map each value from hexadecimal `[00, FF]` to `[-1.0, 1.0]` and encode as a purely imaginary quaternion, yielding four float values.


 A real-valued up-projection matrix for a 512-dimensional embedding would require only a $512 \times 4$ matrix (2,048 parameters)—a reduction of over 99.999999% from the more than 8.5 billion parameters needed to explicitly store embeddings for over 16 million distinct values. 

effectively quartering parameter counts. While traditionally projecting four real dimensions would require 16 real weights, quaternion arithmetic allows the same projection using just four quaternion weights. Thus, the original real-valued matrix of size $512 \times 4 = 2048$ parameters becomes just 128 quaternion weights (512 real-valued parameters).



However, by using quaternion-valued weight matrices, our real parameter count is cut from 2048 to 512, 128 4D quaternions, an additional 1/4 reduction. 


However, by using quaternion-valued weight matrices further quarters this to 512 real-valued parameters (128 quaternion weights).

This approach resolves the vocab/tokenization problem by replacing large discrete embedding tables with a small discrete table of types, each paired with a compact, learned mapping function for values. It enhances efficiency and representational power without the memory requirements of standard token embedding tables.

During generation, the model outputs two high-dimensional embeddings: one for type and one for value. The type embedding is decoded via a standard lookup to identify the data type (e.g., RGB), which selects the appropriate inversion function to decode the value embedding back to its original low-dimensional form (e.g., RGB components). Decoding is efficient and exact, as quaternion inversion recovers original values in constant time $O(1)$. Quaternion algebra is well-documented in the literature, so we omit basic details here.


---


1. **Vocab/Tokenization Problem**: Standard token-based approaches in LLMs use constrained vocabularies that can't efficiently handle vast combinatorial spaces, such as RGB colors (over 16 million values) or int64 integers (about 9.22 × 10¹⁸ values), leading to computational infeasibility and poor representation.

2. **Spatial Generalization Deficits**: Real-valued neural networks fail to naturally encode and preserve interdependencies in multidimensional data (e.g., RGB channels or time-correlated signals), causing feature disentanglement issues and poor generalization in tasks like vision or signal processing.

3. **Frequency-Biased Updates in Learned Embeddings**: Updates to weights in low-dimensional embeddings are unevenly distributed based on token frequency in the training corpus, resulting in poor generalization for rare or unseen values in sparse spaces like int64.


---


1. **Vocab/Tokenization Problem**: The primary cause is the inherent need to discretize continuous or combinatorially vast input spaces (like natural language's infinite possible strings, non-English texts, or data types such as RGB colors and int64 integers) into a finite, manageable vocabulary for embedding tables in neural networks. This explosion arises from factors like the proliferation of unique identifiers in code, which balloons vocab size; inefficiencies in subword algorithms (e.g., Byte-Pair Encoding) that split rare or complex terms into more tokens, leading to longer sequences and higher inference costs; and language-specific biases where non-English texts require disproportionately more tokens due to English-centric designs. Tokenization was adopted as a solution to the original problem of handling out-of-vocabulary (OOV) words and variable-length text in early language models, where full-word vocabularies could reach millions (causing sparse one-hot encodings and memory issues), by breaking text into frequent subwords for a compact ~30k-200k token set that enables generalization to unseen combinations. The key tradeoff is computational efficiency and scalability (smaller embedding matrices, faster training) versus representational incompleteness and expressivity (e.g., struggles with spelling, arithmetic, or non-text data where explicit enumeration is infeasible, leading to poor handling of rare patterns or vast domains).

2. **Spatial Generalization Deficits**: This stems from the independent treatment of dimensions in real-valued neural networks, which lack built-in mechanisms (inductive biases) to enforce or preserve hidden interdependencies and correlations in multidimensional data, such as rotations in 3D space, color channel interactions, or temporal signals—resulting in feature disentanglement failures during optimization. In ML contexts, this is exacerbated by weak regularization or insufficient architectural constraints that fail to embed spatial structures effectively, drawing parallels to biological disruptions like lesions in navigation-related brain networks that impair spatial cognition. Real-valued networks were chosen for their simplicity and generality in solving the broader problem of scalable gradient-based optimization across diverse tasks, avoiding the complexity of specialized algebras. The tradeoff is ease of implementation and broad applicability (e.g., in transformers) versus reduced expressivity in structured domains, where models generalize poorly to spatial transformations or multimodal data without additional hacks like data augmentation.

3. **Frequency-Biased Updates in Learned Embeddings**: The root cause is the reliance on stochastic gradient descent in corpus-based training, where updates to embedding weights are proportional to token frequency in the data—common items receive abundant gradients for robust representations, while rare or long-tail values (e.g., obscure integers) get sparse updates, amplifying biases from imbalanced datasets and leading to overfitting on frequent patterns. Embeddings were introduced to address the inefficiency of one-hot encodings in capturing semantic similarities (solving the "curse of dimensionality" in early NLP), by learning dense vectors that encode proximity and context. The tradeoff is improved representational power for dense, meaningful features versus vulnerability to data distribution biases, resulting in poor generalization to infrequent or unseen values and potential inheritance of societal/frequency-based prejudices from training corpora.