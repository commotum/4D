Ok, that's far too many categories. I'm looking for maybe 3-4 categories.

Maybe:

# Positional Encoding Improvement Proposal

**Criteria:**

1. Does work on transformer based models (Attention is the core mechanism)
2. Critiques some aspect of any method of positional encoding used with prior models 
3. The paper's core presentation is a modification, change, or innovation within positional encoding which aims to improve it in some fashion. (This is SHOULD BE distinct from increasing the number of dimensions encoded.)

**Examples:**
- Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings (PoPE)
- ComRoPE: Scalable and Robust Rotary Position Embedding
- LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate
- YaRN: Efficient Context Window Extension of Large Language Models

---  

# Increasing Transformer's Dimensions

**Criteria:**
1. Does work on transformer based models (Attention is the core mechanism) 
2. Highlights the success of transformer based models in modeling language and other 1-dimensional tasks and/or notes the need for 2D, 3D, 4D, nD, graph-based, or other type of positional encodings for tasks in other domains.
3. The paper’s core presentation is a Transformer adaptation that enables modeling of a higher-dimensional domain (e.g., images/video/3D/graphs). This adaptation may be architectural (attention structure, encoder/decoder changes) or representational (tokenization/serialization/ordering/patching and associated positional treatment), as long as it is central to the paper’s contribution. (This SHOULD BE distinct from modifying or improving the positional encoding mechanism.)

**Examples:**
- Generative Pretraining from Pixels
- An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale
- RoFormer: Enhanced Transformer with Rotary Position Embedding

