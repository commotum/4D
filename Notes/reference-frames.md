# The Architecture

## Token Values

Token values are high-dimensional embedding vectors representing the specific value of any given token (e.g., numerical values, strings, colors, vectors).

## Token Types

Token types are also represented by high-dimensional embedding vectors, capturing the semantic category of each token (e.g., `color`, `integer`, `string`, `coordinate`, `Self`, or `Action`). Token embeddings are computed by multiplying their corresponding type and value embeddings element-wise. This structure enables the model to generalize effectively across similar semantic categories while distinguishing unique values.

### Example Token Types and Values:

* Type: `Color`, Value: "Red"
* Type: `Integer`, Value: "42"
* Type: `String`, Value: "import"
* Type: `Coordinate`, Value: "(10,20)"
* Type: `Self`, Value: "Model"
* Type: `Action`, Value: "Move", "Write", "Read"

## Autoregressive Hyperspace Navigation

Using these embeddings, the model autoregressively generates structured sequences that define both spatial navigation and meaningful operations within the latent hyperspace. For example, the model can produce sequences like:

* Type: `Action` - Value: "Move"
* Type: `Vector` - Value: "Left"
* Type: `Action` - Value: "Write"
* Type: `String` - Value: "import"

These sequences enable the model not only to perform spatial transformations but also to dynamically interact with specific files, images, code modules, or other discrete entities represented in hyperspace (e.g., moving directly to "image\_1" or "image\_2"). This architecture ensures powerful compositional reasoning capabilities, allowing the model to generalize visual and code-based transformations learned from training examples.
