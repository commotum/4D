# The Challenge

An ARC-AGI task involves a small “training” set of three to five input–output pixel grids, each grid represented using integer-encoded colors. All training pairs exemplify a shared but hidden transformation. After observing these examples, the task provides exactly one additional input grid (with its correct output withheld). The objective is to predict the withheld output by generalizing from the training grids.


# The Dataset

Our training dataset contains millions of explicit code-image pairs. Each pair consists of a p5.js sketch and the specific image or set of images it generates. Specifically, the dataset includes:

- Single Images: sketches generating one static image.
- Before–After Image Pairs: sketches applying a simple visual transformation through two images depicting before and after states.
- Before–After Image Sets: sketches applying a common visual transformation to multiple images with clear before and after states.

These pairs are structured into a graduated curriculum, beginning with simple black-and-white shapes (points, lines, triangles, rectangles, circles), progressively adding spatial arrangements, color fills and outlines, basic 2D transformations (rotation, scaling, shear), layered operations (clipping, masking), fundamental 3D shapes (boxes, spheres), lighting and material effects, and ultimately fully composed 2D and 3D scenes. This stepwise progression allows the model to first learn basic visual concepts, then combine these elements into more complex visual structures.


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
