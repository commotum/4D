Number of distinct tasks evaluated: 2 (image-text similarity search via a web demo using CLIP embeddings and kNN indices; text-to-image generation via training a DALL-E model on a LAION-400M subset). Evidence: "Web demo and similarity search... search images and texts based on a query image or text using the CLIP embeddings of the input and our precomputed kNN indices" and "Training DALL-E model. We ran DALLE-pytorch... to assess the dataset’s capability to train a text-to-image model." (2021_LAION400M.txt, Sec. 3, p. 4)

Number of trained model instances required to cover all tasks: 2. The similarity-search task relies on CLIP embeddings for image/text queries, while the generation task uses a separately trained DALL-E model (with a VQGAN image-tokenizer and CLIP-based ranking), indicating a distinct generative model in addition to the CLIP retrieval model. Evidence: "search images and texts... using the CLIP embeddings" and "We ran DALLE-pytorch... The VQGAN... is used to encode image tokens. For generation, we use CLIP ViT-B/16..." (2021_LAION400M.txt, Sec. 3, p. 4)

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
