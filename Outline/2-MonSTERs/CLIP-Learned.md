Yes—**for the ViT image encoder in OpenAI’s CLIP, the patch tokens use a learned (trainable) absolute positional embedding**.

In the released OpenAI CLIP code, the VisionTransformer defines `self.positional_embedding` as an `nn.Parameter` with one embedding per patch position **plus one for the class token**, i.e. shape roughly `(grid_size^2 + 1, width)`. ([GitHub][1])

(Separately: CLIP’s **text** encoder also uses a learned absolute positional embedding with a fixed context length—commonly 77 tokens in the released models. ([GitHub][2]))

[1]: https://github.com/openai/clip/blob/main/clip/model.py?utm_source=chatgpt.com "CLIP/clip/model.py at main · openai/CLIP"
[2]: https://github.com/openai/CLIP/issues/468?utm_source=chatgpt.com "Input is too long for context length 77. No truncation passed ..."
