1. Number of distinct tasks evaluated: 4
- Text-to-video prediction (VideoBERT generates video tokens from recipe text) (2019_VideoBERT.pdf, Figure 1 caption: "Given some recipe text divided into sentences... we generate a sequence of video tokens... using VideoBERT.")
- Future forecasting / video-to-video prediction (VideoBERT forecasts future video tokens) (2019_VideoBERT.pdf, Figure 1 caption: "Given a video token, we show the top three future tokens forecasted by VideoBERT.")
- Zero-shot action classification (verb/noun prediction from video tokens) (2019_VideoBERT.pdf, Section 4.4: "Once pretrained, the VideoBERT model can be used for 'zero-shot' classification on novel datasets, such as YouCook II.")
- Video captioning (dense video captioning) (2019_VideoBERT.pdf, Section 4.6: "We evaluate the extracted features on video captioning... train a supervised model mapping video segments to captions.")

2. Number of trained model instances required to cover all tasks: 2
- Model 1: The pretrained VideoBERT model used directly for text-to-video prediction and future forecasting (2019_VideoBERT.pdf, Figure 1 caption: "we generate a sequence of video tokens... using VideoBERT"; "future tokens forecasted by VideoBERT"), and for zero-shot action classification without task-specific fine-tuning (2019_VideoBERT.pdf, Section 4.4: "Once pretrained, the VideoBERT model can be used for 'zero-shot' classification...").
- Model 2: A task-specific supervised captioning model (transformer encoder-decoder) trained on YouCook II using VideoBERT features, which counts as a separate trained instance (2019_VideoBERT.pdf, Section 4.6: "train a supervised model mapping video segments to captions"; "We use the same model that they do, namely a transformer encoder-decoder, but we replace the inputs to the encoder with the features derived from VideoBERT...").

3. Task-Model Ratio:

$$
\boxed{
\frac{4\ \text{tasks}}{2\ \text{models}} = 2
}
$$
