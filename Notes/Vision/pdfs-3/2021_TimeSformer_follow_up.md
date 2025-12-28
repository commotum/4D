Distinct tasks evaluated: 5
- Action recognition on Kinetics-400, Kinetics-600, Something-Something-V2, Diving-48. Evidence: "We evaluate TimeSformer on four popular action recognition datasets: Kinetics-400 (Carreira & Zisserman, 2017), Kinetics-600 (Carreira et al., 2018), Something-Something-V2 (Goyal et al., 2017b), and Diving-48 (Li et al., 2018)." (4. Experiments)
- Long-term task classification on HowTo100M. Evidence: "Lastly, we evaluate TimeSformer on the task of long-term video modeling using HowTo100M (Miech et al., 2019)." (4.6. Long-Term Video Modeling)

Trained model instances required: 5
- Each dataset/task has its own video-class prediction head, and HowTo100M uses a separately fine-tuned model instance. Evidence: "On top of this representation we append a 1-hidden-layer MLP, which is used to predict the final video classes." (3. The TimeSformer Model) + "All models in this comparison are pretrained on Kinetics-400 before finetuning on HowTo100M." (Table 8)

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
