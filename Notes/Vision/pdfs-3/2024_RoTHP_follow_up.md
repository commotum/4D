1. Number of distinct tasks evaluated: 2.
   - Next-event prediction (event type + timestamp). Evidence: "For the prediction of next event type and timestamp, we train two linear layers W e , W t" and the associated event/time prediction losses (2024_RoTHP.txt:391-406).
   - Future event prediction (train on past, predict future). Evidence: "we use the previous information to predict the future ones" and define S[1:m] as the training sample and S[m+1:n] as the testing sample (2024_RoTHP.txt:476-480), plus the future-prediction experiment section (2024_RoTHP.txt:664-666).

2. Number of trained model instances required to cover all tasks: 1.
   - Rationale: The model defines the event-type and time prediction heads once (2024_RoTHP.txt:391-406), and future prediction is evaluated via a different train/test split on the same sequences (S[1:m] training, S[m+1:n] testing) without any new head or architecture (2024_RoTHP.txt:476-480). Thus a single trained model instance (per dataset) suffices to support both tasks.

3. Task-Model Ratio:

$$
\boxed{
\frac{2\ \text{tasks}}{1\ \text{model}} = 2
}
$$
