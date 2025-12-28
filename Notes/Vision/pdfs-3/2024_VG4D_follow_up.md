1. Number of distinct tasks evaluated: 2.
   - 4D point cloud action recognition (single-modality). Evidence: "Experiments demonstrate that our method achieves state-of-the-art performance for action recognition on both the NTU RGB+D 60 dataset and the NTU RGB+D 120 dataset." (Abstract) and the evaluation of "single modal baseline methods" for point cloud recognition in the experiments (Table I; "Our im-PSTNet outperforms other single modal baseline methods").
   - Multi-modal action recognition (RGB + point cloud + text). Evidence: "we synergize the exceptional capabilities of Vision-Language Models (VLMs) in video understanding with 4D point cloud representation to enhance multi-modal action recognition" and "we achieve robust multi-modal action recognition by integrating multi-modal prediction scores and utilizing text information as classifiers." (Introduction). The VG4D setup explicitly uses multiple modalities: "VG4D framework consists of 3 networks: 4D point cloud encoder EP, video encoder EV and text encoder ET from VLM." (Method).

2. Number of trained model instances required to cover all tasks: 2.
   - 4D point cloud action recognition uses a single-modal model (im-PSTNet with a classification head), which is evaluated as a point-cloud-only system: "Our im-PSTNet outperforms other single modal baseline methods" and Table I lists "Uni-modal recognition methods" under Point Cloud. (Experiments; Table I).
   - Multi-modal action recognition uses the VG4D framework with additional modalities and fusion beyond the single-modal model: "VG4D framework consists of 3 networks: 4D point cloud encoder EP, video encoder EV and text encoder ET from VLM" and "we fuse four 4D-text, RGB-text, 4D, and RGB scores as the final classification result." (Method). This requires extra modality-specific components and outputs beyond the single-modal model, so it is a separate trained model instance for the multi-modal task.

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
