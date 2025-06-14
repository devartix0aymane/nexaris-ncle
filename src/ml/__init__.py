# Machine Learning Package
# This package will contain modules for machine learning models, training, and inference
# related to cognitive load estimation and other analyses.

# Example imports:
# from .models.cognitive_model import CognitiveModelTrainer, CognitiveModelPredictor
# from .training.trainer import train_model_on_session_data

# Configuration for ML module could be loaded here or passed from main config
ML_CONFIG = {
    "model_path": "data/models/cognitive_load_v1.pkl", # Example path
    "default_features": [
        "heart_rate_variability", 
        "skin_conductance_level", 
        "pupil_diameter_mean",
        "blink_rate",
        "emotion_neutral_confidence",
        "task_difficulty",
        "hesitation_rate"
    ]
}

def get_ml_config():
    return ML_CONFIG