# NEXARIS Model Directory

This directory contains models used by the NEXARIS Cognitive Load Estimator for various analysis tasks.

## Model Types

### Face Detection Models

- `haarcascade_frontalface_default.xml`: Haar cascade classifier for frontal face detection using OpenCV
- `face_detection_model.pb`: Optional DNN-based face detection model for improved accuracy

### Emotion Recognition Models

- `emotion_model.onnx`: ONNX format model for facial emotion recognition

### Cognitive Load Estimation Models

- `cognitive_load_model.pkl`: Machine learning model for cognitive load estimation based on behavioral and facial features

## Model Sources

- Haar cascade models can be obtained from OpenCV's GitHub repository: https://github.com/opencv/opencv/tree/master/data/haarcascades
- Emotion recognition models can be trained using frameworks like TensorFlow or PyTorch and converted to ONNX format
- Cognitive load models are trained using the application's built-in training functionality

## Adding Custom Models

To add a custom model:

1. Place the model file in this directory
2. Update the application settings to point to your custom model
3. Ensure the model format is compatible with the application (ONNX, Pickle, TensorFlow SavedModel, etc.)

## Model Performance Considerations

- Haar cascade models are faster but less accurate than DNN models
- ONNX models provide good performance across different platforms
- Consider model size and inference speed when selecting models for real-time analysis