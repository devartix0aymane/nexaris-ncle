"""
NEXARIS Cognitive Load Estimator (NCLE)

Face Analysis Module

Handles webcam capture, face detection, and emotion recognition.
"""

import cv2
import time
import os
import csv
from datetime import datetime

# Placeholder for a real emotion recognition model/library
# e.g., DeepFace, fer, or a custom model

class EmotionRecognizer:
    """Placeholder for an emotion recognition engine."""
    def __init__(self, model_path=None):
        # In a real implementation, load a pre-trained model here
        # For example, using OpenCV's dnn module or a library like TensorFlow/PyTorch
        self.model = None # Placeholder for the actual model
        if model_path and os.path.exists(model_path):
            print(f"Loading emotion recognition model from {model_path}")
            # self.model = cv2.dnn.readNetFromONNX(model_path) # Example for ONNX
        else:
            print("Emotion recognition model not found or not specified. Using mock recognizer.")
        self.emotions = ["neutral", "happy", "sad", "angry", "surprised", "fear", "disgust"]

    def predict_emotion(self, face_roi):
        """Predict emotion from a face Region of Interest (ROI).

        Args:
            face_roi: A NumPy array representing the detected face.

        Returns:
            A dictionary of emotions and their probabilities, or a dominant emotion string.
        """
        if self.model is not None:
            # Preprocess face_roi (resize, normalize, etc.)
            # Pass to the model for inference
            # Postprocess the output to get emotion probabilities
            # Example: (this is highly dependent on the model)
            # blob = cv2.dnn.blobFromImage(face_roi, 1.0/255, (64, 64), (0,0,0), swapRB=True, crop=False)
            # self.model.setInput(blob)
            # preds = self.model.forward()
            # dominant_emotion_idx = preds[0].argmax()
            # return self.emotions[dominant_emotion_idx]
            pass # Replace with actual model inference

        # Mock implementation if no model is loaded
        import random
        return random.choice(self.emotions) 

class FacialAnalyzer:
    """Analyzes facial expressions from a webcam feed."""
    def __init__(self, camera_index=0, log_directory="../logs/", cascade_path=None):
        self.camera_index = camera_index
        self.video_capture = None
        self.face_cascade = None
        self.emotion_recognizer = EmotionRecognizer() # Add model_path if available
        self.active = False
        self.current_task_id = "general"

        self.log_directory = log_directory
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.emotion_log_path = os.path.join(self.log_directory, f"emotion_log_{self.session_id}.csv")
        self._setup_logging()

        # Load Haar cascade for face detection
        # A common path could be: cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        default_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if cascade_path and os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        elif os.path.exists(default_cascade_path):
            self.face_cascade = cv2.CascadeClassifier(default_cascade_path)
        else:
            print("Error: Face detection cascade file not found.")
            # Fallback or error handling needed if cascade is essential

    def _setup_logging(self):
        """Create log directory and initialize CSV log file with headers."""
        os.makedirs(self.log_directory, exist_ok=True)
        with open(self.emotion_log_path, 'w', newline='') as f_emotion:
            emotion_writer = csv.writer(f_emotion)
            emotion_writer.writerow(["timestamp", "detected_emotion", "face_detected", "task_id"])
        print(f"Logging emotions to {self.log_directory}")

    def start_analysis(self, task_id="general"):
        """Start facial analysis."""
        if not self.face_cascade:
            print("Cannot start analysis: Face cascade not loaded.")
            return False
        
        self.video_capture = cv2.VideoCapture(self.camera_index)
        if not self.video_capture.isOpened():
            print(f"Error: Could not open camera {self.camera_index}.")
            self.video_capture = None
            return False
        
        self.active = True
        self.current_task_id = task_id
        print(f"Facial analysis started for task: {task_id} on camera {self.camera_index}.")
        return True

    def stop_analysis(self):
        """Stop facial analysis."""
        self.active = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        print("Facial analysis stopped.")

    def process_frame(self):
        """Process a single frame from the webcam."""
        if not self.active or not self.video_capture or not self.video_capture.isOpened():
            return None, None # Frame, Emotion

        ret, frame = self.video_capture.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            return None, None

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )

        detected_emotion = None
        face_detected_in_frame = False

        for (x, y, w, h) in faces:
            face_detected_in_frame = True
            face_roi = gray_frame[y:y+h, x:x+w] # Emotion model might prefer color: frame[y:y+h, x:x+w]
            
            # Get emotion prediction
            detected_emotion = self.emotion_recognizer.predict_emotion(face_roi)
            
            # Draw rectangle around the face and put emotion text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
            break # Process first detected face for simplicity
        
        # Log emotion data
        timestamp = time.time()
        with open(self.emotion_log_path, 'a', newline='') as f_emotion:
            emotion_writer = csv.writer(f_emotion)
            emotion_writer.writerow([timestamp, detected_emotion if detected_emotion else "N/A", face_detected_in_frame, self.current_task_id])

        return frame, detected_emotion

    def __del__(self):
        self.stop_analysis() # Ensure camera is released when object is deleted

if __name__ == '__main__':
    # Example Usage
    analyzer = FacialAnalyzer(log_directory="../../logs/") # Adjusted path

    if analyzer.start_analysis(task_id="test_emotion_task_001"):
        print("Processing frames... Press 'q' to quit.")
        while True:
            frame, emotion = analyzer.process_frame()
            if frame is not None:
                cv2.imshow('Facial Analysis - NCLE', frame)
                if emotion:
                    print(f"Detected emotion: {emotion}")
            else:
                # Handle case where frame could not be read or analysis not active
                if not analyzer.active:
                    break # Exit if analysis was stopped
                time.sleep(0.1) # Wait a bit if no frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        analyzer.stop_analysis()
        cv2.destroyAllWindows()
        print(f"Emotion logs saved to: {os.path.abspath(analyzer.emotion_log_path)}")
    else:
        print("Could not start facial analyzer. Check camera and cascade file.")