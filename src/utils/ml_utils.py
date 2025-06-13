#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Learning utilities for NEXARIS Cognitive Load Estimator

This module provides functions for loading, using, and managing ML models
for cognitive load estimation.
"""

import os
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

# Optional imports - will be imported only when needed
_sklearn_available = False
_tensorflow_available = False

# Logger for this module
logger = logging.getLogger(__name__)


def check_ml_dependencies() -> Dict[str, bool]:
    """
    Check if ML dependencies are available
    
    Returns:
        Dictionary with availability status of each dependency
    """
    global _sklearn_available, _tensorflow_available
    
    dependencies = {}
    
    # Check for scikit-learn
    try:
        import sklearn
        _sklearn_available = True
        dependencies['sklearn'] = True
        dependencies['sklearn_version'] = sklearn.__version__
    except ImportError:
        _sklearn_available = False
        dependencies['sklearn'] = False
    
    # Check for TensorFlow
    try:
        import tensorflow as tf
        _tensorflow_available = True
        dependencies['tensorflow'] = True
        dependencies['tensorflow_version'] = tf.__version__
    except ImportError:
        _tensorflow_available = False
        dependencies['tensorflow'] = False
    
    return dependencies


def load_model(model_path: str) -> Any:
    """
    Load a machine learning model from file
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model object
    
    Raises:
        ImportError: If required dependencies are not available
        FileNotFoundError: If model file doesn't exist
        ValueError: If model format is not supported
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine model type from file extension
    if model_path.suffix == '.pkl':
        # Load scikit-learn or pickle-compatible model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded pickle model from {model_path}")
        return model
    
    elif model_path.suffix == '.h5' or model_path.name == 'saved_model.pb' or model_path.is_dir():
        # Load TensorFlow/Keras model
        if not _tensorflow_available:
            check_ml_dependencies()  # Try to import again
            if not _tensorflow_available:
                raise ImportError("TensorFlow is required to load this model type")
        
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_path))
        logger.info(f"Loaded TensorFlow model from {model_path}")
        return model
    
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")


def preprocess_features(
    features: Dict[str, Any], 
    feature_config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Preprocess features for model input
    
    Args:
        features: Dictionary of raw features
        feature_config: Configuration for feature preprocessing
        
    Returns:
        Numpy array of preprocessed features ready for model input
    """
    if feature_config is None:
        feature_config = {}
    
    # Default feature order and scaling
    default_feature_order = [
        'response_time', 'mouse_distance', 'click_count', 
        'hesitation_time', 'emotion_score'
    ]
    
    # Get feature order from config or use default
    feature_order = feature_config.get('feature_order', default_feature_order)
    
    # Extract features in the correct order
    feature_values = []
    for feature_name in feature_order:
        if feature_name in features:
            feature_values.append(features[feature_name])
        else:
            # Use default value (0) for missing features
            logger.warning(f"Missing feature: {feature_name}, using default value 0")
            feature_values.append(0)
    
    # Convert to numpy array
    X = np.array(feature_values, dtype=np.float32).reshape(1, -1)
    
    # Apply scaling if specified in config
    scaling = feature_config.get('scaling', None)
    if scaling == 'standard':
        # Apply standardization (z-score normalization)
        means = np.array(feature_config.get('means', [0] * len(feature_order)))
        stds = np.array(feature_config.get('stds', [1] * len(feature_order)))
        X = (X - means) / stds
    elif scaling == 'minmax':
        # Apply min-max scaling
        mins = np.array(feature_config.get('mins', [0] * len(feature_order)))
        maxs = np.array(feature_config.get('maxs', [1] * len(feature_order)))
        X = (X - mins) / (maxs - mins)
    
    return X


def predict_cognitive_load(
    model: Any, 
    features: Dict[str, Any],
    feature_config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Predict cognitive load using a machine learning model
    
    Args:
        model: Loaded ML model
        features: Dictionary of behavioral and physiological features
        feature_config: Configuration for feature preprocessing
        
    Returns:
        Predicted cognitive load score (0-100)
    """
    # Preprocess features
    X = preprocess_features(features, feature_config)
    
    # Make prediction
    try:
        # Handle different model types
        if hasattr(model, 'predict'):
            # scikit-learn style prediction
            prediction = model.predict(X)[0]
        elif hasattr(model, '__call__'):
            # TensorFlow/Keras style prediction
            prediction = model(X).numpy()[0][0]
        else:
            raise ValueError("Unsupported model type")
        
        # Scale prediction to 0-100 range if needed
        if prediction < 0 or prediction > 100:
            prediction = max(0, min(100, prediction))
        
        return float(prediction)
    
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        # Return a default score if prediction fails
        return 50.0


def train_simple_model(X: np.ndarray, y: np.ndarray) -> Any:
    """
    Train a simple regression model for cognitive load estimation
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target values (cognitive load scores)
        
    Returns:
        Trained model
    
    Raises:
        ImportError: If scikit-learn is not available
    """
    if not _sklearn_available:
        check_ml_dependencies()  # Try to import again
        if not _sklearn_available:
            raise ImportError("scikit-learn is required for model training")
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    logger.info(f"Model trained. Validation MSE: {mse:.4f}, R²: {r2:.4f}")
    
    return model


def save_model(model: Any, model_path: str) -> bool:
    """
    Save a machine learning model to file
    
    Args:
        model: Model to save
        model_path: Path to save the model
        
    Returns:
        True if successful, False otherwise
    """
    model_path = Path(model_path)
    
    # Create directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Determine model type and use appropriate saving method
        if hasattr(model, 'save') and callable(model.save):
            # TensorFlow/Keras model
            model.save(str(model_path))
        else:
            # Default to pickle for scikit-learn and other models
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Model saved to {model_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


# Initialize by checking dependencies
check_ml_dependencies()