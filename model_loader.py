import os
import tensorflow as tf
import numpy as np
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Load the MobileNetV2 model from the specified path
    
    Args:
        model_path: Path to the .h5 model file
        
    Returns:
        Loaded TensorFlow model
    """
    try:
        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image for model input
    
    Args:
        image: Input image (numpy array from OpenCV)
        target_size: Target size for model input (default: 224x224)
        
    Returns:
        Preprocessed image ready for model input
    """
    try:
        # Convert BGR to RGB if from OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image_resized = cv2.resize(image, target_size)
        
        # Convert to float and normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Expand dimensions to create batch
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_disease(model, preprocessed_image, class_labels):
    """
    Predict disease from preprocessed image
    
    Args:
        model: Loaded TensorFlow model
        preprocessed_image: Preprocessed image batch
        class_labels: Dictionary mapping class indices to labels
        
    Returns:
        Tuple of (predicted disease label, confidence percentage)
    """
    try:
        # Make prediction
        predictions = model.predict(preprocessed_image)
        
        # Get the predicted class index
        predicted_class_idx = np.argmax(predictions[0])
        
        # Get confidence score
        confidence = float(predictions[0][predicted_class_idx] * 100)
        
        # Convert to label
        label = class_labels.get(str(predicted_class_idx), f"Unknown class {predicted_class_idx}")
        
        return label, confidence
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise
