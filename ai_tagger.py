# ai_tagger.py
# This module contains the logic for loading the pre-trained mood classification
# model and making predictions on new audio files.

import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
import logging

# --- Configuration ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
MODEL_PATH = 'mood_cnn_augmented_model.keras'
ENCODER_PATH = 'mood_cnn_label_encoder.joblib'

# --- Global Variables to hold the loaded models ---
# We load them once to avoid reloading on every prediction.
model = None
label_encoder = None

def load_models():
    """
    Loads the Keras model and the label encoder from disk into memory.
    """
    global model, label_encoder
    logging.info("Loading AI Tagger models...")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        logging.error("Model files not found! Ensure .keras and .joblib files are in the project directory.")
        return False
        
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        logging.info("AI Tagger models loaded successfully.")
        return True
    except Exception as e:
        logging.error(f"Error loading AI Tagger models: {e}")
        return False

def create_spectrogram_from_data(y, sr):
    """
    Creates a Mel Spectrogram from audio data and prepares it for the CNN.
    (This is copied directly from your training script).
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_HEIGHT)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    S_db_norm = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db))
    
    img = np.stack([S_db_norm]*3, axis=-1)
    resized_img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    
    return resized_img

def predict_mood(audio_file_path):
    """
    The main prediction function. Takes an audio file path, analyzes it,
    and returns the predicted mood as a string.
    """
    global model, label_encoder
    if model is None or label_encoder is None:
        if not load_models():
            return "Error: Models not loaded."

    try:
        logging.info(f"Analyzing mood for: {audio_file_path}")
        # Load the first 30 seconds for a consistent analysis
        y, sr = librosa.load(audio_file_path, mono=True, duration=30)
        
        # Create the spectrogram image
        spectrogram = create_spectrogram_from_data(y, sr)
        
        # The model expects a "batch" of images, so we add an extra dimension
        spectrogram_batch = np.expand_dims(spectrogram, axis=0)
        
        # Make the prediction
        prediction_probabilities = model.predict(spectrogram_batch)
        
        # Get the index of the highest probability
        predicted_index = np.argmax(prediction_probabilities, axis=1)[0]
        
        # Decode the index back to its string label (e.g., 'Happy/Excited')
        predicted_mood = label_encoder.inverse_transform([predicted_index])[0]
        
        logging.info(f"Predicted mood: {predicted_mood}")
        return predicted_mood
        
    except Exception as e:
        logging.error(f"Error during mood prediction: {e}")
        return "Error during analysis."

# Example usage (for testing this file directly)
if __name__ == '__main__':
    # You'll need to create a dummy audio file named 'test.wav' to run this test
    # or change the path to a real audio file on your system.
    test_file = 'test.wav' 
    if os.path.exists(test_file):
        mood = predict_mood(test_file)
        print(f"\nThe predicted mood for '{test_file}' is: {mood}")
    else:
        print(f"\nCould not find '{test_file}'. Please provide a valid audio file to test.")
        # Try loading models anyway to check if they are present
        load_models()
