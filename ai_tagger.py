# ai_tagger.py (v2.0 - Feature Branch)
# This version adds the ability to save a visualization of the
# spectrogram to a file, to be used as a creative input.

import os
import numpy as np
import joblib
import librosa
import librosa.display # <-- Needed for visualizing
import matplotlib.pyplot as plt # <-- The plotting library
import tensorflow as tf
from PIL import Image # For resizing

# --- Configuration ---
# We define these as constants so they're easy to change.
MODEL_PATH = 'mood_cnn_augmented_model.keras'
ENCODER_PATH = 'mood_cnn_label_encoder.joblib'
IMG_HEIGHT = 128
IMG_WIDTH = 128

# --- Private Helper Functions ---

def _load_models():
    """Loads the ML model and label encoder, handles errors."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print(f"ERROR: Model '{MODEL_PATH}' or '{ENCODER_PATH}' not found.")
        return None, None
    try:
        print("Loading AI Tagger models...")
        model = tf.keras.models.load_model(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        print("AI Tagger models loaded successfully.")
        return model, label_encoder
    except Exception as e:
        print(f"ERROR loading models: {e}")
        return None, None

def _create_spectrogram_for_model(y, sr):
    """Creates a spectrogram suitable for model prediction."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_HEIGHT)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db_norm = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db))
    img = np.stack([S_db_norm]*3, axis=-1)
    resized_img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return resized_img

# --- NEW PUBLIC FUNCTION ---
def predict_mood_and_save_spectrogram(audio_file_path):
    """
    Analyzes an audio file, predicts its mood, and saves a visualization of its spectrogram.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        tuple: A tuple containing (predicted_mood, path_to_spectrogram_image).
               Returns (error_message, None) on failure.
    """
    model, label_encoder = _load_models()
    if not model or not label_encoder:
        return "Error: Could not load AI models.", None

    print(f"Analyzing mood for: {audio_file_path}")
    try:
        y, sr = librosa.load(audio_file_path, mono=True, duration=30)
        
        # 1. Create spectrogram for the model
        spectrogram_for_model = _create_spectrogram_for_model(y, sr)
        
        # 2. Make the prediction
        spectrogram_batch = np.expand_dims(spectrogram_for_model, axis=0)
        prediction = model.predict(spectrogram_batch)
        predicted_index = np.argmax(prediction)
        predicted_mood = label_encoder.inverse_transform([predicted_index])[0]
        print(f"Predicted mood: {predicted_mood}")

        # 3. Create and save a high-quality spectrogram for visualization
        plt.figure(figsize=(8, 6))
        S_full = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db_full = librosa.power_to_db(S_full, ref=np.max)
        librosa.display.specshow(S_db_full, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off') # Remove axes for a cleaner image
        plt.tight_layout(pad=0)
        
        # Save to a temporary file
        spectrogram_path = os.path.join(tempfile.gettempdir(), "spectrogram.png")
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close() # Close the figure to free memory

        print(f"Spectrogram saved to: {spectrogram_path}")
        return predicted_mood, spectrogram_path

    except Exception as e:
        print(f"ERROR during mood prediction: {e}")
        return f"Error: {e}", None