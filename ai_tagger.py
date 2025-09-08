# ai_tagger.py (v2.1 - Final)
# This version fixes two critical bugs:
# 1. Imports the missing 'tempfile' module.
# 2. Sets the Matplotlib backend to 'Agg' to prevent GUI conflicts
#    when running in a background thread.

import os
import tempfile # <-- FIX #1: Import the missing module
import numpy as np
import joblib
import librosa
import librosa.display

# --- FIX #2: Set the Matplotlib backend ---
# This must be done BEFORE pyplot is imported
import matplotlib
matplotlib.use('Agg')
# --- END FIX ---

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# --- Configuration ---
MODEL_PATH = 'mood_cnn_augmented_model.keras'
ENCODER_PATH = 'mood_cnn_label_encoder.joblib'
IMG_HEIGHT = 128
IMG_WIDTH = 128


def _load_models():
    """Loads the ML model and label encoder, handles errors."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        logging.error(f"Model '{MODEL_PATH}' or '{ENCODER_PATH}' not found.")
        return None, None
    try:
        logging.info("Loading AI Tagger models...")
        model = tf.keras.models.load_model(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        logging.info("AI Tagger models loaded successfully.")
        return model, label_encoder
    except Exception as e:
        logging.exception("ERROR loading models")
        return None, None


def _create_spectrogram_for_model(y, sr):
    """Creates a spectrogram suitable for model prediction."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_HEIGHT)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db_norm = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db))
    img = np.stack([S_db_norm]*3, axis=-1)
    resized_img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return resized_img


def predict_mood_and_save_spectrogram(audio_file_path):
    """
    Analyzes an audio file, predicts its mood, and saves a visualization of its spectrogram.
    """
    model, label_encoder = _load_models()
    if not model or not label_encoder:
        return "Error: Could not load AI models.", None

    logging.info(f"Analyzing mood for: {audio_file_path}")
    try:
        y, sr = librosa.load(audio_file_path, mono=True, duration=30)
        
        spectrogram_for_model = _create_spectrogram_for_model(y, sr)
        
        spectrogram_batch = np.expand_dims(spectrogram_for_model, axis=0)
        prediction = model.predict(spectrogram_batch)
        predicted_index = np.argmax(prediction)
        predicted_mood = label_encoder.inverse_transform([predicted_index])[0]
        logging.info(f"Predicted mood: {predicted_mood}")

        plt.figure(figsize=(8, 6))
        S_full = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db_full = librosa.power_to_db(S_full, ref=np.max)
        librosa.display.specshow(S_db_full, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        spectrogram_path = os.path.join(tempfile.gettempdir(), "spectrogram.png")
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        logging.info(f"Spectrogram saved to: {spectrogram_path}")
        return predicted_mood, spectrogram_path

    except Exception as e:
        logging.exception("ERROR during mood prediction")
        return f"Error: {e}", None

