# ai_tagger.py (v3.0 - The "Musicologist")
# This version upgrades the module from a simple mood predictor to a
# full-fledged musicologist that extracts a rich technical brief from the audio.

import os
import tempfile
import numpy as np
import joblib
import librosa
import librosa.display
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# --- Configuration ---
MODEL_PATH = 'mood_cnn_augmented_model.keras'
ENCODER_PATH = 'mood_cnn_label_encoder.joblib'
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Cached models to avoid reloading on every call
_model = None
_label_encoder = None

def _load_models():
    """Loads the ML model and label encoder, caching them for efficiency."""
    global _model, _label_encoder
    if _model and _label_encoder:
        return _model, _label_encoder

    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        logging.error(f"Model '{MODEL_PATH}' or '{ENCODER_PATH}' not found.")
        return None, None
    try:
        logging.info("Loading AI Tagger models for the first time...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        _label_encoder = joblib.load(ENCODER_PATH)
        logging.info("AI Tagger models loaded and cached successfully.")
        return _model, _label_encoder
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

# --- THIS IS THE NEW, PRIMARY FUNCTION ---
def analyze_song(audio_file_path):
    """
    Acts as a "Musicologist" to generate a full technical brief for a song.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        dict: A dictionary containing the song's analysis (mood, tempo, etc.),
              or an error dictionary on failure.
    """
    model, label_encoder = _load_models()
    if not model or not label_encoder:
        return {"error": "Could not load AI models."}

    logging.info(f"Analyzing song: {audio_file_path}")
    try:
        # Load the first 30 seconds for efficient analysis
        y, sr = librosa.load(audio_file_path, mono=True, duration=30)
        
        # 1. Predict Mood (existing logic)
        spectrogram_for_model = _create_spectrogram_for_model(y, sr)
        spectrogram_batch = np.expand_dims(spectrogram_for_model, axis=0)
        prediction = model.predict(spectrogram_batch)
        predicted_index = np.argmax(prediction)
        predicted_mood = label_encoder.inverse_transform([predicted_index])[0]
        logging.info(f"Predicted mood: {predicted_mood}")
        
        # 2. Extract Technical Features (new logic)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))
        
        # 3. Classify features into descriptive terms
        tempo_class = "fast" if tempo > 120 else "moderate" if tempo > 90 else "slow"
        brightness_class = "bright" if spectral_centroid > 2000 else "warm" if spectral_centroid > 1000 else "dark"
        density_class = "dense" if rms > 0.1 else "moderate" if rms > 0.05 else "sparse"

        # 4. Return the complete technical brief
        tech_brief = {
            "mood": predicted_mood,
            "tempo": f"{tempo:.0f} BPM ({tempo_class})",
            "brightness": brightness_class,
            "density": density_class
        }
        logging.info(f"Generated Technical Brief: {tech_brief}")
        return tech_brief

    except Exception as e:
        logging.exception("ERROR during song analysis")
        return {"error": str(e)}



    
    

