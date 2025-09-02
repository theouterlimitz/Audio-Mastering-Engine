# audio_mastering_engine.py (Cloud Version)
# This is the core audio processing engine, adapted to run in the cloud.
# It reads files from a GCS URI, processes them, and uploads the results.

import os
import numpy as np
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
from scipy.signal import butter, sosfilt, lfilter
import pyloudnorm as pyln
import traceback
import io
from google.cloud import storage

# --- CORE PROCESSING LOGIC ---

def process_audio_from_gcs(gcs_uri, settings):
    """
    Main cloud function entry point. Downloads, processes, and uploads a file.
    """
    try:
        storage_client = storage.Client()
        
        # --- 1. Download the file from GCS ---
        print(f"Step 1/5: Downloading {gcs_uri} from GCS...")
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        source_bucket = storage_client.bucket(bucket_name)
        source_blob = source_bucket.blob(blob_name)
        
        # Download the file content into an in-memory bytes buffer
        in_mem_file = io.BytesIO()
        source_blob.download_to_file(in_mem_file)
        in_mem_file.seek(0) # Rewind the buffer to the beginning

        print("Download complete.")
        
        # --- 2. Load into Pydub and Process ---
        print("Step 2/5: Loading audio into memory...")
        audio = AudioSegment.from_file(in_mem_file)

        # Ensure audio is stereo for processing
        if audio.channels == 1:
            audio = audio.set_channels(2)
        
        print("Starting audio processing...")
        # The same chunking logic as our local app
        chunk_size_ms = 30 * 1000
        processed_chunks = []
        num_chunks = len(range(0, len(audio), chunk_size_ms))
        
        for i, start_ms in enumerate(range(0, len(audio), chunk_size_ms)):
            print(f"  - Processing chunk {i+1}/{num_chunks}...")
            chunk = audio[start_ms:start_ms+chunk_size_ms]
            
            # Apply Analog Character
            if settings.get("analog_character", 0) > 0:
                chunk = apply_analog_character(chunk, settings.get("analog_character"))

            chunk_samples = audio_segment_to_float_array(chunk)
            processed_samples = apply_eq_to_samples(chunk_samples, chunk.frame_rate, settings)
            
            if settings.get("width", 1.0) != 1.0:
                processed_samples = apply_stereo_width(processed_samples, settings.get("width"))
                
            processed_chunk = float_array_to_audio_segment(processed_samples, chunk)
            
            if settings.get("multiband"):
                processed_chunk = apply_multiband_compressor(processed_chunk, settings)
            
            processed_chunks.append(processed_chunk)
            
        print("Assembling processed chunks...")
        processed_audio = sum(processed_chunks)
        
        final_samples = audio_segment_to_float_array(processed_audio)

        if settings.get("lufs") is not None:
            print("Normalizing loudness...")
            final_samples = normalize_to_lufs(final_samples, processed_audio.frame_rate, settings.get("lufs"))

        final_samples = soft_limiter(final_samples)
        final_audio = float_array_to_audio_segment(final_samples, processed_audio)
        print("Audio processing complete.")

        # --- 3. Upload the processed file back to GCS ---
        original_filename = settings.get('original_filename', 'unknown.wav')
        processed_filename = f"processed/mastered_{original_filename}"
        print(f"Step 3/5: Uploading processed file to {processed_filename}...")
        
        dest_blob = source_bucket.blob(processed_filename)
        
        # Export to an in-memory buffer to upload directly
        out_mem_file = io.BytesIO()
        final_audio.export(out_mem_file, format="wav", parameters=["-acodec", "pcm_s16le"])
        out_mem_file.seek(0)
        
        dest_blob.upload_from_file(out_mem_file, content_type='audio/wav')
        print("Processed file upload complete.")
        
        # --- 4. Create the ".complete" flag file ---
        print(f"Step 4/5: Creating completion flag...")
        complete_blob = source_bucket.blob(f"{processed_filename}.complete")
        complete_blob.upload_from_string("")
        print("Completion flag created.")
        
        # --- 5. Clean up the original upload ---
        print(f"Step 5/5: Deleting original raw upload: {blob_name}")
        source_blob.delete()
        print("Cleanup complete.")

    except Exception as e:
        print(f"CRITICAL ERROR in mastering engine: {e}")
        traceback.print_exc()
        # In a real app, you might move the failed file to an "error" folder.
        # For now, we just log the error.
        raise e

# --- AUDIO HELPER FUNCTIONS (Identical to local version) ---

def audio_segment_to_float_array(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
    return samples.astype(np.float32) / (2**(audio_segment.sample_width * 8 - 1))

def float_array_to_audio_segment(float_array, audio_segment_template):
    clipped_array = np.clip(float_array, -1.0, 1.0)
    int_array = (clipped_array * (2**(audio_segment_template.sample_width * 8 - 1))).astype(np.int16)
    return audio_segment_template._spawn(int_array.tobytes())

def apply_analog_character(chunk, character_percent):
    if character_percent == 0: return chunk
    saturation_amount = character_percent * 0.3 
    samples = audio_segment_to_float_array(chunk)
    saturated_samples = np.tanh(samples * (1 + saturation_amount / 100.0))
    chunk = float_array_to_audio_segment(saturated_samples, chunk)
    low_bump_db = (character_percent / 100.0) * 1.0 
    low_end = chunk.low_pass_filter(120)
    chunk = chunk.overlay(low_end.apply_gain(low_bump_db))
    high_sparkle_db = (character_percent / 100.0) * 1.5
    high_end = chunk.high_pass_filter(12000)
    chunk = chunk.overlay(high_end.apply_gain(high_sparkle_db))
    return chunk

def apply_stereo_width(samples, width_factor):
    if samples.ndim == 1 or samples.shape[1] != 2: return samples
    left, right = samples[:, 0], samples[:, 1]
    mid, side = (left + right) / 2, (left - right) / 2
    side *= width_factor
    new_left, new_right = mid + side, mid - side
    return np.array([new_left, new_right]).T

def apply_eq_to_samples(samples, sample_rate, settings):
    if samples.ndim > 1 and samples.shape[1] == 2:
        left, right = samples[:, 0], samples[:, 1]
        left = apply_shelf_filter(left, sample_rate, 250, settings.get("bass_boost", 0.0), 'low')
        right = apply_shelf_filter(right, sample_rate, 250, settings.get("bass_boost", 0.0), 'low')
        left = apply_peak_filter(left, sample_rate, 1000, -settings.get("mid_cut", 0.0))
        right = apply_peak_filter(right, sample_rate, 1000, -settings.get("mid_cut", 0.0))
        left = apply_peak_filter(left, sample_rate, 4000, settings.get("presence_boost", 0.0))
        right = apply_peak_filter(right, sample_rate, 4000, settings.get("presence_boost", 0.0))
        left = apply_shelf_filter(left, sample_rate, 8000, settings.get("treble_boost", 0.0), 'high')
        right = apply_shelf_filter(right, sample_rate, 8000, settings.get("treble_boost", 0.0), 'high')
        return np.array([left, right]).T
    else:
        samples = apply_shelf_filter(samples, sample_rate, 250, settings.get("bass_boost", 0.0), 'low')
        samples = apply_peak_filter(samples, sample_rate, 1000, -settings.get("mid_cut", 0.0))
        samples = apply_peak_filter(samples, sample_rate, 4000, settings.get("presence_boost", 0.0))
        samples = apply_shelf_filter(samples, sample_rate, 8000, settings.get("treble_boost", 0.0), 'high')
        return samples

def apply_shelf_filter(samples, sample_rate, cutoff_hz, gain_db, filter_type, order=2):
    if gain_db == 0: return samples
    gain = 10.0 ** (gain_db / 20.0)
    b, a = butter(order, cutoff_hz / (sample_rate / 2.0), btype=filter_type)
    return lfilter(b, a, samples) * gain if gain_db > 0 else lfilter(b, a, samples)

def apply_peak_filter(samples, sample_rate, center_hz, gain_db, q=1.0):
    if gain_db == 0: return samples
    nyquist = 0.5 * sample_rate
    normal_center = center_hz / nyquist
    bandwidth = normal_center / q
    low_freq, high_freq = normal_center - (bandwidth / 2), normal_center + (bandwidth / 2)
    if low_freq <= 0: low_freq = 1e-9
    if high_freq >= 1.0: high_freq = 0.999999
    sos = butter(2, [low_freq, high_freq], btype='bandpass', output='sos')
    filtered_samples = sosfilt(sos, samples)
    gain_factor = 10 ** (gain_db / 20.0)
    return samples + (filtered_samples * (gain_factor - 1))

def apply_multiband_compressor(chunk, settings, low_crossover=250, high_crossover=4000):
    samples = audio_segment_to_float_array(chunk)
    low_sos = butter(4, low_crossover, btype='lowpass', fs=chunk.frame_rate, output='sos')
    high_sos = butter(4, high_crossover, btype='highpass', fs=chunk.frame_rate, output='sos')
    low_band_samples = sosfilt(low_sos, samples, axis=0)
    high_band_samples_for_sub = sosfilt(high_sos, samples, axis=0)
    mid_band_samples = samples - low_band_samples - high_band_samples_for_sub
    high_band_samples = high_band_samples_for_sub
    low_band_chunk = float_array_to_audio_segment(low_band_samples, chunk)
    mid_band_chunk = float_array_to_audio_segment(mid_band_samples, chunk)
    high_band_chunk = float_array_to_audio_segment(high_band_samples, chunk)
    low_compressed = compress_dynamic_range(low_band_chunk,
        threshold=settings.get("low_thresh"), ratio=settings.get("low_ratio"))
    mid_compressed = compress_dynamic_range(mid_band_chunk,
        threshold=settings.get("mid_thresh"), ratio=settings.get("mid_ratio"))
    high_compressed = compress_dynamic_range(high_band_chunk,
        threshold=settings.get("high_thresh"), ratio=settings.get("high_ratio"))
    return low_compressed.overlay(mid_compressed).overlay(high_compressed)

def normalize_to_lufs(samples, sample_rate, target_lufs=-14.0):
    meter = pyln.Meter(sample_rate)
    if samples.ndim == 2:
        mono_samples = samples.mean(axis=1)
    else:
        mono_samples = samples
    if np.max(np.abs(mono_samples)) == 0:
        return samples
    loudness = meter.integrated_loudness(mono_samples)
    gain_db = target_lufs - loudness
    gain_linear = 10.0 ** (gain_db / 20.0)
    return samples * gain_linear

def soft_limiter(samples, threshold=0.98):
    clipped_indices = np.abs(samples) > threshold
    samples[clipped_indices] = np.tanh(samples[clipped_indices] / threshold) * threshold
    return samples
