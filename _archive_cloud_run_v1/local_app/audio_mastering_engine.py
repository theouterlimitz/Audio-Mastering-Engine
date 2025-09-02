# audio_mastering_engine.py (v2.3 - Local Desktop Version)
#
# This version contains two critical bug fixes:
# 1. The multiband compressor is re-architected to be phase-coherent, solving the "horrible sound" issue.
# 2. The WAV export is standardized to 16-bit CD quality, solving the massive file size issue.

import os
import numpy as np
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
from scipy.signal import butter, sosfilt, lfilter
import pyloudnorm as pyln
import traceback

# --- PRESET DEFINITIONS ---
EQ_PRESETS = {
    "techno": { "bass_boost": 4.0, "mid_cut": 3.0, "presence_boost": 1.0, "treble_boost": 3.0 },
    "dubstep": { "bass_boost": 5.0, "mid_cut": 4.0, "presence_boost": 2.0, "treble_boost": 3.5 },
    "pop": { "bass_boost": 2.0, "mid_cut": 0.0, "presence_boost": 3.5, "treble_boost": 2.5 },
    "rock": { "bass_boost": 1.5, "mid_cut": -2.0, "presence_boost": 2.5, "treble_boost": 1.0 }
}

# --- CORE PROCESSING LOGIC ---

def process_audio(settings, status_callback=None, progress_callback=None):
    """
    The main audio processing function for the local GUI.
    Takes a dictionary of settings and callbacks for status and progress updates.
    """
    try:
        input_file = settings.get("input_file")
        output_file = settings.get("output_file")

        if not input_file or not output_file:
            if status_callback: status_callback("Error: Input or output file not specified.")
            return

        if not os.path.exists(input_file):
            if status_callback: status_callback(f"Error: Input file not found at '{input_file}'")
            return

        if status_callback: status_callback(f"Loading audio file: {input_file}")
        audio = AudioSegment.from_file(input_file)

        # Ensure audio is stereo for processing
        if audio.channels == 1:
            audio = audio.set_channels(2)
        
        chunk_size_ms = 30 * 1000
        processed_chunks = []
        
        num_chunks = len(range(0, len(audio), chunk_size_ms))
        
        if status_callback: status_callback("Processing audio...")
        for i, start_ms in enumerate(range(0, len(audio), chunk_size_ms)):
            chunk = audio[start_ms:start_ms+chunk_size_ms]
            
            # Apply Analog Character
            if settings.get("analog_character", 0) > 0:
                chunk = apply_analog_character(chunk, settings.get("analog_character"))

            # Convert to float array for SciPy processing
            chunk_samples = audio_segment_to_float_array(chunk)
            
            # Apply EQ
            processed_samples = apply_eq_to_samples(chunk_samples, chunk.frame_rate, settings)
            
            # Apply Stereo Width
            if settings.get("width", 1.0) != 1.0:
                processed_samples = apply_stereo_width(processed_samples, settings.get("width"))
                
            processed_chunk = float_array_to_audio_segment(processed_samples, chunk)
            
            # Apply Compressor
            if settings.get("multiband"):
                processed_chunk = apply_multiband_compressor(processed_chunk, settings)
            
            processed_chunks.append(processed_chunk)
            if progress_callback: progress_callback(i + 1, num_chunks)
            
        if status_callback: status_callback("Assembling processed chunks...")
        processed_audio = sum(processed_chunks)
        
        final_samples = audio_segment_to_float_array(processed_audio)

        if settings.get("lufs") is not None:
            if status_callback: status_callback("Normalizing loudness...")
            final_samples = normalize_to_lufs(final_samples, processed_audio.frame_rate, settings.get("lufs"), status_callback)

        final_samples = soft_limiter(final_samples)
        final_audio = float_array_to_audio_segment(final_samples, processed_audio)

        if status_callback: status_callback(f"Exporting processed audio to: {output_file}")
        
        # --- FIX for File Size: Export as 16-bit WAV ---
        output_format = os.path.splitext(output_file)[1][1:].lower() or "wav"
        if output_format == "wav":
            # Set sample width to 2 bytes for 16-bit audio
            final_audio.export(output_file, format="wav", parameters=["-acodec", "pcm_s16le"])
        else:
            final_audio.export(output_file, format=output_format)

        if status_callback: status_callback("Processing complete!")

    except Exception as e:
        if status_callback: status_callback(f"An unexpected error occurred: {e}")
        traceback.print_exc()

# --- AUDIO HELPER FUNCTIONS ---

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
    if character_percent == 0:
        return chunk

    # 1. Saturation
    saturation_amount = character_percent * 0.3 
    samples = audio_segment_to_float_array(chunk)
    saturated_samples = np.tanh(samples * (1 + saturation_amount / 100.0))
    chunk = float_array_to_audio_segment(saturated_samples, chunk)

    # 2. Low-end Bump
    low_bump_db = (character_percent / 100.0) * 1.0 
    low_end = chunk.low_pass_filter(120)
    chunk = chunk.overlay(low_end.apply_gain(low_bump_db))

    # 3. High-end Sparkle
    high_sparkle_db = (character_percent / 100.0) * 1.5
    high_end = chunk.high_pass_filter(12000)
    chunk = chunk.overlay(high_end.apply_gain(high_sparkle_db))

    return chunk

def apply_stereo_width(samples, width_factor):
    if samples.ndim == 1 or samples.shape[1] != 2: return samples
    left, right = samples[:, 0], samples[:, 1]
    mid = (left + right) / 2
    side = (left - right) / 2
    side *= width_factor
    new_left = mid + side
    new_right = mid - side
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
    low_freq = normal_center - (bandwidth / 2)
    high_freq = normal_center + (bandwidth / 2)
    if low_freq <= 0: low_freq = 1e-9
    if high_freq >= 1.0: high_freq = 0.999999
    sos = butter(2, [low_freq, high_freq], btype='bandpass', output='sos')
    filtered_samples = sosfilt(sos, samples)
    gain_factor = 10 ** (gain_db / 20.0)
    return samples + (filtered_samples * (gain_factor - 1))

def apply_multiband_compressor(chunk, settings, low_crossover=250, high_crossover=4000):
    # --- FIX for "Horrible Sound": Use phase-coherent crossover ---
    # This is a more advanced technique that prevents phase cancellation.
    samples = audio_segment_to_float_array(chunk)
    
    # Create low-pass and high-pass filters
    low_sos = butter(4, low_crossover, btype='lowpass', fs=chunk.frame_rate, output='sos')
    high_sos = butter(4, high_crossover, btype='highpass', fs=chunk.frame_rate, output='sos')
    
    # Apply filters to get the bands
    low_band_samples = sosfilt(low_sos, samples, axis=0)
    
    # To get the mid-band, we subtract the low and high from the original
    high_band_samples_for_sub = sosfilt(high_sos, samples, axis=0)
    mid_band_samples = samples - low_band_samples - high_band_samples_for_sub
    
    # The final high band is the same
    high_band_samples = high_band_samples_for_sub

    # Convert back to AudioSegment to use pydub's compressor
    low_band_chunk = float_array_to_audio_segment(low_band_samples, chunk)
    mid_band_chunk = float_array_to_audio_segment(mid_band_samples, chunk)
    high_band_chunk = float_array_to_audio_segment(high_band_samples, chunk)
    
    low_compressed = compress_dynamic_range(low_band_chunk,
        threshold=settings.get("low_thresh"), ratio=settings.get("low_ratio"))
    
    mid_compressed = compress_dynamic_range(mid_band_chunk,
        threshold=settings.get("mid_thresh"), ratio=settings.get("mid_ratio"))
        
    high_compressed = compress_dynamic_range(high_band_chunk,
        threshold=settings.get("high_thresh"), ratio=settings.get("high_ratio"))
        
    # Recombine the bands
    return low_compressed.overlay(mid_compressed).overlay(high_compressed)

def normalize_to_lufs(samples, sample_rate, target_lufs=-14.0, status_callback=None):
    meter = pyln.Meter(sample_rate)
    if samples.ndim == 2:
        mono_samples = samples.mean(axis=1)
    else:
        mono_samples = samples
    
    # Prevent error on silent chunks
    if np.max(np.abs(mono_samples)) == 0:
        return samples

    loudness = meter.integrated_loudness(mono_samples)
    gain_db = target_lufs - loudness
    gain_linear = 10.0 ** (gain_db / 20.0)
    if status_callback: status_callback(f"Current loudness: {loudness:.2f} LUFS. Applying {gain_db:.2f} dB gain...")
    return samples * gain_linear

def soft_limiter(samples, threshold=0.98):
    clipped_indices = np.abs(samples) > threshold
    samples[clipped_indices] = np.tanh(samples[clipped_indices] / threshold) * threshold
    return samples
