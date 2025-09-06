# audio_mastering_engine.py (Final Architecture with Race Condition Fix)
# This version adds a small time.sleep() delay after uploading the AI art
# but before creating the .complete flag. This resolves a classic race condition
# in distributed storage systems.

import os
import tempfile
import numpy as np
import subprocess
import json
import logging
import psutil
import time # <--- Import the time library
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
from scipy.signal import butter, sosfilt, lfilter
import traceback

# GCP Libraries
from google.cloud import storage
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# --- All code is the same until the very end of the main function ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"MEMORY USAGE: {mem_info.rss / 1024 ** 2:.2f} MB")

def normalize_loudness_on_disk_with_ffmpeg(input_path, output_path, target_lufs=-14.0):
    logging.info(f"Starting true disk-based loudness normalization for {input_path}...")
    try:
        command_pass1 = [
            'ffmpeg', '-i', input_path,
            '-af', f'loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json',
            '-f', 'null', '-'
        ]
        result_pass1 = subprocess.run(command_pass1, capture_output=True, text=True)
        
        output_lines = result_pass1.stderr.splitlines()
        json_str = ""
        json_started = False
        for line in output_lines:
            if line.strip().startswith('{'): json_started = True
            if json_started: json_str += line
            if line.strip().endswith('}'): break
        
        if not json_str:
            raise RuntimeError("Could not parse loudnorm stats from ffmpeg's first pass.")

        measured_stats = json.loads(json_str)
        logging.info(f"FFmpeg Pass 1 stats: {measured_stats}")

        command_pass2 = [
            'ffmpeg', '-i', input_path,
            '-af', f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
                   f"measured_I={measured_stats['input_i']}:"
                   f"measured_LRA={measured_stats['input_lra']}:"
                   f"measured_TP={measured_stats['input_tp']}:"
                   f"measured_thresh={measured_stats['input_thresh']}:"
                   f"offset={measured_stats['target_offset']}",
            '-y', output_path
        ]
        
        subprocess.run(command_pass2, check=True)
        logging.info(f"FFmpeg Pass 2 complete. Normalized file at {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logging.exception("CRITICAL FFMPEG ERROR during disk-based normalization.")
        logging.error(f"FFMPEG STDERR:\n{e.stderr}")
        subprocess.run(['cp', input_path, output_path], check=True)
        return output_path
    except Exception:
        logging.exception("CRITICAL UNKNOWN ERROR during disk-based normalization. Falling back.")
        subprocess.run(['cp', input_path, output_path], check=True)
        return output_path


def process_audio_with_ffmpeg_pipeline(settings):
    input_file = settings.get("input_file")
    output_file = settings.get("output_file")
    if not input_file or not output_file:
        raise ValueError("Input or output file not specified.")

    logging.info(f"STARTING FFmpeg pipeline for {input_file}")
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info("Splitting large audio file into chunks on disk...")
        chunk_duration_sec = 30
        split_command = [
            'ffmpeg', '-i', input_file, '-f', 'segment',
            '-segment_time', str(chunk_duration_sec),
            '-c', 'copy', os.path.join(temp_dir, 'input_chunk_%04d.wav')
        ]
        subprocess.run(split_command, check=True)
        logging.info("Splitting complete.")

        input_chunk_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('input_chunk_')])
        processed_chunk_files = []
        num_chunks = len(input_chunk_files)
        logging.info(f"Found {num_chunks} chunks to process.")

        for i, chunk_filename in enumerate(input_chunk_files):
            try:
                logging.info(f"Processing chunk {i+1}/{num_chunks}: {chunk_filename}")
                chunk_path = os.path.join(temp_dir, chunk_filename)
                chunk = AudioSegment.from_file(chunk_path)
                if chunk.channels == 1: chunk = chunk.set_channels(2)
                if chunk.sample_width != 2: chunk = chunk.set_sample_width(2)
                if settings.get("analog_character", 0) > 0:
                    chunk = apply_analog_character(chunk, settings.get("analog_character"))
                chunk_samples = audio_segment_to_float_array(chunk)
                processed_samples = apply_eq_to_samples(chunk_samples, chunk.frame_rate, settings)
                if settings.get("width", 1.0) != 1.0:
                    processed_samples = apply_stereo_width(processed_samples, settings.get("width"))
                processed_chunk = float_array_to_audio_segment(processed_samples, chunk)
                if settings.get("multiband"):
                    processed_chunk = apply_multiband_compressor(processed_chunk, settings)
                processed_chunk_filename = os.path.join(temp_dir, f"processed_chunk_{i:04d}.wav")
                processed_chunk.export(processed_chunk_filename, format="wav")
                processed_chunk_files.append(processed_chunk_filename)
            except Exception:
                logging.exception(f"CRITICAL: Failed during processing of chunk {i+1}.")
                raise
        
        logging.info("Concatenating all processed chunks...")
        concatenated_file_path = os.path.join(temp_dir, "concatenated.wav")
        file_list_path = os.path.join(temp_dir, "filelist.txt")
        with open(file_list_path, 'w') as f:
            for filename in processed_chunk_files:
                f.write(f"file '{os.path.basename(filename)}'\n")
        subprocess.run(
            ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'filelist.txt', '-c', 'copy', 'concatenated.wav'],
            check=True, cwd=temp_dir
        )
        logging.info("Concatenation complete.")
        
        final_file_to_export = concatenated_file_path
        if settings.get("lufs") is not None:
            normalized_file_path = os.path.join(temp_dir, "normalized.wav")
            final_file_to_export = normalize_loudness_on_disk_with_ffmpeg(
                concatenated_file_path, normalized_file_path, settings.get("lufs")
            )
        
        logging.info("Applying final soft limit...")
        subprocess.run(
            ['ffmpeg', '-i', final_file_to_export, '-filter:a', 'alimiter=level_in=1:level_out=1:limit=0.98:attack=5:release=50', '-y', output_file],
            check=True
        )
        logging.info(f"Finished FFmpeg pipeline, exported to {output_file}")


def generate_cover_art(prompt, audio_filename_base, bucket, gcp_project_id):
    logging.info("--- Starting generate_cover_art ---")
    try:
        gcp_project = gcp_project_id
        gcp_location = 'us-east1'
        logging.info(f"Initializing Vertex AI for project '{gcp_project}' in '{gcp_location}'")
        if not gcp_project:
             raise ValueError("GCP Project ID is missing, cannot initialize Vertex AI.")
        vertexai.init(project=gcp_project, location=gcp_location)
        logging.info("Vertex AI initialized successfully.")
    except Exception:
        logging.exception("CRITICAL ERROR during Vertex AI initialization.")
        raise
    try:
        logging.info("Loading ImageGenerationModel...")
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        logging.info("ImageGenerationModel loaded successfully.")
    except Exception:
        logging.exception("CRITICAL ERROR loading the Imagen model.")
        raise
    try:
        logging.info("Calling model.generate_images()...")
        images = model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio="1:1")
        logging.info("model.generate_images() call completed.")
    except Exception:
        logging.exception("CRITICAL ERROR during the generate_images API call.")
        raise
    try:
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_image_file:
            images[0].save(location=temp_image_file.name, include_generation_parameters=True)
            image_filename = f"art_{os.path.splitext(audio_filename_base)[0]}.png"
            image_blob_name = f"processed/{image_filename}"
            blob = bucket.blob(image_blob_name)
            blob.upload_from_filename(temp_image_file.name, content_type='image/png')
            logging.info("Image uploaded to GCS successfully.")
            return image_blob_name
    except Exception:
        logging.exception("CRITICAL ERROR saving or uploading the image.")
        raise


def process_audio_from_gcs(gcs_uri, settings, key_file_path):
    log_memory_usage()
    storage_client = storage.Client.from_service_account_json(key_file_path)
    if not gcs_uri.startswith('gs://'): raise ValueError("Invalid GCS URI")
    bucket_name, blob_name = gcs_uri[5:].split('/', 1)
    bucket = storage_client.bucket(bucket_name)

    input_blob = bucket.blob(blob_name)

    with tempfile.TemporaryDirectory() as base_temp_dir:
        temp_in_path = os.path.join(base_temp_dir, 'input.wav')
        temp_out_path = os.path.join(base_temp_dir, 'output.wav')
        
        logging.info(f"Downloading {gcs_uri} to {temp_in_path}")
        input_blob.download_to_filename(temp_in_path)
        log_memory_usage()
        
        original_filename = settings.get('original_filename', 'unknown.wav')
        output_filename_base = f"mastered_{original_filename}"
        
        new_settings = settings.copy()
        new_settings["input_file"] = temp_in_path
        new_settings["output_file"] = temp_out_path
        
        process_audio_with_ffmpeg_pipeline(new_settings)

        output_blob_name = f"processed/{output_filename_base}"
        output_blob = bucket.blob(output_blob_name)
        logging.info(f"Uploading processed file from {temp_out_path} to gs://{bucket_name}/{output_blob_name}")
        output_blob.upload_from_filename(temp_out_path, content_type='audio/wav')
        
        art_prompt = settings.get("art_prompt")
        gcp_project_id = settings.get("gcp_project_id")
        if art_prompt and art_prompt.strip():
            try:
                generate_cover_art(art_prompt, output_filename_base, bucket, gcp_project_id)
            except Exception:
                logging.error("Cover art generation failed, but mastering succeeded.")
        
        # <<< THIS IS THE FIX >>>
        # Give GCS a moment to ensure the image object is globally consistent
        # before we tell the frontend that the job is done.
        logging.info("Waiting 2 seconds for GCS consistency...")
        time.sleep(2)

        complete_flag_blob = bucket.blob(f"{output_blob_name}.complete")
        complete_flag_blob.upload_from_string("done")

    logging.info("--- TOP LEVEL: process_audio_from_gcs COMPLETE ---")
    log_memory_usage()

# --- All other helper functions remain unchanged ---
# ... (the rest of the file is identical)

def audio_segment_to_float_array(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2: samples = samples.reshape((-1, 2))
    return samples.astype(np.float32) / (2**(audio_segment.sample_width * 8 - 1))

def float_array_to_audio_segment(float_array, audio_segment_template):
    clipped_array = np.clip(float_array, -1.0, 1.0)
    int_array = (clipped_array * 32767).astype(np.int16)
    return audio_segment_template._spawn(int_array.tobytes())

def apply_analog_character(chunk, character_percent):
    if character_percent == 0: return chunk
    character_factor = character_percent / 100.0
    samples = audio_segment_to_float_array(chunk)
    drive = 1.0 + (character_factor * 0.5)
    saturated_samples = np.tanh(samples * drive)
    low_bump_db = character_factor * 1.0 
    saturated_samples = apply_shelf_filter(saturated_samples, chunk.frame_rate, 120, low_bump_db, 'low')
    high_sparkle_db = character_factor * 1.5
    final_samples = apply_shelf_filter(saturated_samples, chunk.frame_rate, 12000, high_sparkle_db, 'high')
    return float_array_to_audio_segment(final_samples, chunk)

def apply_stereo_width(samples, width_factor):
    if samples.ndim != 2 or samples.shape[1] != 2: return samples
    left, right = samples[:, 0], samples[:, 1]
    mid = (left + right) / 2
    side = (left - right) / 2
    side *= width_factor
    new_left = np.clip(mid + side, -1.0, 1.0)
    new_right = np.clip(mid - side, -1.0, 1.0)
    return np.stack([new_left, new_right], axis=1)

def apply_eq_to_samples(samples, sample_rate, settings):
    if samples.ndim == 2:
        for i in range(samples.shape[1]):
            samples[:, i] = _apply_eq_to_channel(samples[:, i], sample_rate, settings)
    else:
        samples = _apply_eq_to_channel(samples, sample_rate, settings)
    return samples

def _apply_eq_to_channel(channel_samples, sample_rate, settings):
    channel_samples = apply_shelf_filter(channel_samples, sample_rate, 250, settings.get("bass_boost", 0.0), 'low')
    channel_samples = apply_peak_filter(channel_samples, sample_rate, 1000, -settings.get("mid_cut", 0.0))
    channel_samples = apply_peak_filter(channel_samples, sample_rate, 4000, settings.get("presence_boost", 0.0))
    channel_samples = apply_shelf_filter(channel_samples, sample_rate, 8000, settings.get("treble_boost", 0.0), 'high')
    return channel_samples

def apply_shelf_filter(samples, sample_rate, cutoff_hz, gain_db, filter_type, order=2):
    if gain_db ==.0: return samples
    gain = 10.0 ** (gain_db / 20.0)
    nyquist = 0.5 * sample_rate
    b, a = butter(order, cutoff_hz / nyquist, btype=filter_type)
    y = lfilter(b, a, samples)
    if gain_db > 0: return samples + (y - samples) * (gain - 1)
    else: return samples * gain + (y - samples * gain)
        
def apply_peak_filter(samples, sample_rate, center_hz, gain_db, q=1.41):
    if gain_db == 0: return samples
    nyquist = 0.5 * sample_rate
    center_norm = center_hz / nyquist
    bandwidth = center_norm / q
    low = center_norm - (bandwidth / 2)
    high = center_norm + (bandwidth / 2)
    if low <= 0: low = 1e-9
    if high >= 1.0: high = 0.999999
    sos = butter(4, [low, high], btype='bandpass', output='sos')
    filtered_band = sosfilt(sos, samples)
    gain_factor = 10 ** (gain_db / 20.0)
    return samples + (filtered_band * (gain_factor - 1))

def apply_multiband_compressor(chunk, settings, low_crossover=250, high_crossover=4000):
    samples = audio_segment_to_float_array(chunk)
    low_sos = butter(4, low_crossover, btype='lowpass', fs=chunk.frame_rate, output='sos')
    high_sos = butter(4, high_crossover, btype='highpass', fs=chunk.frame_rate, output='sos')
    low_band_samples = sosfilt(low_sos, samples, axis=0)
    high_band_samples = sosfilt(high_sos, samples, axis=0)
    mid_band_samples = samples - low_band_samples - high_band_samples
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


    

