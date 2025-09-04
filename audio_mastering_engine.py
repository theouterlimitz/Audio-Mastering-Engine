# audio_mastering_engine.py (Disk-Based Chunking Architecture)
# This is the definitive version designed for massive scalability. It processes
# audio in chunks and writes them to disk, avoiding memory overload. At the end,
# it uses ffmpeg to concatenate the chunks for final processing.

import os
import tempfile
import numpy as np
import subprocess
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
from scipy.signal import butter, sosfilt, lfilter
import pyloudnorm as pyln
import traceback

# GCP Libraries
from google.cloud import storage
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# --- New Disk-Based Chunking Processing Logic ---

def process_audio_in_chunks(settings):
    """
    The main audio processing function, redesigned for maximum memory efficiency.
    It saves processed chunks to disk and uses ffmpeg for concatenation.
    """
    input_file = settings.get("input_file")
    output_file = settings.get("output_file")

    if not input_file or not output_file:
        raise ValueError("Input or output file not specified.")

    print(f"BREADCRUMB: Starting DISK-BASED CHUNKED process_audio for {input_file}")

    # Use a temporary directory to store the processed chunks
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_info = AudioSegment.from_file(input_file)
        chunk_size_ms = 30 * 1000
        num_chunks = (len(audio_info) // chunk_size_ms) + 1
        chunk_files = []

        print(f"BREADCRUMB: Beginning to process {len(audio_info)}ms of audio in {num_chunks} chunks.")

        for i in range(num_chunks):
            start_ms = i * chunk_size_ms
            end_ms = start_ms + chunk_size_ms
            chunk = audio_info[start_ms:end_ms]

            # --- The same high-quality processing logic as before ---
            if chunk.channels == 1:
                chunk = chunk.set_channels(2)
            if chunk.sample_width != 2:
                chunk = chunk.set_sample_width(2)

            if settings.get("analog_character", 0) > 0:
                chunk = apply_analog_character(chunk, settings.get("analog_character"))

            chunk_samples = audio_segment_to_float_array(chunk)
            processed_samples = apply_eq_to_samples(chunk_samples, chunk.frame_rate, settings)
            
            if settings.get("width", 1.0) != 1.0:
                processed_samples = apply_stereo_width(processed_samples, settings.get("width"))
                
            processed_chunk = float_array_to_audio_segment(processed_samples, chunk)
            
            if settings.get("multiband"):
                processed_chunk = apply_multiband_compressor(processed_chunk, settings)
            # --- End of processing logic ---

            # Save the processed chunk to a temporary file on disk
            chunk_filename = os.path.join(temp_dir, f"chunk_{i:04d}.wav")
            processed_chunk.export(chunk_filename, format="wav")
            chunk_files.append(chunk_filename)
            print(f"BREADCRUMB: Processed and saved chunk {i+1}/{num_chunks}")

        # --- Concatenate chunks using ffmpeg for memory efficiency ---
        print("BREADCRUMB: Concatenating all processed chunks using ffmpeg...")
        concatenated_file_path = os.path.join(temp_dir, "concatenated.wav")
        
        # Create a file list for ffmpeg
        file_list_path = os.path.join(temp_dir, "filelist.txt")
        with open(file_list_path, 'w') as f:
            for filename in chunk_files:
                f.write(f"file '{os.path.basename(filename)}'\n")
        
        # Run ffmpeg command
        # The command must be run from within the temp_dir for the relative paths to work
        subprocess.run(
            ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'filelist.txt', '-c', 'copy', 'concatenated.wav'],
            check=True,
            cwd=temp_dir
        )
        print("BREADCRUMB: Concatenation complete.")

        # --- Final Normalization and Limiting ---
        print("BREADCRUMB: Loading concatenated file for final normalization...")
        full_processed_audio = AudioSegment.from_file(concatenated_file_path)
        
        final_samples = audio_segment_to_float_array(full_processed_audio)

        if settings.get("lufs") is not None:
            final_samples = normalize_to_lufs(final_samples, full_processed_audio.frame_rate, settings.get("lufs"))

        final_samples = soft_limiter(final_samples)
        final_audio = float_array_to_audio_segment(final_samples, full_processed_audio)
        
        # Export the final combined file
        output_format = os.path.splitext(output_file)[1][1:].lower() or "wav"
        final_audio.export(output_file, format=output_format)
            
        print(f"BREADCRUMB: Finished DISK-BASED CHUNKED process_audio, exported to {output_file}")


# --- All other functions (generate_cover_art, process_audio_from_gcs, etc.) remain largely the same ---

def generate_cover_art(prompt, audio_filename_base, bucket):
    """
    Generates cover art using Vertex AI's Imagen model and uploads it to GCS.
    """
    print("BREADCRUMB: --- Starting generate_cover_art ---")
    
    try:
        gcp_project = os.environ.get('GCP_PROJECT_ID')
        gcp_location = 'us-east1'
        print(f"BREADCRUMB: Initializing Vertex AI for project '{gcp_project}' in '{gcp_location}'")
        vertexai.init(project=gcp_project, location=gcp_location)
        print("BREADCRUMB: Vertex AI initialized successfully.")
    except Exception as e:
        print(f"BREADCRUMB: CRITICAL ERROR during Vertex AI initialization: {e}")
        traceback.print_exc()
        raise

    try:
        print("BREADCRUMB: Loading ImageGenerationModel...")
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        print("BREADCRUMB: ImageGenerationModel loaded successfully.")
    except Exception as e:
        print(f"BREADCRUMB: CRITICAL ERROR loading the Imagen model: {e}")
        traceback.print_exc()
        raise
        
    try:
        print("BREADCRUMB: Calling model.generate_images()...")
        images = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1"
        )
        print("BREADCRUMB: model.generate_images() call completed.")
    except Exception as e:
        print(f"BREADCRUMB: CRITICAL ERROR during the generate_images API call: {e}")
        traceback.print_exc()
        raise

    try:
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_image_file:
            print(f"BREADCRUMB: Saving generated image to temporary file: {temp_image_file.name}")
            images[0].save(location=temp_image_file.name, include_generation_parameters=True)
            print("BREADCRUMB: Image saved to temp file successfully.")

            image_filename = f"art_{os.path.splitext(audio_filename_base)[0]}.png"
            image_blob_name = f"processed/{image_filename}"
            
            print(f"BREADCRUMB: Uploading image to GCS at: {image_blob_name}")
            blob = bucket.blob(image_blob_name)
            blob.upload_from_filename(temp_image_file.name, content_type='image/png')
            print("BREADCRUMB: Image uploaded to GCS successfully.")
            
            return image_blob_name
    except Exception as e:
        print(f"BREADCRUMB: CRITICAL ERROR saving or uploading the image: {e}")
        traceback.print_exc()
        raise


def process_audio_from_gcs(gcs_uri, settings, key_file_path):
    """
    Downloads, processes, and uploads audio using GCS, with a dedicated service account.
    Also handles AI art generation.
    """
    storage_client = storage.Client.from_service_account_json(key_file_path)
    
    if not gcs_uri.startswith('gs://'):
        raise ValueError("Invalid GCS URI")
    bucket_name, blob_name = gcs_uri[5:].split('/', 1)
    
    bucket = storage_client.bucket(bucket_name)
    input_blob = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(blob_name)[1]) as temp_in:
        print(f"Downloading {gcs_uri} to {temp_in.name}")
        input_blob.download_to_filename(temp_in.name)

        original_filename = settings.get('original_filename', 'unknown.wav')
        output_filename_base = f"mastered_{original_filename}"
        
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_out:
            new_settings = settings.copy()
            new_settings["input_file"] = temp_in.name
            new_settings["output_file"] = temp_out.name
            
            # <<< THIS IS THE KEY CHANGE >>>
            # We now call our new memory-efficient disk-based chunking function
            process_audio_in_chunks(new_settings)

            output_blob_name = f"processed/{output_filename_base}"
            output_blob = bucket.blob(output_blob_name)
            
            print(f"Uploading processed file to gs://{bucket_name}/{output_blob_name}")
            output_blob.upload_from_filename(temp_out.name, content_type='audio/wav')

            # --- AI Art Generation Step ---
            art_prompt = settings.get("art_prompt")
            if art_prompt and art_prompt.strip():
                try:
                    generate_cover_art(art_prompt, output_filename_base, bucket)
                except Exception:
                    # We print the traceback in the function now, so this can be simpler
                    print("="*50)
                    print("WARNING: Cover art generation failed, but mastering succeeded.")
                    print("="*50)
            
            # Create a ".complete" flag file for the status checker
            complete_flag_blob = bucket.blob(f"{output_blob_name}.complete")
            complete_flag_blob.upload_from_string("done")

# --- All the other helper functions (audio_segment_to_float_array, etc.) remain unchanged ---

def audio_segment_to_float_array(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
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
    if gain_db == 0: return samples
    gain = 10.0 ** (gain_db / 20.0)
    nyquist = 0.5 * sample_rate
    b, a = butter(order, cutoff_hz / nyquist, btype=filter_type)
    y = lfilter(b, a, samples)
    if gain_db > 0: 
        return samples + (y - samples) * (gain - 1)
    else:
        return samples * gain + (y - samples * gain)
        
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

def normalize_to_lufs(samples, sample_rate, target_lufs=-14.0):
    meter = pyln.Meter(sample_rate)
    mono_samples = samples.mean(axis=1) if samples.ndim == 2 else samples
    
    if np.max(np.abs(mono_samples)) == 0: return samples

    loudness = meter.integrated_loudness(mono_samples)
    gain_db = target_lufs - loudness
    gain_linear = 10.0 ** (gain_db / 20.0)
    return samples * gain_linear

def soft_limiter(samples, threshold=0.98):
    peak = np.max(np.abs(samples))
    if peak > threshold:
        gain_reduction = threshold / peak
        samples *= gain_reduction
    return samples

