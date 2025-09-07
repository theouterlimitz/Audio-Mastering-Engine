# audio_mastering_engine.py (v3.11 - Final Local Version with Direct API)
# This version fixes the stubborn 404 error by bypassing the Vertex AI SDK
# for prompt generation and using a direct REST API call instead.

import os
import tempfile
import numpy as np
import subprocess
import json
import logging
import psutil
import time
import gc
import traceback
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
from scipy.signal import butter, sosfilt, lfilter

try:
    import google.auth
    import google.auth.transport.requests # <-- Needed for direct API calls
    import requests # <-- The standard library for making HTTP requests
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    # We no longer need the GenerativeModel import for this function
except ImportError:
    print("WARNING: Google Cloud libraries not found. AI Art generation will be disabled.")
    google, vertexai, requests = None, None, None

import ai_tagger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

def process_audio(settings, status_callback, progress_callback, art_callback, tag_callback):
    """
    Main entry point for the GUI. Orchestrates the full AI pipeline.
    """
    try:
        process_audio_with_ffmpeg_pipeline(settings, status_callback, progress_callback)
        status_callback("Mastering complete. Preparing for AI analysis...")
        
        auto_generate = settings.get("auto_generate_prompt", False)
        manual_prompt = settings.get("art_prompt", "").strip()
        final_art_prompt = None

        if auto_generate:
            status_callback("Analyzing audio for mood...")
            input_file = settings.get("input_file")
            mood = ai_tagger.predict_mood(input_file)
            tag_callback(mood)

            if "Error" in mood:
                status_callback(f"Failed: Could not analyze mood. {mood}")
            else:
                status_callback(f"Mood detected: {mood}. Brainstorming creative prompt...")
                final_art_prompt = generate_creative_prompt(mood)
                tag_callback(f"{mood} -> \"{final_art_prompt}\"")
        
        elif manual_prompt:
            final_art_prompt = manual_prompt
            tag_callback("Using manual prompt.")

        if final_art_prompt and vertexai:
            status_callback("Starting AI art generation...")
            try:
                output_file = settings.get("output_file")
                art_file_path = generate_cover_art_locally(final_art_prompt, output_file)
                status_callback("Success: AI art generation complete!")
                art_callback(art_file_path)
            except Exception as art_error:
                logging.error(f"Art generation failed: {art_error}")
                status_callback(f"Failed: Mastering complete, but AI art failed.")
                art_callback(None)
        else:
            status_callback("Success: Processing complete! (No art generated)")
            art_callback(None)
            
    except Exception as e:
        error_details = traceback.format_exc()
        logging.error(f"FATAL ERROR in process_audio: {error_details}")
        status_callback(f"Error: {e}")
        progress_callback(0, 1)
        art_callback(None)
        tag_callback("Processing failed.")


def generate_creative_prompt(mood):
    """
    Acts as a "Gemini Art Director" using a direct REST API call.
    """
    if not google or not requests:
        raise RuntimeError("Required Google/Requests libraries are not available.")

    logging.info(f"Brainstorming creative prompt for mood: {mood} via REST API")
    try:
        # 1. Get Application Default Credentials
        credentials, project_id = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        access_token = credentials.token
        
        # 2. Define the API endpoint and the model
        gcp_location = 'us-central1'
        api_endpoint = f"https://{gcp_location}-aiplatform.googleapis.com"
        model_id = "gemini-1.5-pro-latest" # Using the most stable model name
        url = f"{api_endpoint}/v1/projects/{project_id}/locations/{gcp_location}/publishers/google/models/{model_id}:generateContent"

        # 3. Construct the meta-prompt
        meta_prompt = f"""
        You are an expert creative art director for album covers. Your task is to brainstorm a short, evocative, and highly artistic image prompt for an AI image generator. The prompt should capture the essence of a song with the following mood: '{mood}'.
        Rules:
        - Do not mention the mood word (e.g., '{mood}') in the final prompt.
        - Focus on strong visual metaphors, color palettes, lighting, and composition.
        - The final prompt should be a single, concise phrase, no more than 25 words.
        - Example for 'Sad/Depressed': 'A single, withered glowing flower in a vast, empty, rain-slicked city at twilight, cinematic lighting.'
        - Example for 'Happy/Excited': 'Geometric shapes of pure light and vibrant color exploding in a joyful supernova, digital art.'
        Generate the prompt now.
        """

        # 4. Create the JSON payload for the API request
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": meta_prompt}
                    ]
                }
            ]
        }
        
        # 5. Make the authenticated POST request
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status() # This will raise an error for non-200 responses
        
        # 6. Parse the response to get the generated text
        response_json = response.json()
        creative_prompt = response_json['candidates'][0]['content']['parts'][0]['text'].strip().replace('"', '')
        
        logging.info(f"Generated creative prompt: '{creative_prompt}'")
        return creative_prompt

    except Exception as e:
        logging.exception("CRITICAL ERROR during direct API call for creative prompt.")
        logging.warning("Falling back to a simple prompt.")
        return f"An artistic representation of the mood: {mood}, detailed, vibrant colors."


# --- All other functions remain unchanged ---
def generate_cover_art_locally(prompt, audio_output_path):
    if not vertexai: raise RuntimeError("Vertex AI library is not available.")
    logging.info("--- Starting generate_cover_art_locally ---")
    try:
        credentials, gcloud_project_id = google.auth.default()
        gcp_location = 'us-central1'
        if not gcloud_project_id:
            try: gcloud_project_id = subprocess.check_output(['gcloud', 'config', 'get-value', 'project']).strip().decode('utf-8')
            except Exception: raise RuntimeError("Could not determine GCP Project ID. Run 'gcloud config set project YOUR_PROJECT_ID'.")
        logging.info(f"Initializing Vertex AI for project '{gcloud_project_id}' in '{gcp_location}'")
        vertexai.init(project=gcloud_project_id, location=gcp_location, credentials=credentials)
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        images = model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio="1:1")
        path, _ = os.path.split(audio_output_path)
        name, _ = os.path.splitext(os.path.basename(audio_output_path))
        image_output_path = os.path.join(path, f"{name}_art.png")
        images[0].save(location=image_output_path, include_generation_parameters=False)
        logging.info(f"Image saved locally to: {image_output_path}")
        return image_output_path
    except Exception as e:
        logging.exception("CRITICAL ERROR during local art generation.")
        raise e


def process_audio_with_ffmpeg_pipeline(settings, status_callback, progress_callback):
    input_file, output_file = settings.get("input_file"), settings.get("output_file")
    if not input_file or not output_file: raise ValueError("Input or output file not specified.")
    log_memory_usage(f"Pipeline Start")
    with tempfile.TemporaryDirectory() as temp_dir:
        status_callback("Splitting audio into manageable chunks...")
        progress_callback(0, 100)
        split_command = ['ffmpeg', '-i', input_file, '-f', 'segment', '-segment_time', '30', os.path.join(temp_dir, 'input_chunk_%04d.wav')]
        subprocess.run(split_command, check=True, capture_output=True, text=True)
        status_callback("Splitting complete.")
        log_memory_usage("After Splitting")
        input_chunk_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith('input_chunk_')])
        processed_chunk_files, num_chunks = [], len(input_chunk_files)
        total_steps = num_chunks + 3
        
        for i, chunk_path in enumerate(input_chunk_files):
            status_callback(f"Processing chunk {i+1} of {num_chunks}...")
            progress_callback(i + 1, total_steps)
            try:
                chunk = AudioSegment.from_file(chunk_path)
                if chunk.channels == 1: chunk = chunk.set_channels(2)
                if chunk.sample_width != 2: chunk = chunk.set_sample_width(2)
                if settings.get("analog_character", 0) > 0: chunk = apply_analog_character(chunk, settings.get("analog_character"))
                chunk_samples = audio_segment_to_float_array(chunk)
                processed_samples = apply_eq_to_samples(chunk_samples, chunk.frame_rate, settings)
                if settings.get("width", 1.0) != 1.0: processed_samples = apply_stereo_width(processed_samples, settings.get("width"))
                processed_chunk = float_array_to_audio_segment(processed_samples, chunk)
                if settings.get("multiband"): processed_chunk = apply_multiband_compressor(processed_chunk, settings)
                processed_chunk_filename = os.path.join(temp_dir, f"processed_chunk_{i:04d}.wav")
                processed_chunk.export(processed_chunk_filename, format="wav")
                processed_chunk_files.append(processed_chunk_filename)
                del chunk, chunk_samples, processed_samples, processed_chunk
                gc.collect()
                log_memory_usage(f"After Chunk {i+1}")
            except Exception: logging.exception(f"CRITICAL: Failed during processing of chunk {i+1}."); raise
        
        status_callback("Re-assembling processed chunks with concat filter...")
        progress_callback(num_chunks + 1, total_steps)
        concatenated_file_path = os.path.join(temp_dir, "concatenated.wav")
        
        input_args = []
        for chunk_file in processed_chunk_files:
            input_args.extend(['-i', chunk_file])
            
        filter_complex_string = "".join([f"[{i}:a]" for i in range(len(processed_chunk_files))])
        filter_complex_string += f"concat=n={len(processed_chunk_files)}:v=0:a=1[out]"
        
        concat_command = ['ffmpeg', *input_args, '-filter_complex', filter_complex_string, '-map', '[out]', concatenated_file_path]
        subprocess.run(concat_command, check=True, capture_output=True, text=True)
        status_callback("Concatenation complete.")
        log_memory_usage("After Concatenation")
        
        final_file_to_export = concatenated_file_path
        if settings.get("lufs") is not None:
            status_callback("Normalizing final loudness...")
            progress_callback(num_chunks + 2, total_steps)
            normalized_file_path = os.path.join(temp_dir, "normalized.wav")
            final_file_to_export = normalize_loudness_on_disk_with_ffmpeg(concatenated_file_path, normalized_file_path, settings.get("lufs"))
        
        status_callback("Applying final limiting and exporting...")
        progress_callback(num_chunks + 3, total_steps)
        subprocess.run(['ffmpeg', '-i', final_file_to_export, '-filter:a', 'alimiter=level_in=1:level_out=1:limit=0.98:attack=5:release=50', '-y', output_file], check=True, capture_output=True, text=True)
        
        progress_callback(total_steps, total_steps)
        logging.info(f"Finished FFmpeg pipeline, exported to {output_file}")
        log_memory_usage("Pipeline End")


def normalize_loudness_on_disk_with_ffmpeg(input_path, output_path, target_lufs=-14.0):
    logging.info(f"Starting true disk-based loudness normalization for {input_path}...")
    try:
        command_pass1 = ['ffmpeg', '-i', input_path, '-af', f'loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json', '-f', 'null', '-']
        result_pass1 = subprocess.run(command_pass1, capture_output=True, text=True)
        output_lines, json_str, json_started = result_pass1.stderr.splitlines(), "", False
        for line in output_lines:
            if line.strip().startswith('{'): json_started = True
            if json_started: json_str += line
            if line.strip().endswith('}'): break
        if not json_str: raise RuntimeError("Could not parse loudnorm stats from ffmpeg's first pass.")
        measured_stats = json.loads(json_str)
        if measured_stats.get('input_i') == '-inf':
            logging.warning("Measured loudness is -inf (silent audio). Skipping normalization.")
            subprocess.run(['cp', input_path, output_path], check=True)
            return output_path
        logging.info(f"FFmpeg Pass 1 stats: {measured_stats}")
        command_pass2 = ['ffmpeg', '-i', input_path, '-af', f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:measured_I={measured_stats['input_i']}:measured_LRA={measured_stats['input_lra']}:measured_TP={measured_stats['input_tp']}:measured_thresh={measured_stats['input_thresh']}:offset={measured_stats['target_offset']}", '-y', output_path]
        subprocess.run(command_pass2, check=True, capture_output=True, text=True)
        return output_path
    except Exception as e:
        logging.exception("Error during disk-based normalization.")
        if isinstance(e, subprocess.CalledProcessError): logging.error(f"FFMPEG STDERR:\n{e.stderr}")
        subprocess.run(['cp', input_path, output_path], check=True)
        return output_path

EQ_PRESETS = {"Vocal Clarity": {"bass_boost": -1.0, "mid_cut": 2.0, "presence_boost": 2.5, "treble_boost": 1.0}, "Bass Punch": {"bass_boost": 2.5, "mid_cut": 1.0, "presence_boost": -1.0, "treble_boost": 0.5}, "Vintage Warmth": {"bass_boost": 1.5, "mid_cut": 0.0, "presence_boost": -1.5, "treble_boost": -2.0}, "Lo-Fi Haze": {"bass_boost": -2.0, "mid_cut": 3.0, "presence_boost": -2.0, "treble_boost": -4.0}, "EDM Kick & Highs": {"bass_boost": 2.0, "mid_cut": 4.0, "presence_boost": 1.0, "treble_boost": 3.0}}
def log_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"MEMORY USAGE at '{stage}': {mem_info.rss / 1024 ** 2:.2f} MB")
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
    samples = audio_segment_to_float_array(chunk); drive = 1.0 + (character_factor * 0.5)
    saturated_samples = np.tanh(samples * drive)
    saturated_samples = apply_shelf_filter(saturated_samples, chunk.frame_rate, 120, character_factor * 1.0, 'low')
    final_samples = apply_shelf_filter(saturated_samples, chunk.frame_rate, 12000, character_factor * 1.5, 'high')
    return float_array_to_audio_segment(final_samples, chunk)
def apply_stereo_width(samples, width_factor):
    if samples.ndim != 2 or samples.shape[1] != 2: return samples
    left, right = samples[:, 0], samples[:, 1]; mid, side = (left + right) / 2, (left - right) / 2
    side *= width_factor; new_left, new_right = np.clip(mid + side, -1.0, 1.0), np.clip(mid - side, -1.0, 1.0)
    return np.stack([new_left, new_right], axis=1)
def apply_eq_to_samples(samples, sample_rate, settings):
    if samples.ndim == 2:
        for i in range(samples.shape[1]): samples[:, i] = _apply_eq_to_channel(samples[:, i], sample_rate, settings)
    else: samples = _apply_eq_to_channel(samples, sample_rate, settings)
    return samples
def _apply_eq_to_channel(channel_samples, sample_rate, settings):
    channel_samples = apply_shelf_filter(channel_samples, sample_rate, 250, settings.get("bass_boost", 0.0), 'low')
    channel_samples = apply_peak_filter(channel_samples, sample_rate, 1000, -settings.get("mid_cut", 0.0))
    channel_samples = apply_peak_filter(channel_samples, sample_rate, 4000, settings.get("presence_boost", 0.0))
    channel_samples = apply_shelf_filter(channel_samples, sample_rate, 8000, settings.get("treble_boost", 0.0), 'high')
    return channel_samples
def apply_shelf_filter(samples, sample_rate, cutoff_hz, gain_db, filter_type, order=2):
    if gain_db == 0.0: return samples
    gain = 10.0 ** (gain_db / 20.0); b, a = butter(order, cutoff_hz / (0.5 * sample_rate), btype=filter_type)
    y = lfilter(b, a, samples)
    if gain_db > 0: return samples + (y - samples) * (gain - 1)
    else: return samples * gain + (y - samples * gain)
def apply_peak_filter(samples, sample_rate, center_hz, gain_db, q=1.41):
    if gain_db == 0: return samples
    nyquist = 0.5 * sample_rate
    center_norm = center_hz / nyquist
    bandwidth = center_norm / q; low, high = center_norm - (bandwidth / 2), center_norm + (bandwidth / 2)
    if low <= 0: low = 1e-9
    if high >= 1.0: high = 0.999999
    sos = butter(4, [low, high], btype='bandpass', output='sos'); filtered_band = sosfilt(sos, samples)
    return samples + (filtered_band * (10 ** (gain_db / 20.0) - 1))
def apply_multiband_compressor(chunk, settings, low_crossover=250, high_crossover=4000):
    samples = audio_segment_to_float_array(chunk)
    low_sos = butter(4, low_crossover, btype='lowpass', fs=chunk.frame_rate, output='sos')
    high_sos = butter(4, high_crossover, btype='highpass', fs=chunk.frame_rate, output='sos')
    low_band_samples, high_band_samples = sosfilt(low_sos, samples, axis=0), sosfilt(high_sos, samples, axis=0)
    mid_band_samples = samples - low_band_samples - high_band_samples
    low_band_chunk, mid_band_chunk, high_band_chunk = float_array_to_audio_segment(low_band_samples, chunk), float_array_to_audio_segment(mid_band_samples, chunk), float_array_to_audio_segment(high_band_samples, chunk)
    low_compressed = compress_dynamic_range(low_band_chunk, threshold=settings.get("low_thresh"), ratio=settings.get("low_ratio"))
    mid_compressed = compress_dynamic_range(mid_band_chunk, threshold=settings.get("mid_thresh"), ratio=settings.get("mid_ratio"))
    high_compressed = compress_dynamic_range(high_band_chunk, threshold=settings.get("high_thresh"), ratio=settings.get("high_ratio"))
    return low_compressed.overlay(mid_compressed).overlay(high_compressed)