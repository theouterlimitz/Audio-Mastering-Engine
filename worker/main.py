# worker/main.py
# This is the final version of the worker's main application.
# It now uses a standard, direct import for the shared engine.

import os
import sys
import json
import logging
import tempfile
from flask import Flask, request, jsonify

# No more sys.path manipulation needed! Python can now find the
# 'shared' directory directly within the worker's own context.
from shared import audio_mastering_engine

# Import Google Cloud libraries
from google.cloud import firestore, storage

# --- Configuration & Initialization ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

try:
    db = firestore.Client()
    storage_client = storage.Client()
    GCP_PROJECT_ID = os.environ.get('GCP_PROJECT')
    BUCKET_NAME = f"{GCP_PROJECT_ID}.appspot.com"
except Exception as e:
    logging.critical(f"FATAL: Could not initialize GCP clients: {e}")
    db = None
    storage_client = None

@app.route('/process-task', methods=['POST'])
def process_task():
    """
    This is the main entry point for the worker, triggered by a Cloud Task.
    """
    if not db or not storage_client:
        logging.error("Worker is misconfigured; GCP clients not available.")
        return "Internal Server Error: Misconfigured", 500

    job_data = {}
    try:
        job_data = json.loads(request.data.decode('utf-8'))
        job_id = job_data.get('job_id')
        
        if not job_id:
            logging.error("Received task with no job_id.")
            return "Bad Request: Missing job_id", 400

        job_ref = db.collection('mastering_jobs').document(job_id)
        job_ref.update({'status': 'Processing audio master...', 'worker_received_at': firestore.SERVER_TIMESTAMP})

        process_gcs_file(job_data, job_ref)
        
        logging.info(f"Successfully completed job {job_id}.")
        job_ref.update({'status': 'Complete', 'completed_at': firestore.SERVER_TIMESTAMP})
        
        return "OK", 200

    except Exception as e:
        logging.exception(f"FATAL ERROR processing task.")
        job_id = job_data.get('job_id')
        if job_id:
            job_ref = db.collection('mastering_jobs').document(job_id)
            job_ref.update({'status': 'Error', 'error_message': str(e)})
        return "OK", 200

def process_gcs_file(job_data, job_ref):
    """
    A wrapper that handles downloading from GCS, running the local engine,
    and uploading the results back to GCS.
    """
    gcs_uri = job_data.get('gcs_uri')
    if not gcs_uri: raise ValueError("gcs_uri not found in job data")

    bucket = storage_client.bucket(BUCKET_NAME)
    input_blob = bucket.blob(gcs_uri.replace(f"gs://{BUCKET_NAME}/", ""))

    with tempfile.TemporaryDirectory() as temp_dir:
        base_filename = os.path.basename(job_data['settings']['original_filename'])
        name, _ = os.path.splitext(base_filename)
        
        local_input_path = os.path.join(temp_dir, base_filename)
        local_output_wav = os.path.join(temp_dir, f"{name}_mastered.wav")

        logging.info(f"Downloading {gcs_uri} to {local_input_path}...")
        input_blob.download_to_filename(local_input_path)

        settings = job_data['settings']
        settings['input_file'] = local_input_path
        settings['output_file'] = local_output_wav

        def update_status(message):
            logging.info(f"Job {job_data['job_id']}: {message}")
            job_ref.update({'status': message})
        
        def update_progress(current, total):
            progress = int((current / total) * 100) if total > 0 else 0
            job_ref.update({'progress': progress})
        
        def update_job_with_notes(note):
            job_ref.update({'studio_notes': note})

        audio_mastering_engine.process_audio(
            settings, update_status, update_progress, 
            lambda art_path: None, # Art callback not needed by worker
            update_job_with_notes # Use the new tag callback for studio notes
        )

        output_files_to_upload = {
            'output_wav_uri': local_output_wav,
            'output_mp3_uri': local_output_wav.replace('.wav', '.mp3'),
            'output_art_uri': local_output_wav.replace('.wav', '_art.png')
        }
        
        uploaded_paths = {}
        for key, local_path in output_files_to_upload.items():
            if os.path.exists(local_path):
                gcs_path = f"processed/{job_data['job_id']}/{os.path.basename(local_path)}"
                blob = bucket.blob(gcs_path)
                logging.info(f"Uploading {local_path} to gs://{BUCKET_NAME}/{gcs_path}")
                blob.upload_from_filename(local_path)
                uploaded_paths[key] = f"gs://{BUCKET_NAME}/{gcs_path}"

        job_ref.update(uploaded_paths)

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=PORT, debug=True)


