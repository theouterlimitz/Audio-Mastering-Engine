# worker/main.py
# This is the "Chef" - the main application for our backend worker service.
# It listens for jobs from Cloud Tasks, processes them using the shared engine,
# and updates the job status in Firestore.

import os
import sys
import json
import logging
import tempfile
from flask import Flask, request, jsonify

# This is a crucial step to allow the worker to import our "Golden Master"
# engine from the shared/ directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Google Cloud libraries
from google.cloud import firestore, storage

# Import our shared, proven engine
from shared import audio_mastering_engine

# --- Configuration & Initialization ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Initialize GCP clients. They will automatically use the service account
# permissions of the App Engine instance they are running on.
try:
    db = firestore.Client()
    storage_client = storage.Client()
    # Get the project ID and default bucket name from the environment
    GCP_PROJECT_ID = os.environ.get('GCP_PROJECT')
    BUCKET_NAME = f"{GCP_PROJECT_ID}.appspot.com"
except Exception as e:
    logging.critical(f"FATAL: Could not initialize GCP clients: {e}")
    db = None
    storage_client = None

# --- Main Worker Endpoint ---
@app.route('/process-task', methods=['POST'])
def process_task():
    """
    This is the main entry point for the worker, triggered by a Cloud Task.
    """
    if not db or not storage_client:
        logging.error("Worker is misconfigured; GCP clients not available.")
        # Return a 500 error to tell Cloud Tasks to retry later.
        return "Internal Server Error: Misconfigured", 500

    try:
        # The job "ticket" is in the request body, sent by Cloud Tasks.
        job_data = json.loads(request.data.decode('utf-8'))
        job_id = job_data.get('job_id')
        
        if not job_id:
            logging.error("Received task with no job_id.")
            return "Bad Request: Missing job_id", 400

        # Update the "status board" (Firestore) to show we've started.
        job_ref = db.collection('mastering_jobs').document(job_id)
        job_ref.update({'status': 'Processing audio master...', 'worker_received_at': firestore.SERVER_TIMESTAMP})

        # --- Execute the main processing pipeline ---
        process_gcs_file(job_data, job_ref)
        
        # If we get here, the process was successful.
        logging.info(f"Successfully completed job {job_id}.")
        job_ref.update({'status': 'Complete', 'completed_at': firestore.SERVER_TIMESTAMP})
        
        # Tell Cloud Tasks that we have successfully handled the job.
        return "OK", 200

    except Exception as e:
        logging.exception(f"FATAL ERROR processing task.")
        # If a job_id was present, update Firestore with the error.
        if 'job_id' in locals() and job_id:
            job_ref = db.collection('mastering_jobs').document(job_id)
            job_ref.update({'status': 'Error', 'error_message': str(e)})
        # Return a 200 OK even on failure. We've logged the error in Firestore,
        # so we don't want Cloud Tasks to retry the failed job.
        return "OK", 200

def process_gcs_file(job_data, job_ref):
    """
    A wrapper that handles downloading from GCS, running the local engine,
    and uploading the results back to GCS.
    """
    gcs_uri = job_data.get('gcs_uri')
    if not gcs_uri:
        raise ValueError("gcs_uri not found in job data")

    bucket = storage_client.bucket(BUCKET_NAME)
    input_blob = bucket.blob(gcs_uri.replace(f"gs://{BUCKET_NAME}/", ""))

    with tempfile.TemporaryDirectory() as temp_dir:
        base_filename = os.path.basename(job_data['settings']['original_filename'])
        name, ext = os.path.splitext(base_filename)
        
        # Define local file paths inside the temporary directory
        local_input_path = os.path.join(temp_dir, base_filename)
        local_output_wav = os.path.join(temp_dir, f"{name}_mastered.wav")

        # Download the raw file from GCS
        logging.info(f"Downloading {gcs_uri} to {local_input_path}...")
        input_blob.download_to_filename(local_input_path)

        # Prepare the settings for our "Golden Master" engine
        settings = job_data['settings']
        settings['input_file'] = local_input_path
        settings['output_file'] = local_output_wav

        # --- Define Cloud-Aware Callbacks ---
        # These functions will update our Firestore "status board" in real-time.
        def update_status(message):
            logging.info(f"Job {job_data['job_id']}: {message}")
            job_ref.update({'status': message})
        
        def update_progress(current, total):
            progress = int((current / total) * 100) if total > 0 else 0
            job_ref.update({'progress': progress})
        
        # Art and Tag callbacks are not needed for the worker's logic,
        # but the engine function expects them.
        def no_op_callback(data): pass

        # --- Call the "Golden Master" ---
        audio_mastering_engine.process_audio(
            settings, update_status, update_progress, 
            no_op_callback, no_op_callback
        )

        # --- Upload all resulting files back to GCS ---
        output_files_to_upload = {
            'output_wav_uri': local_output_wav,
            'output_mp3_uri': local_output_wav.replace('.wav', '.mp3'),
            'output_art_uri': local_output_wav.replace('.wav', '_art.png')
        }
        
        uploaded_paths = {}
        for key, local_path in output_files_to_upload.items():
            if os.path.exists(local_path):
                gcs_path = f"processed/{os.path.basename(local_path)}"
                blob = bucket.blob(gcs_path)
                logging.info(f"Uploading {local_path} to gs://{BUCKET_NAME}/{gcs_path}")
                blob.upload_from_filename(local_path)
                uploaded_paths[key] = f"gs://{BUCKET_NAME}/{gcs_path}"

        # Final update to Firestore with the paths to the final files
        job_ref.update(uploaded_paths)

if __name__ == "__main__":
    # This is used for local testing of the worker.
    # On App Engine, Gunicorn runs the 'app' object defined above.
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=PORT, debug=True)

