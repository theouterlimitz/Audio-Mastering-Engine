# main.py (Production Version)
# This is the final, clean version of the server code.
# The '/test-task' debugging endpoint has been removed.

import os
import json
import datetime
import traceback
from flask import Flask, render_template, request, jsonify

# Import the necessary Google Cloud libraries
from google.cloud import storage
from google.cloud import tasks_v2
import google.auth
from google.auth.transport import requests

app = Flask(__name__)

# --- Configuration ---
KEY_FILE_PATH = 'sa-key.json'
TASK_QUEUE = 'mastering-queue'
TASK_LOCATION = 'us-east1' 

# Initialize clients using the dedicated service account key
try:
    credentials, GCP_PROJECT_ID = google.auth.load_credentials_from_file(KEY_FILE_PATH)
    storage_client = storage.Client(credentials=credentials)
    tasks_client = tasks_v2.CloudTasksClient(credentials=credentials)
    BUCKET_NAME = f"{GCP_PROJECT_ID}.appspot.com"
except Exception as e:
    print(f"CRITICAL STARTUP ERROR: Could not initialize clients from '{KEY_FILE_PATH}'. Error: {e}")
    storage_client = None
    tasks_client = None
    GCP_PROJECT_ID = "error"
    BUCKET_NAME = "error"

# --- Frontend Route ---
@app.route('/')
def index():
    """Serves the main frontend page."""
    return render_template('index.html')

# --- API Endpoints ---
@app.route('/generate-upload-url', methods=['POST'])
def generate_upload_url():
    """Generates a secure URL for the client to upload a file directly to GCS."""
    if not storage_client:
        return jsonify({"error": "Backend server is misconfigured: Missing service account key."}), 500
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({"error": "Filename not provided"}), 400

        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"raw_uploads/{data['filename']}")
        
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=15),
            method="PUT",
            content_type=data.get('contentType', 'application/octet-stream'),
            credentials=credentials
        )
        
        gcs_uri = f"gs://{BUCKET_NAME}/raw_uploads/{data['filename']}"
        return jsonify({"url": url, "gcs_uri": gcs_uri}), 200
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"ERROR in generate_upload_url: {error_details}")
        return jsonify({"error": f"Backend Error: {error_details}"}), 500

@app.route('/start-processing', methods=['POST'])
def start_processing():
    """Receives confirmation of an upload and creates a background processing task."""
    if not tasks_client:
        return jsonify({"error": "Backend server is misconfigured: Missing service account key."}), 500
    try:
        data = request.get_json()
        if not data or 'gcs_uri' not in data or 'settings' not in data:
            return jsonify({"error": "Missing GCS URI or settings"}), 400
        
        queue_path = tasks_client.queue_path(GCP_PROJECT_ID, TASK_LOCATION, TASK_QUEUE)

        # Add the key file path to the task payload so the worker can use it
        task_payload = data.copy()
        task_payload['key_file_path'] = KEY_FILE_PATH
        
        task = {
            "app_engine_http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "relative_uri": "/process-task",
                "headers": {"Content-type": "application/json"},
                "body": json.dumps(task_payload).encode(),
            }
        }
        
        created_task = tasks_client.create_task(parent=queue_path, task=task)
        print(f"Successfully created task: {created_task.name}")
        
        original_filename = data['settings'].get('original_filename', 'unknown.wav')
        processed_filename = f"processed/mastered_{original_filename}"
        return jsonify({"message": "Processing job started.", "processed_filename": processed_filename}), 200
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"CRITICAL ERROR in start_processing: {error_details}")
        return jsonify({"error": f"Backend Error: {error_details}"}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Checks if a processed file exists and provides a download link."""
    if not storage_client:
        return jsonify({"error": "Backend server is misconfigured: Missing service account key."}), 500
    try:
        filename = request.args.get('filename')
        if not filename:
            return jsonify({"error": "Filename parameter is required"}), 400
            
        bucket = storage_client.bucket(BUCKET_NAME)
        
        complete_flag_blob = bucket.blob(f"{filename}.complete")
        if not complete_flag_blob.exists():
            return jsonify({"status": "processing"}), 200

        audio_blob = bucket.blob(filename)
        if not audio_blob.exists():
             return jsonify({"status": "error", "message": "Processing complete but output file is missing."}), 404
        
        download_url = audio_blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=60),
            credentials=credentials
        )
        return jsonify({"status": "done", "download_url": download_url}), 200
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"ERROR in get_status: {error_details}")
        return jsonify({"error": f"Backend Error: {error_details}"}), 500

@app.route('/process-task', methods=['POST'])
def process_task():
    """This endpoint is called by Cloud Tasks to run the background job."""
    from audio_mastering_engine import process_audio_from_gcs
    
    job_data = json.loads(request.data.decode('utf-8'))
    
    gcs_uri = job_data.get('gcs_uri')
    settings = job_data.get('settings')
    key_file = job_data.get('key_file_path')

    if not gcs_uri or not settings or not key_file:
        print(f"ERROR: Invalid real task data received: {job_data}")
        return "Bad Request: Invalid real task data", 400

    try:
        print(f"Starting background processing for {gcs_uri}")
        process_audio_from_gcs(gcs_uri, settings, key_file)
        print(f"Successfully completed processing for {gcs_uri}")
        return "OK", 200
    except Exception as e:
        print(f"CRITICAL ERROR processing {gcs_uri}: {traceback.format_exc()}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    # This is used for local testing. On App Engine, Gunicorn runs the 'app' object.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

