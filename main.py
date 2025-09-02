# main.py (App Engine Version - Corrected)
#
# This version uses App Engine Tasks correctly and calls the new GCS-aware
# function in our mastering engine.

import os
import json
import datetime
from flask import Flask, render_template, request, jsonify

# Import the necessary Google Cloud libraries
from google.cloud import storage
from google.cloud import tasks_v2

# Import our known-good mastering engine!
# We only need the GCS function for this cloud application.
from audio_mastering_engine import process_audio_from_gcs

app = Flask(__name__)

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
BUCKET_NAME = f"{GCP_PROJECT_ID}.appspot.com"
TASK_QUEUE = 'mastering-queue'
TASK_LOCATION = 'us-central1' 

# Initialize the clients for Google Cloud services
storage_client = storage.Client()
tasks_client = tasks_v2.CloudTasksClient()

# --- Frontend Route ---
# This serves your index.html from a 'templates' folder.
# Ensure your index.html is inside a folder named 'templates'.
@app.route('/')
def index():
    """Serves the main index.html page."""
    return render_template('index.html')

# --- API Endpoints ---
@app.route('/generate-upload-url', methods=['POST'])
def generate_upload_url():
    """Generates a secure, short-lived URL for the client to upload a file directly to GCS."""
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({"error": "Filename not provided"}), 400

    # Sanitize filename to prevent security issues
    safe_filename = "".join(c for c in data['filename'] if c.isalnum() or c in ('.', '_', '-')).strip()
    blob_name = f"raw_uploads/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{safe_filename}"

    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=15),
        method="PUT",
        content_type=data.get('contentType', 'application/octet-stream'),
    )
    
    gcs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
    return jsonify({"url": url, "gcs_uri": gcs_uri}), 200

@app.route('/start-processing', methods=['POST'])
def start_processing():
    """Receives confirmation of a successful upload and creates a background task."""
    data = request.get_json()
    if not data or 'gcs_uri' not in data or 'settings' not in data:
        return jsonify({"error": "Missing GCS URI or settings"}), 400

    parent = tasks_client.queue_path(GCP_PROJECT_ID, TASK_LOCATION, TASK_QUEUE)
    
    # Use App Engine HTTP Request for reliable task routing
    task = {
        "app_engine_http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "relative_uri": "/process-task",
            "headers": {"Content-type": "application/json"},
            "body": json.dumps(data).encode(),
        }
    }
    tasks_client.create_task(parent=parent, task=task)
    
    original_filename = data['settings'].get('original_filename', 'unknown.wav')
    processed_filename = f"processed/mastered_{original_filename}"
    return jsonify({"message": "Processing job started.", "processed_filename": processed_filename}), 200

@app.route('/status', methods=['GET'])
def get_status():
    """Checks if a processed file exists and provides a download link."""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({"error": "Filename parameter is required"}), 400
        
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # Check for the ".complete" flag file first.
    complete_flag_blob = bucket.blob(f"{filename}.complete")
    if not complete_flag_blob.exists():
        return jsonify({"status": "processing"}), 200

    audio_blob = bucket.blob(filename)
    if not audio_blob.exists():
         return jsonify({"status": "error", "message": "Processing complete but output file is missing."}), 404
    
    download_url = audio_blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=60)
    )
    return jsonify({"status": "done", "download_url": download_url}), 200

# --- Background Worker Route ---
@app.route('/process-task', methods=['POST'])
def process_task():
    """This endpoint is called by Cloud Tasks to run the mastering engine."""
    job_data = json.loads(request.data.decode('utf-8'))
    gcs_uri = job_data.get('gcs_uri')
    settings = job_data.get('settings')

    if not gcs_uri or not settings:
        print(f"ERROR: Invalid task data received: {job_data}")
        return "Bad Request: Invalid task data", 400

    try:
        print(f"Starting background processing for {gcs_uri}")
        process_audio_from_gcs(gcs_uri, settings)
        print(f"Successfully completed processing for {gcs_uri}")
        return "OK", 200 
    except Exception as e:
        print(f"CRITICAL ERROR processing {gcs_uri}: {e}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
