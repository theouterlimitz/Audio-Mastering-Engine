# main.py (App Engine Version)
#
# This is the unified, monolithic application for our Google App Engine deployment.
# It serves the frontend, handles API requests, and manages background tasks.

import os
import json
import datetime
from flask import Flask, render_template, request, jsonify

# Import the necessary Google Cloud libraries
from google.cloud import storage
from google.cloud import tasks_v2

# Import our known-good mastering engine!
# It's in the same directory, so the import is simple.
from audio_mastering_engine import process_audio_from_gcs

app = Flask(__name__)

# --- Configuration ---
# App Engine automatically provides the project ID and a default bucket.
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'tactile-temple-395019') # Replace with your project ID
BUCKET_NAME = f"{GCP_PROJECT_ID}.appspot.com"
TASK_QUEUE = 'mastering-queue' # The name for our background job queue
TASK_LOCATION = 'us-central1' # The region for our queue

# Initialize the clients for Google Cloud services
storage_client = storage.Client()
tasks_client = tasks_v2.CloudTasksClient()

# --- Frontend Route ---
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

    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"raw_uploads/{data['filename']}")

    # Generate a V4 signed URL. App Engine uses its attached service account automatically.
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=15),
        method="PUT",
        content_type=data.get('contentType', 'application/octet-stream'),
    )
    
    gcs_uri = f"gs://{BUCKET_NAME}/raw_uploads/{data['filename']}"
    return jsonify({"url": url, "gcs_uri": gcs_uri}), 200

@app.route('/start-processing', methods=['POST'])
def start_processing():
    """Receives confirmation of a successful upload and creates a background task."""
    data = request.get_json()
    if not data or 'gcs_uri' not in data or 'settings' not in data:
        return jsonify({"error": "Missing GCS URI or settings"}), 400

    # Create a task and add it to the queue.
    parent = tasks_client.queue_path(GCP_PROJECT_ID, TASK_LOCATION, TASK_QUEUE)
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            # The URL points to our own app's /process-task endpoint
            "url": f"/process-task", 
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
    """
    This is the endpoint that our background task queue calls.
    It receives the job data and runs the actual mastering engine.
    """
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
        return "OK", 200 # A success code tells the task queue the job is done.
    except Exception as e:
        print(f"CRITICAL ERROR processing {gcs_uri}: {e}")
        # Return an error code so the task queue knows to retry the job.
        return "Internal Server Error", 500

if __name__ == '__main__':
    # This is used for local testing. On App Engine, Gunicorn runs the 'app' object.
    app.run(debug=True, host='0.0.0.0', port=8080)
