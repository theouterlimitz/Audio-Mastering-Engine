# main.py (Final Version with Full Error Reporting)
#
# This version adds detailed error reporting to the /start-processing endpoint.
# If anything goes wrong during task creation, the full Python traceback
# will be sent to the browser for definitive debugging.

import os
import json
import datetime
import traceback
from flask import Flask, render_template, request, jsonify

from google.cloud import storage
from google.cloud import tasks_v2
import google.auth
from google.auth.transport.requests import Request
from google.iam.credentials_v1 import iam_credentials_client

app = Flask(__name__)

# --- Configuration ---
try:
    _, GCP_PROJECT_ID = google.auth.default()
except google.auth.exceptions.DefaultCredentialsError:
    GCP_PROJECT_ID = os.environ.get('GCP_PROJECT', 'audio-mastering-v2')

BUCKET_NAME = f"{GCP_PROJECT_ID}.appspot.com"
TASK_QUEUE = 'mastering-queue'
TASK_LOCATION = 'us-east1' 

KEY_FILE_PATH = 'sa-key.json'
storage_client = storage.Client.from_service_account_json(KEY_FILE_PATH)
tasks_client = tasks_v2.CloudTasksClient()

# --- Frontend Route ---
@app.route('/')
def index():
    return render_template('index.html')

# --- API Endpoints ---
@app.route('/generate-upload-url', methods=['POST'])
def generate_upload_url():
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
        )
        
        gcs_uri = f"gs://{BUCKET_NAME}/raw_uploads/{data['filename']}"
        return jsonify({"url": url, "gcs_uri": gcs_uri}), 200

    except Exception as e:
        print("ERROR in generate_upload_url:", traceback.format_exc())
        return jsonify({"error": f"Backend Error: {traceback.format_exc()}"}), 500

@app.route('/start-processing', methods=['POST'])
def start_processing():
    try:
        data = request.get_json()
        if not data or 'gcs_uri' not in data or 'settings' not in data:
            return jsonify({"error": "Missing GCS URI or settings"}), 400
        
        parent = f"projects/{GCP_PROJECT_ID}/locations/{TASK_LOCATION}"
        queue_path = tasks_client.queue_path(GCP_PROJECT_ID, TASK_LOCATION, TASK_QUEUE)
        
        # This "get or create" logic for the queue is now covered by the new permission.
        try:
            tasks_client.get_queue(name=queue_path)
        except Exception:
            try:
                tasks_client.create_queue(parent=parent, queue={"name": queue_path})
            except Exception as create_err:
                 if "already exists" not in str(create_err):
                     raise create_err

        task = {
            "app_engine_http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "relative_uri": "/process-task",
                "headers": {"Content-type": "application/json"},
                "body": json.dumps(data).encode(),
            }
        }
        tasks_client.create_task(parent=queue_path, task=task)
        
        original_filename = data['settings'].get('original_filename', 'unknown.wav')
        processed_filename = f"processed/mastered_{original_filename}"
        return jsonify({"message": "Processing job started.", "processed_filename": processed_filename}), 200
    except Exception as e:
        # --- THE FINAL DEBUGGING FIX ---
        # If any part of the process above fails, send the full error to the browser.
        error_details = traceback.format_exc()
        print("ERROR in start_processing:", error_details)
        return jsonify({"error": f"Backend Error: {error_details}"}), 500


@app.route('/status', methods=['GET'])
def get_status():
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
        )
        return jsonify({"status": "done", "download_url": download_url}), 200
    except Exception as e:
        print("ERROR in get_status:", traceback.format_exc())
        return jsonify({"error": f"Backend Error: {traceback.format_exc()}"}), 500

@app.route('/process-task', methods=['POST'])
def process_task():
    from audio_mastering_engine import process_audio_from_gcs
    job_data = json.loads(request.data.decode('utf-8'))
    gcs_uri = job_data.get('gcs_uri')
    settings = job_data.get('settings')

    if not gcs_uri or not settings:
        print(f"ERROR: Invalid task data received: {job_data}")
        return "Bad Request: Invalid task data", 400

    try:
        print(f"Starting background processing for {gcs_uri}")
        process_audio_from_gcs(gcs_uri, settings, KEY_FILE_PATH)
        print(f"Successfully completed processing for {gcs_uri}")
        return "OK", 200
    except Exception as e:
        print(f"CRITICAL ERROR processing {gcs_uri}: {traceback.format_exc()}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

