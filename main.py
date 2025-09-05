# main.py (Final AI Version with Project ID passing)
# This version explicitly adds the GCP_PROJECT_ID to the settings payload,
# ensuring the background worker knows which project to use for AI calls.

import os
import json
import datetime
from flask import Flask, render_template, request, jsonify
import google.auth
from google.cloud import storage
from google.cloud import tasks_v2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from audio_mastering_engine import process_audio_from_gcs

app = Flask(__name__)

# --- Configuration ---
credentials, GCP_PROJECT_ID = google.auth.default()
BUCKET_NAME = f"{GCP_PROJECT_ID}.appspot.com"
TASK_QUEUE = 'mastering-queue'
TASK_LOCATION = 'us-east1' 
KEY_FILE_PATH = "sa-key.json"

storage_client = storage.Client.from_service_account_json(KEY_FILE_PATH)
tasks_client = tasks_v2.CloudTasksClient()

@app.route('/')
def index():
    return render_template('index.html')

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
        logging.exception("Error in generate_upload_url")
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500

@app.route('/start-processing', methods=['POST'])
def start_processing():
    try:
        data = request.get_json()
        if not data or 'gcs_uri' not in data or 'settings' not in data:
            return jsonify({"error": "Missing GCS URI or settings"}), 400

        data['key_file_path'] = KEY_FILE_PATH
        
        # <<< THIS IS THE FIX >>>
        data['settings']['gcp_project_id'] = GCP_PROJECT_ID

        parent = tasks_client.queue_path(GCP_PROJECT_ID, TASK_LOCATION, TASK_QUEUE)
        task = {
            "app_engine_http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "relative_uri": "/process-task",
                "headers": {"Content-type": "application/json"},
                "body": json.dumps(data).encode(),
                "app_engine_routing": { "service": "staging" }
            },
            'dispatch_deadline': {'seconds': 3600 * 2 }
        }
        tasks_client.create_task(parent=parent, task=task)
        
        original_filename = data['settings'].get('original_filename', 'unknown.wav')
        processed_filename = f"processed/mastered_{original_filename}"
        image_filename = ""
        if data['settings'].get('art_prompt'):
            image_filename = f"processed/art_{os.path.splitext(original_filename)[0]}.png"

        return jsonify({
            "message": "Processing job started.", 
            "processed_filename": processed_filename,
            "image_filename": image_filename
        }), 200

    except Exception as e:
        logging.exception("Error in start_processing")
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def get_status():
    audio_filename = request.args.get('audio_filename')
    image_filename = request.args.get('image_filename')
    if not audio_filename:
        return jsonify({"error": "Audio filename parameter is required"}), 400
    bucket = storage_client.bucket(BUCKET_NAME)
    complete_flag_blob = bucket.blob(f"{audio_filename}.complete")
    if not complete_flag_blob.exists():
        return jsonify({"status": "processing"}), 200
    audio_blob = bucket.blob(audio_filename)
    if not audio_blob.exists():
        return jsonify({"status": "error", "message": "Processing complete but output audio file is missing."}), 500
    response_data = {"status": "done"}
    response_data['download_url'] = audio_blob.generate_signed_url(
        version="v4", expiration=datetime.timedelta(minutes=60)
    )
    if image_filename and image_filename != 'null' and image_filename != '':
        image_blob = bucket.blob(image_filename)
        if image_blob.exists():
            response_data['art_url'] = image_blob.generate_signed_url(
                version="v4", expiration=datetime.timedelta(minutes=60)
            )
    return jsonify(response_data), 200

@app.route('/process-task', methods=['POST'])
def process_task():
    try:
        job_data = json.loads(request.data.decode('utf-8'))
        gcs_uri = job_data.get('gcs_uri')
        settings = job_data.get('settings')
        key_file_path = job_data.get('key_file_path')
        if not all([gcs_uri, settings, key_file_path]):
            logging.error(f"ERROR: Invalid task data received: {job_data}")
            return "Bad Request: Invalid task data", 400
        logging.info(f"Starting background processing for {gcs_uri}")
        process_audio_from_gcs(gcs_uri, settings, key_file_path)
        logging.info(f"Successfully completed processing for {gcs_uri}")
        return "OK", 200
    except Exception as e:
        logging.exception(f"CRITICAL ERROR processing {gcs_uri}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

