# main.py (Final AI Version)
# This version correctly handles the filenames and status checks for both
# the processed audio and the generated AI art, resolving the status check error.

import os
import json
import datetime
from flask import Flask, render_template, request, jsonify
import google.auth
from google.cloud import storage
from google.cloud import tasks_v2
import logging

# --- Set up professional logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import our audio engine
from audio_mastering_engine import process_audio_from_gcs

app = Flask(__name__)

# --- Configuration ---
# Use the application's default project ID and service account
credentials, GCP_PROJECT_ID = google.auth.default()
BUCKET_NAME = f"{GCP_PROJECT_ID}.appspot.com"
TASK_QUEUE = 'mastering-queue'
TASK_LOCATION = 'us-east1' 
KEY_FILE_PATH = "sa-key.json"

# Initialize clients using the service account key for signing
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

        # --- THIS IS THE KEY CHANGE ---
        # We now pass the key file path to the background task
        data['key_file_path'] = KEY_FILE_PATH

        parent = tasks_client.queue_path(GCP_PROJECT_ID, TASK_LOCATION, TASK_QUEUE)
        task = {
            "app_engine_http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "relative_uri": "/process-task",
                "headers": {"Content-type": "application/json"},
                "body": json.dumps(data).encode(),
                "app_engine_routing": { "service": "staging" }
            },
            'dispatch_deadline': {'seconds': 3600 * 2 } # 2 hours
        }
        tasks_client.create_task(parent=parent, task=task)
        
        # --- NEW LOGIC TO HANDLE BOTH FILENAMES ---
        original_filename = data['settings'].get('original_filename', 'unknown.wav')
        processed_filename = f"processed/mastered_{original_filename}"
        
        image_filename = "" # Default to empty string
        if data['settings'].get('art_prompt'):
            # Construct the expected image filename if a prompt was provided
            image_filename = f"processed/art_{os.path.splitext(original_filename)[0]}.png"

        return jsonify({
            "message": "Processing job started.", 
            "processed_filename": processed_filename,
            "image_filename": image_filename # Send the expected image filename back to the browser
        }), 200

    except Exception as e:
        logging.exception("Error in start_processing")
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500


@app.route('/status', methods=['GET'])
def get_status():
    # --- NEW LOGIC TO CHECK FOR BOTH FILES ---
    audio_filename = request.args.get('audio_filename')
    image_filename = request.args.get('image_filename') # Get the expected image filename
    
    if not audio_filename:
        return jsonify({"error": "Audio filename parameter is required"}), 400
        
    bucket = storage_client.bucket(BUCKET_NAME)
    
    complete_flag_blob = bucket.blob(f"{audio_filename}.complete")
    if not complete_flag_blob.exists():
        return jsonify({"status": "processing"}), 200

    audio_blob = bucket.blob(audio_filename)
    if not audio_blob.exists():
        return jsonify({"status": "error", "message": "Processing complete but output audio file is missing."}), 500

    # Prepare response
    response_data = {"status": "done"}
    
    # Generate audio download URL
    response_data['download_url'] = audio_blob.generate_signed_url(
        version="v4", expiration=datetime.timedelta(minutes=60)
    )

    # If an image was expected, try to get its URL too
    if image_filename and image_filename != 'null':
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

