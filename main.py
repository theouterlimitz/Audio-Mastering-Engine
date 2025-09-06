# main.py
# This is the final, production-ready code for the frontend service.

import os
import json
import datetime
from flask import Flask, render_template, request, jsonify
from google.cloud import storage
from google.cloud import tasks_v2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- THIS IS THE ROBUST AUTH FIX ---
# Construct an absolute path to the key file. This is more reliable and
# ensures the application can find its credentials on startup, fixing the 502 error.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE_PATH = os.path.join(BASE_DIR, "sa-key.json")
# --- END OF FIX ---

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT')
BUCKET_NAME = f"{GCP_PROJECT_ID}.appspot.com"
TASK_QUEUE = 'mastering-queue'
TASK_LOCATION = 'us-east1'

# Initialize Google Cloud clients using the absolute path to the service account key.
storage_client = storage.Client.from_service_account_json(KEY_FILE_PATH)
tasks_client = tasks_v2.CloudTasksClient.from_service_account_json(KEY_FILE_PATH)


@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')


@app.route('/generate-upload-url', methods=['POST'])
def generate_upload_url():
    """Generates a secure, temporary URL for the browser to upload a file directly to GCS."""
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
        return jsonify({"error": f"Backend Error in generate-upload-url: {str(e)}"}), 500


@app.route('/start-processing', methods=['POST'])
def start_processing():
    """Creates a Cloud Task to start the backend audio processing job."""
    try:
        data = request.get_json()
        if not data or 'gcs_uri' not in data or 'settings' not in data:
            return jsonify({"error": "Missing GCS URI or settings"}), 400

        parent = tasks_client.queue_path(GCP_PROJECT_ID, TASK_LOCATION, TASK_QUEUE)
        
        task_payload = {
            'gcs_uri': data['gcs_uri'],
            'settings': data['settings'],
            # The worker will construct its own absolute path to the key file.
            'key_file_path': "sa-key.json"
        }
        
        task = {
            "app_engine_http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "relative_uri": "/process-task",
                "headers": {"Content-type": "application/json"},
                "body": json.dumps(task_payload).encode(),
                "app_engine_routing": {"service": "staging-worker"}
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
        return jsonify({"error": f"Backend Error in start-processing: {str(e)}"}), 500


@app.route('/status', methods=['GET'])
def get_status():
    """Allows the frontend to poll for the status of the processing job."""
    try:
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
    except Exception as e:
        return jsonify({"error": f"Backend Error in status check: {str(e)}"}), 500


if __name__ == '__main__':
    # This block is for local development only and is ignored by Google App Engine.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

