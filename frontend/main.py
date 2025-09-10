# frontend/main.py
# This is the final, clean version of the "Waiter" service.
# It is a lightweight Flask app responsible for serving the UI and
# orchestrating the cloud services (GCS, Cloud Tasks, Firestore).

import os
import json
import datetime
import uuid
from flask import Flask, render_template, request, jsonify, g

# Import Google Cloud libraries
from google.cloud import storage
from google.cloud import tasks_v2
from google.cloud import firestore

# --- Configuration & Initialization ---
app = Flask(__name__, template_folder='templates')

# --- GCP Client Caching ---
def get_gcp_clients():
    if 'storage_client' not in g:
        g.storage_client = storage.Client()
        g.tasks_client = tasks_v2.CloudTasksClient()
        g.db = firestore.Client()
    return g.storage_client, g.tasks_client, g.db

# --- Environment Variables ---
GCP_PROJECT = os.environ.get('GCP_PROJECT')
GCP_REGION = os.environ.get('GCP_REGION', 'us-central1')
TASK_QUEUE = os.environ.get('TASK_QUEUE', 'mastering-queue')
BUCKET_NAME = f"{GCP_PROJECT}.appspot.com" if GCP_PROJECT else None

# --- Main Route ---
@app.route('/')
def index():
    """Serves the main web page."""
    firebase_config = os.environ.get('FIREBASE_CONFIG_JSON')
    return render_template('index.html', firebase_config=firebase_config)

# --- API Endpoints ---
@app.route('/generate-upload-url', methods=['POST'])
def generate_upload_url():
    storage_client, _, _ = get_gcp_clients()
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({"error": "Filename not provided"}), 400
    if not BUCKET_NAME:
        return jsonify({"error": "Server is not configured with a bucket name."}), 500

    unique_id = uuid.uuid4().hex
    original_filename = data['filename']
    blob_name = f"raw_uploads/{unique_id}/{original_filename}"
    
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    try:
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=15),
            method="PUT",
            content_type=data.get('contentType', 'application/octet-stream'),
        )
        gcs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
        return jsonify({"url": url, "gcs_uri": gcs_uri}), 200
    except Exception as e:
        app.logger.error(f"Error generating signed URL: {e}")
        return jsonify({"error": "Could not generate upload URL."}), 500

@app.route('/start-processing', methods=['POST'])
def start_processing():
    _, tasks_client, db = get_gcp_clients()
    data = request.get_json()
    if not data or 'gcs_uri' not in data or 'settings' not in data:
        return jsonify({"error": "Missing GCS URI or settings"}), 400

    job_id = str(uuid.uuid4())
    
    try:
        job_ref = db.collection('mastering_jobs').document(job_id)
        job_ref.set({
            'status': 'Job created. Waiting for worker...',
            'created_at': firestore.SERVER_TIMESTAMP,
            'progress': 0,
            'original_filename': data['settings']['original_filename'],
            'settings': data['settings']
        })

        queue_path = tasks_client.queue_path(GCP_PROJECT, GCP_REGION, TASK_QUEUE)
        
        task_payload = {
            'job_id': job_id,
            'gcs_uri': data['gcs_uri'],
            'settings': data['settings']
        }
        
        task = {
            "app_engine_http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "relative_uri": "/process-task",
                "body": json.dumps(task_payload).encode('utf-8'),
                "headers": {"Content-Type": "application/json"},
                "app_engine_routing": {"service": "worker"}
            }
        }
        
        tasks_client.create_task(parent=queue_path, task=task)
        
        return jsonify({"job_id": job_id}), 200

    except Exception as e:
        app.logger.error(f"Error starting processing job {job_id}: {e}")
        if 'job_ref' in locals():
            job_ref.set({'status': 'Error', 'error_message': 'Failed to create task.'})
        return jsonify({"error": "Could not start processing job."}), 500

@app.route('/get-signed-download-url', methods=['POST'])
def get_signed_download_url():
    storage_client, _, _ = get_gcp_clients()
    data = request.get_json()
    gcs_path = data.get('gcs_path')
    if not gcs_path:
        return jsonify({"error": "GCS path not provided"}), 400

    try:
        blob_name = gcs_path.replace(f"gs://{BUCKET_NAME}/", "")
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            return jsonify({"error": "File not found"}), 404

        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(hours=1),
            method="GET",
        )
        return jsonify({"url": url}), 200
    except Exception as e:
        app.logger.error(f"Error generating download URL for {gcs_path}: {e}")
        return jsonify({"error": "Could not generate download URL"}), 500

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=PORT, debug=True)
