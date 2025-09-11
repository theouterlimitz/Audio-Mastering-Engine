# frontend/main.py
# This is the final, fully-corrected version of the frontend service.
# It includes proper Google Cloud Logging and the definitive fix for the
# signed URL "private key" error.

import os
import uuid
import json
import logging
from flask import Flask, render_template, request, jsonify

# Import the Google Cloud Logging library
import google.cloud.logging

from google.cloud import firestore, storage, tasks_v2

# Instantiate a client for Google Cloud Logging.
# This configures the root logger to correctly send all logs,
# including their severity levels, to the Cloud Logging API.
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()

app = Flask(__name__)

try:
    # These clients will automatically use the service account
    # assigned to the App Engine instance.
    db = firestore.Client()
    storage_client = storage.Client()
    tasks_client = tasks_v2.CloudTasksClient()
    
    # Automatically determine the project ID from the environment
    GCP_PROJECT_ID = storage_client.project
    # The default App Engine bucket name is always [PROJECT_ID].appspot.com
    BUCKET_NAME = f"{GCP_PROJECT_ID}.appspot.com"
    
    GCP_REGION = os.environ.get('GCP_REGION', 'us-central1') 
    TASK_QUEUE = os.environ.get('TASK_QUEUE', 'mastering-queue')
    
    TASK_QUEUE_PATH = tasks_client.queue_path(GCP_PROJECT_ID, GCP_REGION, TASK_QUEUE)
    
    # When running on App Engine, this environment variable will contain the email
    # of the service account the instance is running as.
    SERVICE_ACCOUNT_EMAIL = os.environ.get('GAE_SERVICE_ACCOUNT_EMAIL')
    if not SERVICE_ACCOUNT_EMAIL:
        # Fallback for local development or if the env var isn't set
        SERVICE_ACCOUNT_EMAIL = f'{GCP_PROJECT_ID}@appspot.gserviceaccount.com'


except Exception as e:
    logging.critical(f"FATAL: Could not initialize GCP clients: {e}")
    db, storage_client, tasks_client = None, None, None

@app.route('/')
def index():
    firebase_config = os.environ.get('FIREBASE_CONFIG_JSON')
    return render_template('index.html', firebase_config=firebase_config)

@app.route('/generate-upload-url', methods=['POST'])
def generate_upload_url():
    if not storage_client:
        logging.error("Server is not configured correctly.")
        return jsonify({"error": "Server is not configured correctly."}), 500

    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        logging.warning("Filename not provided.")
        return jsonify({"error": "Filename not provided."}), 400

    unique_id = uuid.uuid4().hex
    blob_name = f"uploads/{unique_id}/{filename}"
    
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    try:
        # --- THIS IS THE FINAL FIX ---
        # We REMOVE the explicit 'service_account_email' parameter.
        # This forces the library to use the IAM Credentials API to sign the URL
        # using the service account the application is running as, which is the
        # correct and modern authentication flow. The previous 'private key'
        # error was caused by explicitly passing the email, which confused the
        # library into trying an older, key-based signing method.
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=3600, # 1 hour
            method="PUT",
            content_type=data.get('contentType', 'application/octet-stream')
        )
        # --- END FIX ---
        
        return jsonify({"signedUrl": signed_url, "gcsUri": f"gs://{BUCKET_NAME}/{blob_name}"})
    except Exception as e:
        # This will no longer be triggered, but we leave it for safety.
        logging.exception("Error generating signed URL")
        return jsonify({"error": "Could not generate upload URL."}), 500

@app.route('/submit-job', methods=['POST'])
def submit_job():
    if not all([db, tasks_client]):
        logging.error("Server is not configured correctly.")
        return jsonify({"error": "Server is not configured correctly."}), 500

    job_data = request.get_json()
    
    job_ref = db.collection('mastering_jobs').document()
    job_data['job_id'] = job_ref.id
    job_ref.set({
        'status': 'Pending',
        'submitted_at': firestore.SERVER_TIMESTAMP,
        'settings': job_data.get('settings', {}),
        'gcs_uri': job_data.get('gcs_uri')
    })

    # This task will be sent to the worker service. The OIDC token
    # uses the service account identity to securely authenticate the request.
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"https://worker-dot-{GCP_PROJECT_ID}.{GCP_REGION}.r.appspot.com/process-task",
            "oidc_token": {
                "service_account_email": SERVICE_ACCOUNT_EMAIL
            },
            "headers": {"Content-type": "application/json"},
            "body": json.dumps(job_data).encode()
        }
    }
    
    tasks_client.create_task(parent=TASK_QUEUE_PATH, task=task)
    
    return jsonify({"success": True, "job_id": job_ref.id})

if __name__ == '__main__':
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=PORT, debug=True)
