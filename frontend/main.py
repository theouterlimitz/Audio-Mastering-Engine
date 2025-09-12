# frontend/main.py
# This is the final, definitive, simplified version. It uses a robust
# Signed Policy Document to handle file uploads.

import os
import uuid
import json
import logging
import datetime
import google.auth
from flask import Flask, render_template, request, jsonify

import google.cloud.logging
from google.cloud import firestore, storage, tasks_v2

# Setup proper cloud logging immediately.
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()

app = Flask(__name__)

# --- Client Initialization ---
# This block runs once when the container starts.
try:
    credentials, GCP_PROJECT_ID = google.auth.default()

    db = firestore.Client(project=GCP_PROJECT_ID, credentials=credentials)
    storage_client = storage.Client(project=GCP_PROJECT_ID, credentials=credentials)
    tasks_client = tasks_v2.CloudTasksClient(credentials=credentials)
    
    BUCKET_NAME = f"{GCP_PROJECT_ID}.appspot.com"
    GCP_REGION = os.environ.get('GCP_REGION', 'us-central1')
    TASK_QUEUE = os.environ.get('TASK_QUEUE', 'mastering-queue')
    TASK_QUEUE_PATH = tasks_client.queue_path(GCP_PROJECT_ID, GCP_REGION, TASK_QUEUE)
    
    SERVICE_ACCOUNT_EMAIL = credentials.service_account_email

    logging.info(f"Successfully initialized all GCP clients. Running as: {SERVICE_ACCOUNT_EMAIL}")

except Exception:
    logging.exception("FATAL: A critical error occurred during client initialization.")
    db, storage_client, tasks_client = None, None, None
# --- End Initialization ---


@app.route('/')
def index():
    firebase_config = os.environ.get('FIREBASE_CONFIG_JSON')
    return render_template('index.html', firebase_config=firebase_config)

# --- THIS IS THE FINAL FIX ---
# This endpoint now generates a Signed Policy Document, which is a more
# robust method for browser-based uploads than a simple V4 Signed URL.
@app.route('/generate-upload-url', methods=['POST'])
def generate_upload_url():
    if not storage_client:
        logging.error("Server is not configured correctly; storage_client is None.")
        return jsonify({"error": "Configuration error. Please check logs."}), 500

    data = request.get_json()
    filename = data.get('filename')
    content_type = data.get('contentType', 'application/octet-stream')
    if not filename:
        return jsonify({"error": "Filename not provided."}), 400

    unique_id = uuid.uuid4().hex
    blob_name = f"uploads/{unique_id}/{filename}"
    
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    try:
        # Create the policy document for a POST request
        policy = blob.generate_signed_post_policy_v4(
            expiration=datetime.timedelta(minutes=15),
            fields={
                "Content-Type": content_type,
            },
        )
        # The frontend will receive the URL to POST to, and the required form fields.
        return jsonify({"url": policy["url"], "fields": policy["fields"], "gcsUri": f"gs://{BUCKET_NAME}/{blob_name}"})
    except Exception:
        logging.exception("CRITICAL: Signed Policy Document generation failed.")
        return jsonify({"error": "Could not generate upload URL."}), 500
# --- END FIX ---


@app.route('/submit-job', methods=['POST'])
def submit_job():
    if not all([db, tasks_client, TASK_QUEUE_PATH, GCP_PROJECT_ID, GCP_REGION, SERVICE_ACCOUNT_EMAIL]):
        logging.error("Server is not configured correctly; one or more clients/variables are None.")
        return jsonify({"error": "Configuration error. Please check logs."}), 500

    job_data = request.get_json()
    
    job_ref = db.collection('mastering_jobs').document()
    job_data['job_id'] = job_ref.id
    job_ref.set({
        'status': 'Pending',
        'submitted_at': firestore.SERVER_TIMESTAMP,
        'settings': job_data.get('settings', {}),
        'gcs_uri': job_data.get('gcs_uri')
    })

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