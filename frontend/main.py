# frontend/main.py
# This is the final, definitive version. It uses a service account key
# from Secret Manager to guarantee that signed URLs can be generated.

import os
import uuid
import json
import logging
import google.auth
from google.oauth2 import service_account
from flask import Flask, render_template, request, jsonify

import google.cloud.logging
from google.cloud import firestore, storage, tasks_v2
from google.cloud import secretmanager

logging_client = google.cloud.logging.Client()
logging_client.setup_logging()

app = Flask(__name__)

# --- Client Initialization ---
# This block runs once when the container starts. It fetches the key
# and creates all clients with explicit, key-based credentials.
try:
    # Get the full secret path from the environment variable set during deployment
    secret_version_id = os.environ.get("SA_KEY_SECRET_ID")
    if not secret_version_id:
        raise RuntimeError("SA_KEY_SECRET_ID environment variable not set.")

    # Create a client for Secret Manager
    secret_client = secretmanager.SecretManagerServiceClient()
    
    # Access the secret
    response = secret_client.access_secret_version(name=secret_version_id)
    secret_json_string = response.payload.data.decode("UTF-8")
    secret_info = json.loads(secret_json_string)
    
    # Create credentials from the key file info
    credentials = service_account.Credentials.from_service_account_info(secret_info)
    
    # Initialize all other clients using these explicit credentials
    GCP_PROJECT_ID = credentials.project_id
    db = firestore.Client(project=GCP_PROJECT_ID, credentials=credentials)
    storage_client = storage.Client(project=GCP_PROJECT_ID, credentials=credentials)
    tasks_client = tasks_v2.CloudTasksClient(credentials=credentials)
    
    BUCKET_NAME = f"{GCP_PROJECT_ID}.appspot.com"
    GCP_REGION = os.environ.get('GCP_REGION', 'us-central1')
    TASK_QUEUE = os.environ.get('TASK_QUEUE', 'mastering-queue')
    TASK_QUEUE_PATH = tasks_client.queue_path(GCP_PROJECT_ID, GCP_REGION, TASK_QUEUE)
    SERVICE_ACCOUNT_EMAIL = credentials.service_account_email

    logging.info("Successfully initialized all GCP clients using the service account key from Secret Manager.")

except Exception:
    logging.exception("FATAL: A critical error occurred during client initialization.")
    db, storage_client, tasks_client = None, None, None
# --- End Initialization ---


@app.route('/')
def index():
    firebase_config = os.environ.get('FIREBASE_CONFIG_JSON')
    return render_template('index.html', firebase_config=firebase_config)

@app.route('/generate-upload-url', methods=['POST'])
def generate_upload_url():
    if not storage_client:
        logging.error("Server is not configured correctly; storage_client is None.")
        return jsonify({"error": "Configuration error. Please check logs."}), 500

    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({"error": "Filename not provided."}), 400

    unique_id = uuid.uuid4().hex
    blob_name = f"uploads/{unique_id}/{filename}"
    
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    try:
        # This will now succeed because our storage_client was created with a private key.
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=3600,
            method="PUT",
            content_type=data.get('contentType', 'application/octet-stream')
        )
        return jsonify({"signedUrl": signed_url, "gcsUri": f"gs://{BUCKET_NAME}/{blob_name}"})
    except Exception:
        logging.exception("CRITICAL: Signed URL generation failed even with explicit key.")
        return jsonify({"error": "Could not generate upload URL."}), 500

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