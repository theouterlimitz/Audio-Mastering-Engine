# frontend/main.py
# This is the final, corrected version with the robust bucket name fix.

import os
import uuid
import json
import logging
from flask import Flask, render_template, request, jsonify

from google.cloud import firestore, storage, tasks_v2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# --- THIS IS THE FIX ---
# Let the client libraries automatically detect the project ID and default bucket.
# This is the recommended and most reliable way.
try:
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
    
except Exception as e:
    logging.critical(f"FATAL: Could not initialize GCP clients: {e}")
    db, storage_client, tasks_client = None, None, None
# --- END FIX ---

@app.route('/')
def index():
    # Pass the Firebase config to the frontend template
    firebase_config = os.environ.get('FIREBASE_CONFIG_JSON')
    return render_template('index.html', firebase_config=firebase_config)

@app.route('/generate-upload-url', methods=['POST'])
def generate_upload_url():
    if not storage_client:
        return jsonify({"error": "Server is not configured correctly."}), 500

    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({"error": "Filename not provided."}), 400

    # Sanitize and create a unique blob name
    unique_id = uuid.uuid4().hex
    blob_name = f"uploads/{unique_id}/{filename}"
    
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    try:
        # Generate a signed URL for the client to upload the file directly
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=3600, # 1 hour
            method="PUT",
            content_type=data.get('contentType', 'application/octet-stream')
        )
        return jsonify({"signedUrl": signed_url, "gcsUri": f"gs://{BUCKET_NAME}/{blob_name}"})
    except Exception as e:
        logging.exception("Error generating signed URL")
        return jsonify({"error": "Could not generate upload URL."}), 500

@app.route('/submit-job', methods=['POST'])
def submit_job():
    if not all([db, tasks_client]):
        return jsonify({"error": "Server is not configured correctly."}), 500

    job_data = request.get_json()
    
    # Create a job document in Firestore
    job_ref = db.collection('mastering_jobs').document()
    job_data['job_id'] = job_ref.id
    job_ref.set({
        'status': 'Pending',
        'submitted_at': firestore.SERVER_TIMESTAMP,
        'settings': job_data.get('settings', {}),
        'gcs_uri': job_data.get('gcs_uri')
    })

    # Create a task to be sent to the worker service
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"https://worker-dot-{GCP_PROJECT_ID}.{GCP_REGION}.r.appspot.com/process-task",
            "oidc_token": {
                "service_account_email": f"{GCP_PROJECT_ID}@appspot.gserviceaccount.com"
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
