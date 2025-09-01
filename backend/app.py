# backend/app.py
# Final version using Secret Manager and the direct-to-cloud architecture.
# This is the complete and correct code for the public-facing API server.

import os
import json
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# These libraries will be imported and used by our functions.
from google.cloud import storage, pubsub_v1, secretmanager
from google.oauth2 import service_account

app = Flask(__name__)
# This CORS configuration allows your Netlify frontend to communicate with this backend.
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Configuration ---
# These values are specific to your GCP project.
GCP_PROJECT_ID = 'tactile-temple-395019'
BUCKET_NAME = 'tactile-temple-395019-audio-uploads'
PUB_SUB_TOPIC = 'mastering-jobs'
SECRET_ID = 'backend-sa-key'
SECRET_VERSION = 'latest'

# Use a global variable to cache the powerful credentials from the vault for efficiency.
_credentials = None

def get_credentials_from_secret():
    """
    Securely accesses the master key from Secret Manager on startup.
    This is the definitive authentication method.
    """
    global _credentials
    if _credentials:
        return _credentials
    
    try:
        print("Attempting to load credentials from Secret Manager...")
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{GCP_PROJECT_ID}/secrets/{SECRET_ID}/versions/{SECRET_VERSION}"
        response = client.access_secret_version(request={"name": name})
        creds_json = json.loads(response.payload.data.decode("UTF-8"))
        credentials = service_account.Credentials.from_service_account_info(creds_json)
        _credentials = credentials
        print("Successfully loaded credentials from Secret Manager.")
        return _credentials
        
    except Exception as e:
        print(f"CRITICAL STARTUP ERROR: Could not access secret '{SECRET_ID}'. The server cannot start.")
        print(f"Error details: {e}")
        # This will cause the container to crash, and this error will be in the logs.
        raise e

# Initialize credentials on startup to ensure the service is healthy from the start.
# If this fails, the container will crash, and the logs will clearly show the error.
CREDENTIALS = get_credentials_from_secret()

@app.route('/')
def hello_world():
    return "Audio Mastering Backend is running."

@app.route('/generate-upload-url', methods=['POST'])
def generate_upload_url():
    """Generates a secure, short-lived URL for the client to upload a file directly to GCS."""
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({"error": "Filename not provided"}), 400

        storage_client = storage.Client(credentials=CREDENTIALS, project=GCP_PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(data['filename'])

        # Generate a V4 signed URL, the modern and secure standard.
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=15),
            method="PUT",
            content_type=data.get('contentType', 'application/octet-stream'),
        )
        
        gcs_uri = f"gs://{BUCKET_NAME}/{data['filename']}"
        return jsonify({"url": url, "gcs_uri": gcs_uri}), 200

    except Exception as e:
        print(f"ERROR in /generate-upload-url: {e}")
        return jsonify({"error": "Internal server error generating upload URL."}), 500

@app.route('/start-processing', methods=['POST'])
def start_processing():
    """Receives confirmation of a successful upload and publishes a job to Pub/Sub."""
    try:
        data = request.get_json()
        if not data or 'gcs_uri' not in data or 'settings' not in data:
            return jsonify({"error": "Missing GCS URI or settings"}), 400

        publisher = pubsub_v1.PublisherClient(credentials=CREDENTIALS)
        topic_path = publisher.topic_path(GCP_PROJECT_ID, PUB_SUB_TOPIC)
        
        message_data = json.dumps(data).encode("utf-8")
        
        future = publisher.publish(topic_path, message_data)
        future.result() # Wait for the message to be published.

        original_filename = data['settings'].get('original_filename', 'unknown.wav')
        processed_filename = f"processed/mastered_{original_filename}"
        
        return jsonify({"message": "Processing job started.", "processed_filename": processed_filename}), 200

    except Exception as e:
        print(f"ERROR in /start-processing: {e}")
        return jsonify({"error": "Internal server error starting processing job."}), 500
        
@app.route('/status', methods=['GET'])
def get_status():
    """Checks if a processed file exists and provides a download link."""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({"error": "Filename parameter is required"}), 400
        
    try:
        storage_client = storage.Client(credentials=CREDENTIALS, project=GCP_PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Check for the ".complete" flag file first.
        complete_flag_blob = bucket.blob(f"{filename}.complete")
        if not complete_flag_blob.exists():
            return jsonify({"status": "processing"}), 200

        # If the flag exists, generate the download URL for the actual audio file.
        audio_blob = bucket.blob(filename)
        if not audio_blob.exists():
             return jsonify({"status": "error", "message": "Processing complete but output file is missing."}), 404
        
        download_url = audio_blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=60), # Link is valid for 1 hour
            method="GET",
        )
        return jsonify({"status": "done", "download_url": download_url}), 200

    except Exception as e:
        print(f"ERROR in /status check: {e}")
        return jsonify({"status": "error", "message": "Internal server error checking status."}), 500

if __name__ == '__main__':
    # This part is used for local testing, which we won't need for cloud deployment.
    # The Gunicorn command in the Dockerfile is what runs the app in the cloud.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
