# worker.py
# This is a dedicated service for running the heavy audio processing task.
# It only contains the /process-task endpoint.

import os
import json
import logging
from flask import Flask, request

# The actual processing logic is in a separate file for cleanliness.
from audio_mastering_engine import process_audio_from_gcs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

@app.route('/process-task', methods=['POST'])
def process_task():
    try:
        job_data = json.loads(request.data.decode('utf-8'))
        gcs_uri = job_data.get('gcs_uri')
        settings = job_data.get('settings')
        key_file_path = job_data.get('key_file_path')

        if not all([gcs_uri, settings, key_file_path]):
            logging.error(f"WORKER ERROR: Invalid task data received: {job_data}")
            return "Bad Request: Invalid task data", 400
            
        logging.info(f"WORKER: Starting background processing for {gcs_uri}")
        process_audio_from_gcs(gcs_uri, settings, key_file_path)
        logging.info(f"WORKER: Successfully completed processing for {gcs_uri}")
        
        return "OK", 200
        
    except Exception as e:
        logging.exception(f"WORKER CRITICAL ERROR processing {gcs_uri}")
        # IMPORTANT: Return a 500 error to tell Cloud Tasks the job failed.
        return "Internal Server Error", 500

# This is only for local testing, Gunicorn will run the app in production.
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8081)))