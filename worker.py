# worker.py
# This is the complete, final code for the powerful backend service.

import os
import json
import logging
from flask import Flask, request
from audio_mastering_engine import process_audio_from_gcs

# Configure logging to make sure we see everything in the Cloud logs.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
)

app = Flask(__name__)

@app.route('/process-task', methods=['POST'])
def process_task():
    """
    This is the entrypoint for the Cloud Task. It receives job data
    and kicks off the long-running audio processing function.
    """
    try:
        # The task payload is sent as the request body.
        job_data = json.loads(request.data.decode('utf-8'))
        
        # --- THIS IS THE FIX ---
        # We now correctly receive the gcs_uri, settings, AND the key_file_path
        # from the task payload created by main.py.
        gcs_uri = job_data.get('gcs_uri')
        settings = job_data.get('settings')
        key_file_path = job_data.get('key_file_path')
        # --- END OF FIX ---

        if not all([gcs_uri, settings, key_file_path]):
            logging.error(f"ERROR: Invalid task data received: {job_data}")
            # Return a 400 error to signal a bad request to Cloud Tasks.
            return "Bad Request: Invalid task data", 400
            
        logging.info(f"Starting background processing for {gcs_uri}")
        
        # Add the GCP Project ID to the settings for the audio engine to use.
        # This is read automatically from the App Engine environment.
        settings['gcp_project_id'] = os.environ.get('GCP_PROJECT')
        
        # Call the main processing function from the other file, passing all
        # the necessary information, including the path to the credentials.
        process_audio_from_gcs(gcs_uri, settings, key_file_path)
        
        logging.info(f"Successfully completed processing for {gcs_uri}")
        
        # Return a 200 OK to signal success to Cloud Tasks.
        return "OK", 200
        
    except Exception as e:
        # If any exception occurs, log it critically so we can see the traceback.
        logging.exception(f"CRITICAL ERROR processing task.")
        
        # Return a 500 Internal Server Error to signal failure to Cloud Tasks.
        # This will cause the task to be retried according to queue settings.
        return "Internal Server Error", 500

if __name__ == '__main__':
    # This block is for local development only and is ignored by Google App Engine.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

