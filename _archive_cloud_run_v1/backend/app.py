# backend/app.py
#
# This is a special diagnostic script (a "black box recorder").
# Its only purpose is to log every step of the /start-processing
# function to help us find the exact line that is causing a silent crash.

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def hello_world():
    print("HEALTH CHECK: Root URL was hit successfully.")
    return "Diagnostic server is running."

# This is a dummy endpoint to keep the frontend happy for now.
@app.route('/generate-upload-url', methods=['POST'])
def generate_upload_url():
    print("DIAGNOSTIC: /generate-upload-url was called, returning dummy data.")
    return jsonify({"url": "http://example.com", "gcs_uri": "gs://dummy/dummy.wav"}), 200

# THIS IS THE FUNCTION WE ARE INVESTIGATING
@app.route('/start-processing', methods=['POST'])
def start_processing():
    print("STEP 1: Entered /start-processing function.")
    try:
        from google.cloud import pubsub_v1
        print("STEP 2: Successfully imported pubsub_v1.")

        GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'tactile-temple-395019')
        print(f"STEP 3: GCP_PROJECT_ID is '{GCP_PROJECT_ID}'.")

        PUB_SUB_TOPIC = os.environ.get('PUB_SUB_TOPIC', 'mastering-jobs')
        print(f"STEP 4: PUB_SUB_TOPIC is '{PUB_SUB_TOPIC}'.")

        data = request.get_json()
        print(f"STEP 5: Received job data: {data}")
        if not data:
            print("ERROR: No JSON data received.")
            return jsonify({"error": "No JSON data"}), 400

        print("STEP 6: Creating PublisherClient.")
        publisher = pubsub_v1.PublisherClient()
        print("STEP 7: PublisherClient created successfully.")
        
        topic_path = publisher.topic_path(GCP_PROJECT_ID, PUB_SUB_TOPIC)
        print(f"STEP 8: Constructed topic path: {topic_path}")
        
        message_data = json.dumps(data).encode("utf-8")
        print("STEP 9: Encoded message data.")

        print("STEP 10: Publishing message to Pub/Sub...")
        future = publisher.publish(topic_path, message_data)
        print("STEP 11: Publish command sent. Awaiting result...")
        
        future.result() # This is where the original error was.
        print("STEP 12: SUCCESS! Message published successfully.")
        
        # We need to return the expected JSON key
        original_filename = data.get('settings', {}).get('original_filename', 'unknown.wav')
        processed_filename = f"processed/mastered_{original_filename}"
        return jsonify({"message": "Diagnostic job started.", "processed_filename": processed_filename}), 200

    except Exception as e:
        print(f"CRITICAL ERROR in /start-processing: {e}")
        # Also print the full traceback to the logs
        import traceback
        traceback.print_exc()
        return jsonify({"error": "A critical error occurred. Check the server logs."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))