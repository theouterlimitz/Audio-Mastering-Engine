# Python Audio Mastering Suite (Cloud & Desktop)

This repository contains the journey and code for a Python-based audio mastering suite, evolving from a local desktop application to a scalable, cloud-native web service on Google Cloud Platform.

## Version 3: Cloud-Native Web Application on Google App Engine

This is the flagship version of the project: a fully scalable, web-based audio mastering tool. Users can upload an audio file from any modern web browser, set mastering parameters, and receive a professionally mastered track, all powered by a robust backend on Google Cloud.

The live application can be found here: (https://audio-mastering-v2.ue.r.appspot.com/)

### Features

*   **Modern Web UI**: A clean, responsive interface built with HTML and Tailwind CSS.
*   **Scalable Backend**: Built with Python, Flask, and Gunicorn on Google App Engine's Flexible Environment.
*   **Secure File Handling**: Uses signed URLs for secure, direct-to-cloud uploads, keeping large files off the application server.
*   **Asynchronous Processing**: Leverages Cloud Tasks to run CPU-intensive audio mastering jobs in the background, ensuring the web app is always responsive.
*   **Custom Runtime**: Uses a Docker container to include the essential FFmpeg dependency for audio processing.
*   **Dedicated Identity**: Employs a dedicated Service Account with a private key for robust and secure authentication with Google Cloud services.

### High-Level Architecture

The cloud application follows a modern, decoupled architecture:

1.  **Frontend (Browser)** requests a secure upload URL from the App Engine backend.
2.  **Browser** uploads the raw audio file directly to a private Google Cloud Storage (GCS) bucket.
3.  **Browser** notifies the backend that the upload is complete.
4.  **Backend** creates a job in a Cloud Tasks queue with the file details and user settings.
5.  **App Engine** automatically scales up a new worker instance to process the job.
6.  The **Worker** downloads the raw file from GCS, runs the Python mastering engine, and uploads the processed file back to GCS.
7.  **Frontend** periodically polls a status endpoint. Once the worker is finished, it receives a secure download URL for the mastered file.

### Deployment

Deploying this application requires a Google Cloud Platform project with billing enabled.

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/theouterlimitz/Audio-Mastering-Engine.git
    cd Audio-Mastering-Engine
    ```

2.  **Google Cloud Project Setup**:
    *   Create a new GCP Project.
    *   Enable billing for the project.
    *   Enable the following APIs:
        *   App Engine Admin API
        *   Cloud Tasks API
        *   Cloud Build API
        *   IAM Credentials API
    *   Create an App Engine application within the project:
        ```bash
        gcloud app create --region=us-east1
        ```

3.  **Create a Dedicated Service Account & Key**:
    *   Create the service account:
        ```bash
        gcloud iam service-accounts create audio-mastering-sa --display-name="Audio Mastering SA"
        ```
    *   Create and download a private key. This will create the `sa-key.json` file in your project root.
        ```bash
        gcloud iam service-accounts keys create sa-key.json --iam-account="audio-mastering-sa@[YOUR_PROJECT_ID].iam.gserviceaccount.com"
        ```

4.  **Configure IAM Permissions**:
    *   Grant the new service account the necessary roles. Replace `[YOUR_PROJECT_ID]` with your GCP Project ID.
        ```bash
        # Role for managing GCS buckets
        gcloud projects add-iam-policy-binding [YOUR_PROJECT_ID] --member="serviceAccount:audio-mastering-sa@[YOUR_PROJECT_ID].iam.gserviceaccount.com" --role="roles/storage.admin"

        # Role for creating Cloud Tasks
        gcloud projects add-iam-policy-binding [YOUR_PROJECT_ID] --member="serviceAccount:audio-mastering-sa@[YOUR_PROJECT_ID].iam.gserviceaccount.com" --role="roles/cloudtasks.queueAdmin"

        # Role for viewing App Engine to create tasks targeted at it
        gcloud projects add-iam-policy-binding [YOUR_PROJECT_ID] --member="serviceAccount:audio-mastering-sa@[YOUR_PROJECT_ID].iam.gserviceaccount.com" --role="roles/viewer"
        ```

5.  **Create the Task Queue**:
    ```bash
    gcloud tasks queues create mastering-queue
    ```

6.  **Configure CORS for the Storage Bucket**:
    *   Get your app's URL: `gcloud app browse`
    *   Create a file named `cors.json` with the following content, replacing the origin URL with your app's URL:
        ```json
        [
          {
            "origin": ["https://YOUR-APP-URL.com"],
            "method": ["PUT", "GET"],
            "responseHeader": ["Content-Type"],
            "maxAgeSeconds": 3600
          }
        ]
        ```
    *   Apply the policy to your bucket (replace `[YOUR_PROJECT_ID]`):
        ```bash
        gcloud storage buckets update gs://[YOUR_PROJECT_ID].appspot.com --cors-file=cors.json
        ```

7.  **Deploy the Application**:
    *   Make sure the `sa-key.json` file is present in your project root.
    *   ```bash
        gcloud app deploy app.yaml
        ```

---

## Version 2: Local Desktop GUI

This is the original version of the project, a standalone desktop application for Windows, macOS, and Linux, built with Python and Tkinter.

### Features

*   **Intuitive Graphical Interface**: A clean, dark-themed UI built with Tkinter.
*   **Analog Character Control**: Blends harmonic saturation, a sub-bass bump, and high-end sparkle.
*   **Multiband Compressor**: A 3-band compressor with detailed controls.
*   **Professional EQ & Widening**: A 4-band equalizer and a stereo widener.
*   **Streaming-Ready Loudness**: LUFS normalization to target modern streaming platform levels.

### Installation & Setup

*   **Python Environment**: It is recommended to use a dedicated environment (e.g., with Conda or venv) using Python 3.11.
*   **Install Libraries**:
    ```bash
    pip install pydub scipy numpy pyloudnorm ttkthemes
    ```
*   **Install FFmpeg**: The audio engine requires FFmpeg.
    *   **Debian/Ubuntu**: `sudo apt install ffmpeg`
    *   **macOS (Homebrew)**: `brew install ffmpeg`
    *   **Windows**: Download binaries from the official site and add to your system's PATH.
*   **Run the Application**:
    ```bash
    python mastering_gui.py
    ```

---

## Future Roadmap

The journey isn't over! The cloud-native architecture opens up exciting possibilities for integrating AI and machine learning.

*   **AI-Generated Cover Art**: The next major planned feature is to integrate an image generation model (like Google's Imagen) into the workflow.
    *   **Phase 1**: Allow users to provide a text prompt to generate album art alongside their mastered track.
    *   **Phase 2**: Implement an "automatic mood" feature by analyzing the audio file's characteristics (tempo, key, energy) and feeding a creatively generated prompt to the image model.
*   **Security Hardening**:
    *   Implement user authentication
