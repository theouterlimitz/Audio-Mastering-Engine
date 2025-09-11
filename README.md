# AI-Enhanced Audio Mastering Suite

An all-in-one desktop application for mastering audio tracks, enhanced with AI-powered analysis and cover art generation. This tool provides a user-friendly interface to apply professional-grade audio processing and automatically create unique, mood-matching artwork for your music.

## Features

*   **Comprehensive Audio Mastering:** Adjust a wide range of parameters including:
    *   **EQ:** Bass, Mid Cut, Presence, and Treble controls.
    *   **Dynamics:** Multiband compression with detailed threshold and ratio controls for low, mid, and high frequencies.
    *   **Loudness:** Target LUFS normalization for consistent volume.
    *   **Stereo Image:** Adjust the stereo width of your track.
    *   **Analog Character:** Add warmth and subtle saturation.
*   **AI-Powered Audio Analysis:** The "Musicologist" feature automatically analyzes your track to determine its:
    *   **Mood:** (e.g., Happy/Excited, Calm/Content, Sad/Depressed)
    *   **Tempo:** (e.g., 120 BPM (fast))
    *   **Brightness:** (e.g., bright, warm, dark)
    *   **Density:** (e.g., dense, moderate, sparse)
*   **AI Cover Art Generation:**
    *   Leverages Google's Imagen model via Vertex AI to generate high-quality, 1:1 aspect ratio cover art.
    *   Use the AI-generated audio analysis to automatically create a creative prompt.
    *   Write your own manual prompt for full creative control.
*   **Flexible Export Options:**
    *   Export the final master as a high-quality, archival `.wav` file.
    *   Optionally, create a high-quality, compressed `.mp3` file for easy sharing and listening.
*   **User-Friendly Interface:** A clean and intuitive GUI built with Tkinter.

*(Note: A screenshot of the application in action would go here.)*

## How It Works

The application is composed of three main components:

1.  **`mastering_gui.py` (The Interface):** This is the main entry point for the application. It provides the graphical user interface (GUI) for users to load audio, control mastering settings, and view the generated artwork. It's built with Python's native `tkinter` library and the `ttkthemes` extension for a modern look.

2.  **`audio_mastering_engine.py` (The Engineer):** This is the backend processing engine. It receives the audio file and settings from the GUI and uses a powerful `ffmpeg` pipeline to perform the heavy lifting of audio manipulation (chunking, filtering, concatenating, and normalizing). It also orchestrates the AI analysis and art generation steps.

3.  **`ai_tagger.py` (The Musicologist):** This module is responsible for the intelligent audio analysis. It uses a pre-trained `TensorFlow/Keras` neural network to predict the mood of the audio from its spectrogram. It also uses the `librosa` library to extract technical features like tempo, spectral centroid (brightness), and RMS energy (density).

## Technology Stack

*   **GUI:** `Python`, `tkinter`, `ttkthemes`, `Pillow`
*   **Audio Processing:** `ffmpeg`, `pydub`, `numpy`, `scipy`
*   **AI Audio Analysis:** `TensorFlow (Keras)`, `librosa`, `scikit-learn`, `joblib`
*   **AI Art Generation:** `Google Cloud Vertex AI (Imagen)`
*   **Utilities:** `psutil`

## Setup & Installation

Follow these steps to get the Audio Mastering Suite running on your local machine.

### 1. Prerequisites

*   **Python 3:** Make sure you have Python 3 installed. You can download it from [python.org](https://python.org/).
*   **ffmpeg:** This is required for all audio processing. You must install it and ensure it's available in your system's PATH.
    *   **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` directory to your PATH.
    *   **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Linux (using apt):** `sudo apt-get install ffmpeg`

### 2. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 3. Install Python Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up AI Art Generation (Optional)

The AI cover art generation uses Google Cloud Vertex AI. If you want to use this feature, you need to authenticate with Google Cloud.

1.  **Create a Google Cloud Project:** If you don't have one already, create a project in the [Google Cloud Console](https://console.cloud.google.com/).
2.  **Enable the Vertex AI API:** In your project, go to the "APIs & Services" dashboard and enable the "Vertex AI API".
3.  **Authenticate your Environment:** The application uses Application Default Credentials (ADC). The easiest way to set this up for local development is to use the Google Cloud CLI.
    *   Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).
    *   Run the following command to log in and set up your credentials:
        ```bash
        gcloud auth application-default login
        ```

    If you have configured your project correctly, the application will automatically detect your credentials.

## How to Use

1.  **Launch the Application:**
    ```bash
    python mastering_gui.py
    ```
2.  **Select an Input File:** Click "Browse..." next to "Input File" to select a `.wav`, `.mp3`, or `.flac` audio file. The application will automatically suggest an output file name.
3.  **Adjust Mastering Parameters:**
    *   Use the sliders to dial in your desired sound.
    *   Optionally, select a preset from the dropdown menu to get a starting point.
    *   Check "Use Multiband Compressor" to enable and control the multiband dynamics processing.
4.  **Configure AI Cover Art:**
    *   To have the AI generate a prompt based on the music, check the "Auto-generate prompt from audio analysis?" box.
    *   To write your own prompt, leave the box unchecked and type your creative vision into the "Manual Art Prompt" field.
5.  **Choose Final Output:**
    *   By default, the application creates both a `.wav` master and a high-quality `.mp3`. Uncheck the box if you only want the `.wav` file.
6.  **Start Processing:** Click the "Start Processing" button.
    *   The status bar will update you on the progress.
    *   The "Studio Notes" section will display the results of the AI audio analysis if auto-generation is enabled.
    *   When complete, the generated AI cover art will appear in the bottom panel. Your mastered audio file(s) will be in the specified output location.
