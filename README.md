# Python Audio Mastering Suite v2.0

A professional-grade, local desktop application for mastering audio files. Built with Python and a modern Tkinter GUI, this tool provides a complete, feature-rich mastering chain designed for musicians, producers, and DJs who want to add a final layer of polish to their tracks.

This application is the stable, local version of the project. A future release is planned to transition this engine to a scalable, cloud-native web application on Google Cloud Platform.

![Screenshot of the Audio Mastering Suite GUI](gui.png)

---

## Features

This suite goes beyond basic audio processing, incorporating advanced features for professional-grade results:

-   **Intuitive Graphical Interface:** A clean, dark-themed UI built with Tkinter that provides real-time control over all parameters without touching the command line.
-   **Upgraded Mastering Engine:**
    -   **Analog Character Control:** A single, powerful slider that intelligently blends harmonic saturation, a gentle sub-bass "bump," and a high-end "sparkle" to add vintage analog warmth and mojo.
    -   **Phase-Coherent Multiband Compressor:** A re-architected 3-band compressor that avoids phase cancellation issues, ensuring a tight, punchy, and clean sound. Features detailed controls for threshold and ratio on each band.
    -   **Professional EQ & Widening:** A full 4-band equalizer and a stereo widener to shape the tonal balance and spatial image of your mix.
-   **Streaming-Ready Loudness:** Utilizes **LUFS (Loudness Units Full Scale)** normalization to target the consistent, balanced loudness levels required by modern streaming platforms like Spotify and Apple Music.
-   **Robust & Efficient:** Handles audio files of any size by processing them in manageable chunks, with a progress bar for real-time feedback. The processing is run in a background thread to keep the GUI responsive.
-   **CD-Quality Export:** Final masters are exported to a professional, standard **16-bit / 44.1kHz WAV format** to ensure high quality without unnecessarily large file sizes.

---

## Installation & Setup

This application requires a specific Python environment to ensure all audio libraries and UI components work correctly. Please follow these steps.

### 1. Create a Conda Environment

This project is built and tested with **Python 3.11**. The best way to manage this and its dependencies is with a dedicated Conda environment.

```bash
# Create a new environment named "audio_env" with Python 3.11
conda create -n audio_env python=3.11

2. Activate the Environment
You must activate this environment every time you want to run the application.

conda activate audio_env

(Your terminal prompt will change to show (audio_env))

3. Install Required Libraries
Once the environment is active, install all necessary Python packages with this single command.

pip install pydub scipy numpy pyloudnorm ttkthemes

4. Install FFmpeg
The audio engine (pydub) requires the FFmpeg system utility for handling different audio formats like MP3 and WAV.

On Debian/Ubuntu:

sudo apt update && sudo apt install ffmpeg

On macOS (using Homebrew):

brew install ffmpeg

On Windows:
Download the FFmpeg binaries from the official site and add the bin folder to your system's PATH.

How to Run
After completing the setup, you can launch the application from the project's root directory.

Open your terminal.

Navigate to the Audio-Mastering-Engine project folder.

Activate the Conda environment: conda activate audio_env

Run the GUI script:

python mastering_gui.py


