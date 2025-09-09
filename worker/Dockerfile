# Dockerfile for the Worker Service (The "Kitchen")
# This is the final, clean version of the blueprint for our backend.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR 

# Install ffmpeg system dependency
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg &&     rm -rf /var/lib/apt/lists/*

# Copy the worker's specific requirements and install them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker's own main file.
COPY main.py .

# CRITICAL: Copy the shared "golden master" engine from the parent directory
# into a 'shared' folder inside the container.
COPY ../shared/ ./shared

# The CMD tells the server how to run the app.
CMD exec gunicorn --bind : --workers 1 --threads 8 --timeout 0 main:app
