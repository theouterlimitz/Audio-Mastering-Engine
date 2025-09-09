# Dockerfile for the Worker Service (The "Kitchen")
# This file defines the environment for our powerful, long-running backend service.

# 1. Start with a lean, official Python base image.
# Using 'slim' keeps our final application size smaller and more secure.
FROM python:3.11-slim

# 2. Set environment variables for a clean build process.
# PYTHONUNBUFFERED ensures logs are sent straight to the console.
# APP_HOME defines a standard location for our application code.
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# 3. Set the working directory inside the container.
WORKDIR $APP_HOME

# 4. Install system-level dependencies.
# This is the most critical step: we are installing ffmpeg, the powerhouse
# behind our entire audio processing pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    # Clean up the package manager cache to keep the final image lean.
    rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies.
# We copy ONLY the requirements file first to take advantage of Docker's
# layer caching. This layer will only be rebuilt if the requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy our application code into the container.
# This includes the worker's own main file and the shared "golden master" engine.
COPY worker/ ./worker
COPY shared/ ./shared

# 7. Define the command to run the application.
# We use gunicorn, a production-grade Python web server.
# --bind :$PORT is mandatory for App Engine to route traffic.
# --timeout 0 is the CRITICAL setting that allows our mastering jobs to run
# for up to 60 minutes without the server giving up.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 worker.main:app
