# Dockerfile for the Worker Service (The "Kitchen")

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR 

# Install ffmpeg system dependency
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Now we copy from the correct subdirectories relative to the project root
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY worker/ ./worker
COPY shared/ ./shared

# The CMD remains the same
CMD exec gunicorn --bind : --workers 1 --threads 8 --timeout 0 worker.main:app
