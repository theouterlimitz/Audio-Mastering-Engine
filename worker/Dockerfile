# Dockerfile for the Worker Service (The "Kitchen")
# Final version with a more robust CMD instruction.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg &&     rm -rf /var/lib/apt/lists/*

COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY worker/ ./worker
COPY worker/shared/ ./shared

# --- THIS IS THE FIX ---
# Removed 'exec' for more reliable shell variable expansion.
CMD gunicorn --bind : --workers 1 --threads 8 --timeout 0 worker.main:app
