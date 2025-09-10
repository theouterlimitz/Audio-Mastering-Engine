# Dockerfile for the Worker Service (The "Kitchen")
# Final version with corrected relative paths for the build context.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg &&     rm -rf /var/lib/apt/lists/*

# The build context is the worker/ folder, so we copy directly.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker's own main file.
COPY main.py .

# Copy the worker's own copy of the shared engine.
COPY shared/ ./shared

# The CMD is unchanged.
CMD exec gunicorn --bind : --workers 1 --threads 8 --timeout 0 main:app
