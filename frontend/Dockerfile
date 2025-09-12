# Dockerfile for the Frontend Service (The "Waiter")
# This is the final, simplified version with the correct port binding.

FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy and install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary application code from the current context
COPY main.py .
COPY templates/ ./templates
COPY static/ ./static

# --- THIS IS THE FINAL FIX ---
# This "shell form" of the CMD instruction correctly interprets the
# $PORT environment variable provided by Cloud Run.
CMD gunicorn --bind 0.0.0.0:$PORT main:app