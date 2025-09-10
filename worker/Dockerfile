# Dockerfile for the Worker Service (The "Kitchen")
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
# --- FIX: Copy the correctly named shared directory ---
COPY app_shared/ ./app_shared
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app"]
