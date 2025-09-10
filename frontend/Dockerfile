# Dockerfile for the Frontend Service (The "Waiter")
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY templates/ ./templates
COPY static/ ./static
# --- FIX: Copy the correctly named shared directory ---
COPY app_shared/ ./app_shared
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "main:app"]
