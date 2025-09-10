# Dockerfile for the Frontend Service (The "Waiter")
# Final version with a more robust CMD instruction.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./frontend
COPY frontend/shared/ ./shared

# --- THIS IS THE FIX ---
# Removed 'exec' for more reliable shell variable expansion.
CMD gunicorn --bind : --workers 2 --threads 4 --timeout 120 frontend.main:app
