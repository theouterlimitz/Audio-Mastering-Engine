# Dockerfile for the Frontend Service (The "Waiter")

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend's own code and its own copy of the shared engine
COPY frontend/ ./frontend
COPY frontend/shared/ ./shared

CMD exec gunicorn --bind : --workers 2 --threads 4 --timeout 120 frontend.main:app
