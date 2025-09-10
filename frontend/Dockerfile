# Dockerfile for the Frontend Service (The "Waiter")

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary application code from the current context
COPY main.py .
COPY templates/ ./templates
COPY static/ ./static
COPY shared/ ./shared

CMD gunicorn --bind : --workers 2 --threads 4 --timeout 120 main:app
