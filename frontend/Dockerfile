# Dockerfile for the Frontend Service (The "Waiter")
# Final version with the correct, hardcoded port for App Engine Flex.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY templates/ ./templates
COPY static/ ./static
COPY shared/ ./shared

# --- THIS IS THE FIX ---
# We use the exec form and hardcode the port to 8080, which is what
# the App Engine Flexible environment requires.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "main:app"]
