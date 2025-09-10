# Dockerfile for the Frontend Service (The "Waiter")
# Final version with corrected relative paths for the build context.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR /app

# The build context is the frontend/ folder, so we copy directly.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code from the current context.
COPY main.py .
COPY templates/ ./templates
COPY static/ ./static

# The CMD is unchanged.
CMD exec gunicorn --bind : --workers 2 --threads 4 --timeout 120 main:app
