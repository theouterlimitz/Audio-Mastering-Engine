# Dockerfile for the Frontend Service (The "Waiter")
# This defines the simple, lightweight environment for our user-facing web app.

# 1. Start with the same lean Python base image for consistency.
FROM python:3.11-slim

# 2. Set environment variables.
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR $APP_HOME

# 3. Install Python dependencies.
# The frontend has the same Python requirements as the worker.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the frontend's application code into the container.
# Notice we are ONLY copying the frontend and shared code.
COPY frontend/ ./frontend
COPY shared/ ./shared

# 5. Define the command to run the application.
# We use gunicorn, a production-grade server. The --workers and --threads
# are tuned for a lightweight, I/O-bound web server.
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 120 frontend.main:app

