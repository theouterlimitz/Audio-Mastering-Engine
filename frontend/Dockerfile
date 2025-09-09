# Dockerfile for the Frontend Service (The "Waiter")
# Reverted to use a simpler build context.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR $APP_HOME

# The build context is now JUST the frontend folder, so we copy directly.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY main.py .
COPY templates/ ./templates
COPY static/ ./static

# The CMD remains the same
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 120 main:app


