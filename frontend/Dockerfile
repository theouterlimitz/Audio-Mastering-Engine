# Dockerfile for the Frontend Service (The "Waiter")

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR 

# Now we copy from the correct subdirectories relative to the project root
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./frontend
COPY shared/ ./shared

# The CMD remains the same
CMD exec gunicorn --bind : --workers 2 --threads 4 --timeout 120 frontend.main:app
