# Dockerfile
# Use the official Python 3.11 image as a base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# The command to run your app using Gunicorn (from your app.yaml)
# App Engine will automatically substitute the $PORT environment variable.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app