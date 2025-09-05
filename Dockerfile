# Dockerfile (Final Architecture)
# This version explicitly installs the python3-pil package using apt-get.
# This is a more robust installation method than pip for this specific library
# and guarantees that the PIL module is available to the Python runtime.

# Use the official Python 3.11 image as a base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install FFmpeg AND the Python Imaging Library system package
RUN apt-get update && apt-get install -y ffmpeg python3-pil

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# The command to run your app using Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app




   

    

