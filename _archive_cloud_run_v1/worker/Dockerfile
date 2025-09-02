#worker Dockerfile
# Use the official, modern image for Python 3.11, built on the stable "Bullseye" OS.
# This is the definitive, correct foundation for our worker application.
FROM python:3.11-bullseye

# Install system dependencies required by pydub (the audio library).
# This is a key difference from the backend's Dockerfile.
RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory IN THE CONTAINER.
WORKDIR /app

# Copy all the files from the build context (the worker/ folder) into the container.
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Use Gunicorn as the production-grade web server to run our worker app.
# Note that it points to "main:app" to run the main.py script.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "3600", "main:app"]

