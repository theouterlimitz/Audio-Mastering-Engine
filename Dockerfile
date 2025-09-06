# Use the official Python image as a base image.
# We are using python3.12-slim to keep the image size small.
FROM python:3.12-slim

# Set the working directory in the container to /app.
WORKDIR /app

# Copy the requirements file into the container at /app.
COPY requirements.txt .

# Install any dependencies specified in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local code from the current directory to the /app directory in the container.
COPY . .

# Set the entrypoint to a command that runs your application.
# This assumes your application is a Gunicorn server and your main app is in main.py.
# You may need to change 'main:app' to match your application entry point.
# For example, if your file is app.py and your application object is named 'my_app',
# you would use `gunicorn --bind :$PORT --workers 1 --threads 8 app:my_app`.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

# The GAE Flex environment automatically exposes the PORT environment variable.
# We will use that to configure our server.
EXPOSE $PORT
