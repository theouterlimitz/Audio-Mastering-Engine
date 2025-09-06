# Use the official Python image.
FROM python:3.12-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code into the container.
COPY . .

# Command to run the application using Gunicorn.
# App Engine will automatically set the $PORT environment variable.
CMD ["gunicorn", "--timeout", "600", "-b", ":$PORT", "main:app"]
