# Dockerfile for the Worker Service (The "Kitchen")
# This is the final, corrected version for our simple deployment architecture.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Install ffmpeg system dependency
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    # Clean up the package manager cache to keep the final image lean.
    rm -rf /var/lib/apt/lists/*

# The build context is the worker/ folder, so we copy the local requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker's own main file.
COPY main.py .

# CRITICAL: Copy the shared "golden master" engine from the parent directory
# (../) into a 'shared' folder inside the container.
COPY ../shared/ ./shared

# The CMD tells the server how to run the app.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
```

### What Happens Now

This was the final missing piece of the puzzle. With this corrected file, your project structure is now perfect and internally consistent.

The final step before launch is to save this change to your Git repository.

```bash
# Add the corrected Dockerfile
git add worker/Dockerfile

# Commit the fix
git commit -m "fix(deploy): Correct worker Dockerfile to copy shared engine"

# Push the fix to GitHub
git push origin feature/cloud-migration
```

After you have pushed this final change, you will be ready. Run the launch command from your project's **root directory**:

```bash
gcloud app deploy app.yaml worker.yaml


