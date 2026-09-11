# Use a lightweight Python environment
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DICTS_DIR=model_dicts

# Set working directory
WORKDIR /app

# Install system dependencies
# We use slim, but we may need some build tools for certain python packages.
# However, for the current requirements, slim should be sufficient.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser /app
USER appuser

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the application
# --host 0.0.0.0 allows it to be accessible from the outside world
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
