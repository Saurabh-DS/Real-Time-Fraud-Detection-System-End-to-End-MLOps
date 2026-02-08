# Training Pipeline Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY *.py ./

# Create artifacts directory
RUN mkdir -p artifacts

# Default command: generate data then train
CMD ["sh", "-c", "python generate_training_data.py && python train.py"]
