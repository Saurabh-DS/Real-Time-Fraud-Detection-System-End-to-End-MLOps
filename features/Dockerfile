# Feature Processor Dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY *.py ./
COPY *.yaml ./

# Create data directory
RUN mkdir -p data

# Run feature processor
CMD ["python", "feature_processor.py"]
