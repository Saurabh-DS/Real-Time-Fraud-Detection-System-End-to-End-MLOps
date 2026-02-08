# Airflow Custom Image for Fraud Detection MLOps
#
# Includes:
# - Python ML dependencies (pandas, numpy, mlflow, xgboost)
# - Custom DAGs and utilities

FROM apache/airflow:2.10.2-python3.11

USER root
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python dependencies
COPY airflow/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy DAGs and Training Scripts
COPY airflow/dags /opt/airflow/dags
COPY training /opt/airflow/training
COPY ops /opt/airflow/ops
COPY monitoring /opt/airflow/monitoring
