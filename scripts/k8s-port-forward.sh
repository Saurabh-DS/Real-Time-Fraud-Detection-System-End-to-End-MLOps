#!/bin/bash

# Port Forwarding Script for Real-Time Fraud Detection System

echo "Starting port forwarding for Fraud Detection System services..."
echo "Keep this window open to maintain connections."
echo "Press Ctrl+C to stop all forwarding."
echo "------------------------------------------------------------"

RELEASE_NAME="fraud-stack"
CHART_NAME="fraud-detection-system"

# Function to start port forwarding in background
start_port_forward() {
    SERVICE_SHORT_NAME=$1
    LOCAL_PORT=$2
    REMOTE_PORT=$3
    
    FULL_SERVICE_NAME="svc/$RELEASE_NAME-$CHART_NAME-$SERVICE_SHORT_NAME"
    echo "Forwarding $SERVICE_SHORT_NAME : localhost:$LOCAL_PORT -> $REMOTE_PORT"
    
    kubectl port-forward "$FULL_SERVICE_NAME" "$LOCAL_PORT:$REMOTE_PORT" > /dev/null 2>&1 &
    PIDS+=($!)
}

# Array to store background PIDs
PIDS=()

# Trap Ctrl+C to kill all background processes
trap 'echo "Stopping all port forwards..."; for pid in "${PIDS[@]}"; do kill $pid 2>/dev/null; done; exit' INT

# Airflow Webserver
start_port_forward "airflow-webserver" 8080 8080

# Grafana
start_port_forward "grafana" 3000 3000

# MLflow
start_port_forward "mlflow" 5000 5000

# MinIO Console & API
start_port_forward "minio" 9001 9001
start_port_forward "minio" 9000 9000

# Prometheus
start_port_forward "prometheus" 9090 9090

# Fraud Serving API
start_port_forward "serving" 8000 8000

# Fraud Serving Shadow
start_port_forward "serving-shadow" 8001 8000

echo "------------------------------------------------------------"
echo "Service URLs:"
echo "Airflow:        http://localhost:8080"
echo "Grafana:        http://localhost:3000"
echo "MLflow:         http://localhost:5000"
echo "MinIO Console:  http://localhost:9001"
echo "Prometheus:     http://localhost:9090"
echo "Serving:        http://localhost:8000/docs"
echo "Serving Shadow: http://localhost:8001/docs"
echo "------------------------------------------------------------"
echo "Listening..."

# Wait forever (requires Ctrl+C to exit)
wait
