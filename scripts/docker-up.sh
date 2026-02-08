#!/bin/bash
# Start Fraud Detection System (Docker Only)
# Description: Starts the full MLOps stack using Docker Compose without Kubernetes overhead.

set -e

# Support for flags
BUILD=false
CLEAN=false

for arg in "$@"; do
  case $arg in
    --build)
      BUILD=true
      shift
      ;;
    --clean)
      CLEAN=true
      shift
      ;;
  esac
done

echo -e "\033[0;36m\n-> Starting Fraud Detection System (Docker Mode)...\033[0m"

# 1. Cleanup if requested
if [ "$CLEAN" = true ]; then
    echo -e "\033[0;36m-> Cleaning existing Docker resources...\033[0m"
    docker-compose down -v --remove-orphans
fi

# 2. Build if requested
if [ "$BUILD" = true ]; then
    echo -e "\033[0;36m-> Building Docker images...\033[0m"
    docker-compose build
fi

# 3. Airflow Initialization
echo -e "\033[0;36m-> Initializing Airflow database...\033[0m"
docker-compose up -d airflow-postgres
sleep 10
docker-compose up airflow-init

# 4. Start all services
echo -e "\033[0;36m-> Starting all services in detached mode...\033[0m"
docker-compose up -d

echo -e "\n================================================"
echo -e "  PROJECT READY (DOCKER MODE)"

echo -e "  Airflow:    http://localhost:8080"
echo -e "  MLflow:     http://localhost:5000"
echo -e "  Grafana:    http://localhost:3000"
echo -e "  API:        http://localhost:8000"
echo -e "================================================"
