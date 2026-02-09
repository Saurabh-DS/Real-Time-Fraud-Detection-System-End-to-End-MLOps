#!/bin/bash
# Kubernetes Start (Fresh/Clean Mode)
# Description: Wipes the entire environment and starts a fresh Kubernetes deployment.

set -e

echo -e "\033[0;36m\n-> Starting Fraud Detection System (Kubernetes Fresh Mode)...\033[0m"

# 1. Full Cleanup
echo -e "\033[0;36m-> CLEANING: Wiping Kind cluster and pruning Docker...\033[0m"
kind delete cluster --name fraud-detection || true
docker system prune -f

# 2. Cluster Creation
echo -e "\033[0;36m-> Creating fresh Kind cluster (3-Nodes)...\033[0m"
kind create cluster --config kind-config.yaml --image kindest/node:v1.31.1 --wait 60s

# 3. Build & Load Images
echo -e "\033[0;36m-> Building and loading images...\033[0m"
docker-compose build

echo -e "\033[0;36m-> Tagging images for Kubernetes...\033[0m"
docker tag real-time-fraud-detection-system-end-to-end-mlops-producer:latest fraud-producer:v3
docker tag real-time-fraud-detection-system-end-to-end-mlops-feature-processor:latest feature-processor:latest
docker tag real-time-fraud-detection-system-end-to-end-mlops-serving:latest fraud-serving:latest
docker tag real-time-fraud-detection-system-end-to-end-mlops-airflow-webserver:latest airflow-custom:v9

IMAGES=("fraud-producer:v3" "feature-processor:latest" "fraud-serving:latest" "airflow-custom:v9")
for img in "${IMAGES[@]}"; do
    kind load docker-image "$img" --name fraud-detection
done

# 4. Fresh Helm Install
echo -e "\033[0;36m-> Installing Helm Stack...\033[0m"
helm upgrade --install fraud-stack ./helm/fraud-detection-system --wait --timeout 600s

# 5. Port Forwarding
echo -e "\033[0;36m-> Starting Port Forwarding...\033[0m"
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    ./scripts/k8s-port-forward.sh &
else
    echo "Please run ./scripts/k8s-port-forward.sh manually if not on Linux/macOS."
fi

echo -e "\nKubernetes stack is ready (FRESH START)"

kubectl get pods
