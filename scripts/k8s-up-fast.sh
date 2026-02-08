#!/bin/bash
# Kubernetes Start (Fast Mode)
# Description: Starts the project in the existing Kind cluster without rebuilding images or recreating the cluster.

set -e

echo -e "\033[0;36m\n-> Starting Fraud Detection System (Kubernetes Fast Mode)...\033[0m"

# 1. Ensure cluster is up
echo -e "\033[0;36m-> Checking Kind cluster...\033[0m"
CLUSTERS=$(kind get clusters)
if [[ ! $CLUSTERS =~ "fraud-detection" ]]; then
    echo "Cluster missing. Running fresh setup instead..."
    ./scripts/k8s-up-fresh.sh
    exit 0
fi

# 2. Helm Upgrade (Ensures latest config/values are applied)
echo -e "\033[0;36m-> Upgrading Helm Release (Fast)...\033[0m"
helm upgrade --install fraud-stack ./helm/fraud-detection-system --wait --timeout 300s

# 3. Port Forwarding
echo -e "\033[0;36m-> Starting Port Forwarding...\033[0m"
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    ./scripts/k8s-port-forward.sh &
else
    # Fallback/Manual check required for other envs
    echo "Please run ./scripts/k8s-port-forward.sh manually if not on Linux/macOS."
fi

echo -e "\nKubernetes stack is ready (FAST MODE)"

kubectl get pods
