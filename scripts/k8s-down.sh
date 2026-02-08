#!/bin/bash
# Kubernetes Stop Script
# Description: Stops the project by uninstalling the Helm release.

DELETE_CLUSTER=false

for arg in "$@"; do
  case $arg in
    --delete-cluster)
      DELETE_CLUSTER=true
      shift
      ;;
  esac
done

echo -e "\033[0;36mStopping Kubernetes stack...\033[0m"


# 1. Uninstall Helm
helm uninstall fraud-stack --ignore-not-found

# 2. Kill Port Forwarding
echo "Cleaning up port forwarding..."
pkill -f "kubectl port-forward" || true

if [ "$DELETE_CLUSTER" = true ]; then
    echo -e "\033[0;33mDeleting Kind cluster...\033[0m"
    kind delete cluster --name fraud-detection
fi

echo -e "\033[0;32mKubernetes services stopped.\033[0m"

