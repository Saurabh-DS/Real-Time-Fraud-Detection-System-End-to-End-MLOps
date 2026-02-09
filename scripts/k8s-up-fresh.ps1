# Kubernetes Start (Fresh/Clean Mode)
# Description: Wipes the entire environment and starts a fresh Kubernetes deployment.

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host -ForegroundColor Cyan "`n-> $Message"
}

try {
    # 1. Full Cleanup
    Write-Step "CLEANING: Wiping Kind cluster and pruning Docker..."
    kind delete cluster --name fraud-detection
    docker system prune -f
    
    # 2. Cluster Creation
    Write-Step "Creating fresh Kind cluster (3-Nodes)..."
    kind create cluster --config kind-config.yaml --image kindest/node:v1.31.1 --wait 60s

    # 3. Build & Load Images
    Write-Step "Building and loading images..."
    docker-compose build
    
    Write-Step "Tagging images for Kubernetes..."
    docker tag real-time-fraud-detection-system-end-to-end-mlops-producer:latest fraud-producer:v3
    docker tag real-time-fraud-detection-system-end-to-end-mlops-feature-processor:latest feature-processor:latest
    docker tag real-time-fraud-detection-system-end-to-end-mlops-serving:latest fraud-serving:latest
    docker tag real-time-fraud-detection-system-end-to-end-mlops-airflow-webserver:latest airflow-custom:v9

    $images = @("fraud-producer:v3", "feature-processor:latest", "fraud-serving:latest", "airflow-custom:v9")
    foreach ($img in $images) {
        kind load docker-image $img --name fraud-detection
    }

    # 4. Fresh Helm Install
    Write-Step "Installing Helm Stack..."
    helm upgrade --install fraud-stack ./helm/fraud-detection-system --wait --timeout 600s

    # 5. Port Forwarding
    Write-Step "Starting Port Forwarding..."
    powershell -ExecutionPolicy Bypass -File ./scripts/k8s-port-forward.ps1

    Write-Host "`nKubernetes stack is ready (FRESH START)"

    kubectl get pods

} catch {
    Write-Host -ForegroundColor Red "`n[ERROR] Fresh start failed: $($_.Exception.Message)"
    exit 1
}
