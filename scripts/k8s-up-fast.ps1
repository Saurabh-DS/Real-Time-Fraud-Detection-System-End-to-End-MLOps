# Kubernetes Start (Fast Mode)
# Description: Starts the project in the existing Kind cluster without rebuilding images or recreating the cluster.

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host -ForegroundColor Cyan "`n-> $Message"
}

try {
    # 1. Ensure cluster is up
    Write-Step "Checking Kind cluster..."
    $clusters = kind get clusters
    if ($clusters -notcontains "fraud-detection") {
        Write-Host "Cluster missing. Running fresh setup instead..."
        powershell -ExecutionPolicy Bypass -File .\scripts\k8s-up-fresh.ps1
        exit
    }

    # 2. Helm Upgrade (Ensures latest config/values are applied)
    Write-Step "Upgrading Helm Release (Fast)..."
    helm upgrade --install fraud-stack ./helm/fraud-detection-system --wait --timeout 300s

    # 3. Port Forwarding
    Write-Step "Starting Port Forwarding..."
    powershell -ExecutionPolicy Bypass -File ./scripts/k8s-port-forward.ps1

    Write-Host "`nKubernetes stack is ready (FAST MODE)"

    kubectl get pods

} catch {
    Write-Host -ForegroundColor Red "`n[ERROR] Fast start failed: $($_.Exception.Message)"
    exit 1
}
