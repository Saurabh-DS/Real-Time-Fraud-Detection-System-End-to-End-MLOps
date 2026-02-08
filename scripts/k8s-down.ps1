# Kubernetes Stop Script
# Description: Stops the project by uninstalling the Helm release.

param (
    [switch]$DeleteCluster = $false
)

Write-Host -ForegroundColor Cyan "Stopping Kubernetes stack..."

# 1. Uninstall Helm
helm uninstall fraud-stack --ignore-not-found

# 2. Kill Port Forwarding (Optional/Cleanup)
Get-Process | Where-Object { $_.ProcessName -match "kubectl" } | Stop-Process -Force -ErrorAction SilentlyContinue

if ($DeleteCluster) {
    Write-Host -ForegroundColor Yellow "Deleting Kind cluster..."
    kind delete cluster --name fraud-detection
}

Write-Host -ForegroundColor Green "Kubernetes services stopped."

