# Kubernetes Auto-scaling Dashboard (PowerShell)
# Monitors HPA scale-up and scale-down driven by internal Producer load.

function Write-Step {
    param([string]$Message)
    Write-Host "`n-> $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

$maxReplicas = 15

Clear-Host
Write-Step "Starting Dashboard (Producer Load: 1000 TPS)..."
Write-Host "Press Ctrl+C to abort."

while ($true) {
    # 1. HPA Status
    $replicas = (kubectl get hpa fraud-stack-fraud-detection-system-serving-hpa -n default -o jsonpath='{.status.currentReplicas}')
    $cpu = (kubectl get hpa fraud-stack-fraud-detection-system-serving-hpa -n default -o jsonpath='{.status.currentMetrics[0].resource.current.averageUtilization}')
    
    if (-not $replicas) { $replicas = 0 }
    if (-not $cpu) { $cpu = 0 }

    Clear-Host
    Write-Host "---------------------------------------------------"
    Write-Host "HPA Status: $replicas / $maxReplicas Replicas | CPU Load: $cpu%"
    Write-Host "---------------------------------------------------"

    # 2. Pod Metrics (RAM)
    Write-Host "`nPod Resources:"
    kubectl top pods -n default -l app=fraud-stack-fraud-detection-system-serving --no-headers | Sort-Object
    
    # 3. Pod Status
    Write-Host "`nPod Lifecycle Status:"
    kubectl get pods -n default -l app=fraud-stack-fraud-detection-system-serving -o wide --show-labels | Select-String -Pattern "NAME" -Context 0,100

    # 4. Cluster Topology (Node Distribution)
    Write-Host "`nCluster Topology (Where are my containers?):"
    kubectl get pods -n default -l app=fraud-stack-fraud-detection-system-serving -o custom-columns=NODE:.spec.nodeName,NAME:.metadata.name,STATUS:.status.phase,CONTAINER_ID:.status.containerStatuses[0].containerID --sort-by=.spec.nodeName

    Start-Sleep -Seconds 3
}
