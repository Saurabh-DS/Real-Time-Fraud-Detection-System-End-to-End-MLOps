Write-Host "Starting Port Forwarding for Real-Time Fraud Detection System..." -ForegroundColor Cyan


# Function to start a background job for port forwarding
function Start-PortForward {
    param (
        [string]$Service,
        [string]$Ports,
        [string]$Name
    )
    Write-Host "  + Forwarding $Name ($Ports)..." -NoNewline
    $job = Start-Job -ScriptBlock { 
        param($s, $p) 
        try {
            # Retry logic or wait for pod availability could go here
            kubectl port-forward "svc/$s" $p -n default 
        } catch {
            Write-Error $_
        }
    } -ArgumentList $Service, $Ports
    
    if ($job.State -eq 'Running') {
        Write-Host " [OK]" -ForegroundColor Green
    } else {
        Write-Host " [FAILED]" -ForegroundColor Red
        Receive-Job -Job $job
    }
    return $job
}

# Stop any existing kubectl port-forward processes (optional, but clean)
# Stop-Process -Name "kubectl" -ErrorAction SilentlyContinue

$jobs = @()

# Use Helm release name prefix
$prefix = "fraud-stack-fraud-detection-system"

# 1. Fraud API
$jobs += Start-PortForward -Service "$prefix-serving" -Ports "8000:8000" -Name "Fraud API"

# 2. Grafana
$jobs += Start-PortForward -Service "$prefix-grafana" -Ports "3000:3000" -Name "Grafana"

# 3. MLflow
$jobs += Start-PortForward -Service "$prefix-mlflow" -Ports "5000:5000" -Name "MLflow"

# 4. Airflow
$jobs += Start-PortForward -Service "$prefix-airflow-webserver" -Ports "8080:8080" -Name "Airflow"

# 5. Prometheus
$jobs += Start-PortForward -Service "$prefix-prometheus" -Ports "9090:9090" -Name "Prometheus"

# 6. MinIO
$jobs += Start-PortForward -Service "$prefix-minio" -Ports "9000:9000 9001:9001" -Name "MinIO"

# 7. Serving Shadow (Port 8001 -> 8000 target)
$jobs += Start-PortForward -Service "$prefix-serving-shadow" -Ports "8001:8000" -Name "Serving Shadow"

# 8. Feature Processor (Metrics/Internal - if needed)
# $jobs += Start-PortForward -Service "$prefix-feature-processor" -Ports "8002:8000" -Name "Feature Processor"

Write-Host "`nAll tunnels established! Access services at:" -ForegroundColor Cyan

Write-Host "---------------------------------------------------"
Write-Host "  Fraud API:      http://localhost:8000/docs"

Write-Host "  Grafana:        http://localhost:3000 (admin/admin)"

Write-Host "  MLflow:         http://localhost:5000"

Write-Host "  Airflow:        http://localhost:8080 (admin/admin)"

Write-Host "  Prometheus:     http://localhost:9090"

Write-Host "  MinIO Console:  http://localhost:9001 (minioadmin/minioadmin)"

Write-Host "  Serving Shadow: http://localhost:8001/docs"

Write-Host "---------------------------------------------------"
Write-Host "Press Ctrl+C to stop all forwarding..." -ForegroundColor Yellow

try {
    # Keep script running to maintain jobs
    while ($true) {
        Start-Sleep -Seconds 1
        
        # Check if jobs are still running
        foreach ($j in $jobs) {
            if ($j.State -ne 'Running') {
                Write-Host "  Job $($j.Id) stopped unexpectedly. Output:" -ForegroundColor Red

                Receive-Job -Job $j
            }
        }
    }
}
finally {
    Write-Host "`nStopping all background jobs..." -ForegroundColor Yellow

    Stop-Job -Job $jobs
    Remove-Job -Job $jobs
    Write-Host "Done."
}
