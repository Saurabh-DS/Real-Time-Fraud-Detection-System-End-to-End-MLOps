# Start Fraud Detection System (Docker Only)
# Description: Starts the full MLOps stack using Docker Compose without Kubernetes overhead.

param (
    [switch]$Build = $false,
    [switch]$Clean = $false
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host -ForegroundColor Cyan "`n-> $Message"
}

try {
    # 1. Cleanup if requested
    if ($Clean) {
        Write-Step "Cleaning existing Docker resources..."
        docker-compose down -v --remove-orphans
    }

    # 2. Build if requested
    if ($Build) {
        Write-Step "Building Docker images..."
        docker-compose build
    }

    # 3. Airflow Initialization
    Write-Step "Initializing Airflow database..."
    docker-compose up -d airflow-postgres
    Start-Sleep -Seconds 10
    docker-compose up airflow-init

    # 4. Start all services
    Write-Step "Starting all services in detached mode..."
    docker-compose up -d

    Write-Host "`n================================================"
    Write-Host "  PROJECT READY (DOCKER MODE)"

    Write-Host "  Airflow:    http://localhost:8080"
    Write-Host "  MLflow:     http://localhost:5000"
    Write-Host "  Grafana:    http://localhost:3000"
    Write-Host "  API:        http://localhost:8000"
    Write-Host "================================================"

} catch {
    Write-Host -ForegroundColor Red "`n[ERROR] Failed to start Docker stack: $($_.Exception.Message)"
    exit 1
}
