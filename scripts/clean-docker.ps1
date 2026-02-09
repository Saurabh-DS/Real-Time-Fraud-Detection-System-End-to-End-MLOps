# Project-Specific Docker Cleanup Script - PowerShell
# ====================================================
# Stops and removes containers, networks, volumes, and images 
# associated with the docker-compose.yaml in this project ONLY.

$ErrorActionPreference = "Stop"

# Get the project root directory (assuming script is in /scripts)
$scriptPath = $MyInvocation.MyCommand.Path
$projectRoot = Join-Path (Split-Path $scriptPath -Parent) ".."
$composeFile = Join-Path $projectRoot "docker-compose.yaml"

Write-Host "Cleaning Docker resources for project in: $projectRoot" -ForegroundColor Cyan

# Check if docker-compose.yaml exists
if (-not (Test-Path $composeFile)) {
    Write-Error "docker-compose.yaml not found at $composeFile. Is the script in the /scripts folder?"
}

# Run docker compose down
Write-Host "Running docker compose down..."
try {
    # --volumes: Remove named volumes declared in the `volumes` section of the Compose file and anonymous volumes attached to containers.
    # --rmi all: Remove all images used by any service.
    # --remove-orphans: Remove containers for services not defined in the Compose file.
    docker compose -f $composeFile down --volumes --rmi all --remove-orphans
    Write-Host "Project Docker resources cleaned successfully!" -ForegroundColor Green
}
catch {
    Write-Error "Failed to clean Docker resources: $_"
}
