# Stop Fraud Detection System (Docker Only)
# Description: Stops all project services and optionally cleans volumes.

param (
    [switch]$Clean = $false
)

Write-Host -ForegroundColor Cyan "Stopping Fraud Detection System..."

if ($Clean) {
    Write-Host -ForegroundColor Yellow "Cleaning volumes..."
    docker-compose down -v --remove-orphans
} else {
    docker-compose down --remove-orphans
}

Write-Host -ForegroundColor Green "All services stopped."

