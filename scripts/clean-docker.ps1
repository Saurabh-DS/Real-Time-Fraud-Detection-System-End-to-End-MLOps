# Docker Slate Wipe Script - PowerShell
# ====================================
# Stops all containers and removes all images, volumes, networks, and cache.

Write-Host "Wiping the Docker slate clean..." -ForegroundColor Cyan

# Stop all running containers
$runningContainers = docker ps -q
if ($runningContainers) {
    Write-Host "Stopping running containers..."
    docker stop $runningContainers
}

# Remove all containers, images, volumes, networks, and build cache
Write-Host "Clearing all Docker resources (Containers, Images, Volumes, Networks, Cache)..."
docker system prune -a --volumes -f

Write-Host "Docker slate wiped clean!" -ForegroundColor Green
