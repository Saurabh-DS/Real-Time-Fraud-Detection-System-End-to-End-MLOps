#!/bin/bash
# Docker Slate Wipe Script - Bash
# ===============================
# Stops all containers and removes all images, volumes, networks, and cache.

echo -e "\033[0;36mWiping the Docker slate clean...\033[0m"

# Stop all running containers
if [ "$(docker ps -q)" ]; then
    echo "Stopping running containers..."
    docker stop $(docker ps -q)
fi

# Remove all containers, images, volumes, networks, and build cache
echo "Clearing all Docker resources (Containers, Images, Volumes, Networks, Cache)..."
docker system prune -a --volumes -f

echo -e "\033[0;32mDocker slate wiped clean!\033[0m"
