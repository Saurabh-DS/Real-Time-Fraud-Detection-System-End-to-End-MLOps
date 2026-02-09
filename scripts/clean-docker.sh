#!/bin/bash
# Project-Specific Docker Cleanup Script - Bash
# =============================================
# Stops and removes containers, networks, volumes, and images 
# associated with the docker-compose.yaml in this project ONLY.

set -e

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yaml"

echo -e "\033[0;36mCleaning Docker resources for project in: $PROJECT_ROOT\033[0m"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Error: docker-compose.yaml not found at $COMPOSE_FILE"
    exit 1
fi

echo "Running docker compose down..."
# --volumes: Remove named volumes declared in the `volumes` section of the Compose file and anonymous volumes attached to containers.
# --rmi all: Remove all images used by any service.
# --remove-orphans: Remove containers for services not defined in the Compose file.
docker compose -f "$COMPOSE_FILE" down --volumes --rmi all --remove-orphans

echo -e "\033[0;32mProject Docker resources cleaned successfully!\033[0m"
