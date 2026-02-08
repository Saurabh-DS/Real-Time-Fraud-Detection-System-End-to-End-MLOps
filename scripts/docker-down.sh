#!/bin/bash
# Stop Fraud Detection System (Docker Only)
# Description: Stops all project services and optionally cleans volumes.

CLEAN=false

for arg in "$@"; do
  case $arg in
    --clean)
      CLEAN=true
      shift
      ;;
  esac
done

echo -e "\033[0;36mStopping Fraud Detection System...\033[0m"

if [ "$CLEAN" = true ]; then
    echo -e "\033[0;33mCleaning volumes...\033[0m"
    docker-compose down -v --remove-orphans
else
    docker-compose down --remove-orphans
fi

echo -e "\033[0;32mAll services stopped.\033[0m"

