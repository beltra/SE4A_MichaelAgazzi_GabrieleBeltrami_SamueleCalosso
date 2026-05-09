#!/bin/bash

# Default variables
COMPOSE_FILE="docker/docker-compose.yml"
SERVICE_NAME="uav-testing"
REBUILD=false

# Parse input parameters
for arg in "$@"; do
    case $arg in
        --nogpu)
            COMPOSE_FILE="docker/docker-compose-nogpu.yml"
            ;;
        --rebuild)
            REBUILD=true
            ;;
        *)
            echo "Error: Unknown argument '$arg'"
            echo "Usage: $0 [--nogpu] [--rebuild]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Compose File : $COMPOSE_FILE"
echo "Service      : $SERVICE_NAME"
echo "========================================"

# Check if the container is already running
RUNNING=$(docker compose -f "$COMPOSE_FILE" ps -q "$SERVICE_NAME")

if [ "$REBUILD" = true ]; then
    echo "Rebuilding image..."
    docker compose -f "$COMPOSE_FILE" down
    xhost +local:docker
    docker compose -f "$COMPOSE_FILE" up -d --build
elif [ -z "$RUNNING" ]; then
    echo "Container not running. Ensuring it exists..."

    # Grant X server access for Gazebo GUI
    xhost +local:docker

    # Start the container in detached mode.
    # --no-recreate ensures we don't destroy an existing stopped container.
    docker compose -f "$COMPOSE_FILE" up -d --no-recreate
else
    echo "Container is already running. Skipping start..."
fi

# Enter the interactive bash shell
echo "Attaching to interactive bash shell..."
docker compose -f "$COMPOSE_FILE" exec "$SERVICE_NAME" /bin/bash
