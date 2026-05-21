#!/bin/bash

# Default variables
COMPOSE_FILE="docker/docker-compose.yml"
SERVICE_NAME="uav-testing"
REBUILD=false
SIM=false
SIM_ARGS=()

usage() {
    echo "Usage: $0 [--nogpu] [--rebuild] [--sim <cli.py args...>]"
    echo "  --nogpu               Use the CPU-only compose file."
    echo "  --rebuild             Force a rebuild of the container image."
    echo "  --sim <args...>       Run cli.py inside the container with the"
    echo "                        remaining arguments instead of opening a shell."
    echo "                        Example: $0 --sim generate case_studies/mission1.yaml 10"
}

# Parse input parameters
while [ $# -gt 0 ]; do
    case $1 in
        --nogpu)
            COMPOSE_FILE="docker/docker-compose-nogpu.yml"
            shift
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --sim)
            SIM=true
            shift
            # Everything after --sim is forwarded verbatim to cli.py.
            while [ $# -gt 0 ]; do
                SIM_ARGS+=("$1")
                shift
            done
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Compose File : $COMPOSE_FILE"
echo "Service      : $SERVICE_NAME"
if [ "$SIM" = true ]; then
    echo "Mode         : Simulation (cli.py)"
    echo "Sim args     : ${SIM_ARGS[*]}"
else
    echo "Mode         : Interactive shell"
fi
echo "========================================"

# Check if the container is already running
RUNNING=$(docker compose -f "$COMPOSE_FILE" ps -q "$SERVICE_NAME")
FRESH_START=false

# Set X11 permissions
if command -v xhost >/dev/null 2>&1; then
    xhost +local:docker || true
fi

if [ "$REBUILD" = true ]; then
    echo "Rebuilding image..."
    docker compose -f "$COMPOSE_FILE" down
    docker compose -f "$COMPOSE_FILE" up -d --build
    FRESH_START=true
elif [ -z "$RUNNING" ]; then
    echo "Container not running. Ensuring it exists..."
    docker compose -f "$COMPOSE_FILE" up -d --no-recreate
    FRESH_START=true
else
    echo "Container is already running. Skipping start..."
fi

# Apply Aerialist source-tree patches once
if [ "$FRESH_START" = true ]; then
    echo "Applying Aerialist patches..."
    docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE_NAME" \
        /bin/bash -c "cd /src/generator && python3 patchAerialist.py" || true
fi

if [ "$SIM" = true ]; then
    # Kill stale runs
    echo "Killing any leftover simulation processes..."
    docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE_NAME" \
        /bin/bash /src/generator/kill_simulations.sh || true

    CMD="cd /src/generator && python3 cli.py ${SIM_ARGS[*]}"
    echo "Running inside container: $CMD"
    docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE_NAME" /bin/bash -c "$CMD"
else
    # Enter the interactive bash shell
    echo "Attaching to interactive bash shell..."
    docker compose -f "$COMPOSE_FILE" exec "$SERVICE_NAME" /bin/bash
fi
