#!/usr/bin/env bash
# Terminate every Python / Aerialist / PX4 / Gazebo / ROS process that may be running
# on this host (or inside this container).
#
# Usage: kill_simulations.sh [--dry-run]
set -uo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

PATTERNS=(
  "python3 main.py"
  "px4"
  "px4_sitl"
  "sitl_run"
  "gazebo"
  "gzserver"
  "gzclient"
  "rosmaster"
  "roscore"
  "roslaunch"
  "rosout"
  "robot_state_publisher"
  "static_transform_publisher"
  "nodelet"
  "mavros"
  "mavros_node"
  "mavlink_node"
  "jmavsim"
  "mavsdk_server"
  "micrortps"
  "MicroXRCEAgent"
)

# Don't kill ourselves or our parent shell.
SELF_PID=$$
PARENT_PID=$PPID

kill_pattern() {
  local pattern="$1"
  local signal="$2"
  local pids filtered
  pids="$(pgrep -f -- "$pattern" 2>/dev/null || true)"
  [[ -z "$pids" ]] && return

  filtered=""
  for pid in $pids; do
    if [[ "$pid" == "$SELF_PID" || "$pid" == "$PARENT_PID" ]]; then
      continue
    fi
    filtered="$filtered $pid"
  done
  filtered="${filtered# }"
  [[ -z "$filtered" ]] && return

  echo "[sim-kill] $signal '$pattern': $filtered"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    # shellcheck disable=SC2086
    kill -"$signal" $filtered 2>/dev/null || true
  fi
}

echo "[sim-kill] stopping local simulation processes..."

for p in "${PATTERNS[@]}"; do
  kill_pattern "$p" TERM
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  sleep 2
fi

for p in "${PATTERNS[@]}"; do
  kill_pattern "$p" KILL
done

# Host-side cleanup: only stop sibling Aerialist containers when we're NOT
# running inside a container ourselves — otherwise we'd race against the
# container that's executing this very script.
if [[ ! -f /.dockerenv ]] && command -v docker >/dev/null 2>&1; then
  echo "[sim-kill] checking Docker containers..."
  mapfile -t container_ids < <(
    docker ps --format '{{.ID}} {{.Image}} {{.Names}}' 2>/dev/null \
      | grep -Ei 'aerialist|px4|gazebo|jmavsim|ros' \
      | grep -v 'se4a-uav-testing' \
      | awk '{print $1}'
  )

  if [[ "${#container_ids[@]}" -gt 0 ]]; then
    echo "[sim-kill] matched containers: ${container_ids[*]}"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      docker stop "${container_ids[@]}" >/dev/null || true
    fi
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[sim-kill] dry run complete."
else
  echo "[sim-kill] done."
fi
