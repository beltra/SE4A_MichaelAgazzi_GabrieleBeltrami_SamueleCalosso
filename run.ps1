param(
    [switch]$Rebuild,
    [string]$Display = "host.docker.internal:0.0"
)

$ErrorActionPreference = "Stop"

$composeFile = "docker/docker-compose-windows.yml"
$serviceName = "uav-testing"
$env:DISPLAY = $Display

Write-Host "========================================"
Write-Host "Compose File : $composeFile"
Write-Host "Service      : $serviceName"
Write-Host "Display      : $($env:DISPLAY)"
Write-Host "========================================"

$running = docker compose -f $composeFile ps -q $serviceName
if ($LASTEXITCODE -ne 0) {
    throw "Failed to query Docker Compose service state."
}

if ($Rebuild) {
    Write-Host "Rebuilding image..."
    docker compose -f $composeFile down
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to stop existing Docker Compose services."
    }

    docker compose -f $composeFile up -d --build
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to rebuild and start the container."
    }
} elseif ([string]::IsNullOrWhiteSpace($running)) {
    Write-Host "Container not running. Ensuring it exists..."

    docker compose -f $composeFile up -d --no-recreate
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start the container."
    }
} else {
    Write-Host "Container is already running. Skipping start..."
}

Write-Host "Attaching to interactive bash shell..."
docker compose -f $composeFile exec $serviceName /bin/bash
if ($LASTEXITCODE -ne 0) {
    throw "Failed to open an interactive shell in the container."
}
