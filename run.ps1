param(
    [switch]$Rebuild,
    [string]$Display = "host.docker.internal:0.0",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

$composeFile = "docker/docker-compose-windows.yml"
$serviceName = "uav-testing"
$env:DISPLAY = $Display

$sim = $false
$simArgs = @()

function Show-Usage {
    Write-Host "Usage: .\run.ps1 [-Rebuild] [-Display <display>] [--sim <cli.py args...>]"
    Write-Host "  -Rebuild              Force a rebuild of the container image."
    Write-Host "  -Display <display>    X display forwarded into the container."
    Write-Host "  --sim <args...>       Run cli.py inside the container with the"
    Write-Host "                        remaining arguments instead of opening a shell."
    Write-Host "                        Example: .\run.ps1 --sim generate case_studies/mission1.yaml 10"
}

# Parse positional / unknown args. The only supported one is --sim, which
# consumes everything that follows and forwards it verbatim to cli.py.
$i = 0
if ($null -ne $RemainingArgs) {
    while ($i -lt $RemainingArgs.Count) {
        $arg = $RemainingArgs[$i]
        if ($arg -eq "--sim") {
            $sim = $true
            $i++
            while ($i -lt $RemainingArgs.Count) {
                $simArgs += $RemainingArgs[$i]
                $i++
            }
        }
        elseif ($arg -eq "-h" -or $arg -eq "--help") {
            Show-Usage
            exit 0
        }
        else {
            Write-Host "Error: Unknown argument '$arg'"
            Show-Usage
            throw "Unknown argument '$arg'."
        }
    }
}

Write-Host "========================================"
Write-Host "Compose File : $composeFile"
Write-Host "Service      : $serviceName"
Write-Host "Display      : $($env:DISPLAY)"
if ($sim) {
    Write-Host "Mode         : Simulation (cli.py)"
    Write-Host "Sim args     : $($simArgs -join ' ')"
} else {
    Write-Host "Mode         : Interactive shell"
}
Write-Host "========================================"

$running = docker compose -f $composeFile ps -q $serviceName
if ($LASTEXITCODE -ne 0) {
    throw "Failed to query Docker Compose service state."
}

$freshStart = $false

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
    $freshStart = $true
} elseif ([string]::IsNullOrWhiteSpace($running)) {
    Write-Host "Container not running. Ensuring it exists..."

    docker compose -f $composeFile up -d --no-recreate
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start the container."
    }
    $freshStart = $true
} else {
    Write-Host "Container is already running. Skipping start..."
}

# Apply Aerialist source-tree patches once
if ($freshStart) {
    Write-Host "Applying Aerialist patches..."
    docker compose -f $composeFile exec -T $serviceName `
        /bin/bash -c "cd /src/generator && python3 patchAerialist.py"
    # Best-effort: ignore failures so an already-patched tree does not abort the run.
    $LASTEXITCODE = 0
}

if ($sim) {
    Write-Host "Killing any leftover simulation processes..."
    docker compose -f $composeFile exec -T $serviceName `
        /bin/bash /src/generator/kill_simulations.sh
    $LASTEXITCODE = 0

    $argsString = $simArgs -join ' '
    $cmd = "cd /src/generator && python3 cli.py $argsString"
    Write-Host "Running inside container: $cmd"
    docker compose -f $composeFile exec -T $serviceName /bin/bash -c $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Simulation failed with exit code $LASTEXITCODE."
    }
} else {
    Write-Host "Attaching to interactive bash shell..."
    docker compose -f $composeFile exec $serviceName /bin/bash
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to open an interactive shell in the container."
    }
}
