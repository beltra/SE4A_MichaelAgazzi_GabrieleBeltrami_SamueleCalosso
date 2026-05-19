# UAV Testing Competition

<p align="center">
  <img src="figures/uav1.gif" width="45%" height="45%"/>
  <img src="figures/uav2.gif" width="45%" height="45%"/>
</p>

This repository is a fork of the [UAV Testing Competition](https://github.com/skhatiri/UAV-Testing-Competition) starter material. It bundles the sample case studies, a runnable random generator, helper launch scripts, and the Docker environment used to build test generators on top of [Aerialist](https://github.com/skhatiri/Aerialist).

## Competition Context

Unmanned Aerial Vehicles (UAVs) are increasingly used in real-world scenarios such as crop monitoring, surveillance, and delivery. Open platforms such as [PX4](https://github.com/PX4/PX4-Autopilot) and [Ardupilot](https://github.com/ArduPilot/ardupilot) have made development easier, but systematic testing is still a major challenge.

The UAV Testing Competition is organised jointly with [ICST 2026](https://conf.researchr.org/home/icst-2026) and [SBFT@ICSE 2026](https://search-based-and-fuzz-testing.github.io/sbft26/) to support researchers exploring testing techniques for autonomous drones.

## Repository Layout

- [`README.md`](README.md): project overview and onboarding
- [`run.sh`](run.sh): Linux/macOS helper to enter the Docker environment
- [`run.ps1`](run.ps1): Windows PowerShell helper for the Docker environment
- [`docker/docker-compose.yml`](docker/docker-compose.yml): GPU-enabled container setup
- [`docker/docker-compose-nogpu.yml`](docker/docker-compose-nogpu.yml): CPU-only container setup
- [`docker/docker-compose-windows.yml`](docker/docker-compose-windows.yml): Windows Docker Desktop setup with X forwarding
- [`snippets/Dockerfile`](snippets/Dockerfile): image definition for the generator environment
- [`snippets/cli.py`](snippets/cli.py): entry point that dispatches on the chosen generator (selected via the `GENERATOR` constant at the top of the file)
- [`snippets/random_generator.py`](snippets/random_generator.py): sample random-obstacle generator
- [`snippets/testcase.py`](snippets/testcase.py): execution wrapper around an `AerialistTest`
- [`snippets/case_studies/`](snippets/case_studies): sample missions, plans, parameters, logs, and images
- [`snippets/requirements.txt`](snippets/requirements.txt): extra Python dependencies for the generator

## Quick Start

### Linux / macOS

```bash
./run.sh             # GPU-enabled container
./run.sh --nogpu     # CPU-only container
./run.sh --rebuild   # rebuild after editing snippets/Dockerfile or requirements.txt
```

### Windows

```powershell
.\run.ps1            # start (or attach to) the container
.\run.ps1 -Rebuild   # rebuild after editing snippets/Dockerfile or requirements.txt
```

The helpers attach to an interactive shell inside the `uav-testing` service. The host `snippets/` folder is mounted at `/src/generator` so edits made on the host are visible inside the container.

## Running a Test Generator

Once inside the container, invoke the entry point with the `generate` subcommand:

```bash
cd /src/generator
python3 cli.py generate case_studies/mission1.yaml 5
```

This runs the selected generator for `5` test cases on `mission1.yaml`. The active generator is chosen by the `GENERATOR` constant at the top of [`snippets/cli.py`](snippets/cli.py) (`"ga"` for the genetic algorithm, `"random"` for the baseline random generator).

## Case Studies

Three starter missions live in [`snippets/case_studies/`](snippets/case_studies):

- `mission1.yaml`: take off, fly forward about 50 m, land.
- `mission2.yaml`: take off, head to a waypoint, return offset left, land.
- `mission3.yaml`: take off, fly through multiple waypoints, return, land.

All missions share the same valid obstacle area: `-40 < x < 30` and `10 < y < 40`. See [`snippets/case_studies/README.md`](snippets/case_studies/README.md) for the full description.

## License

MIT, see [LICENSE.md](LICENSE.md).
