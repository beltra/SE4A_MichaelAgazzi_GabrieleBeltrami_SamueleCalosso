# Developing Your Own Test Generator

You can integrate Aerialist's python package in your own code and directly define and execute UAV test cases with it. This is useful when working on test-generation approaches for UAVs; an example of such usage can be found in [Surrealist](https://github.com/skhatiri/Surrealist).

1. `pip3 install git+https://github.com/skhatiri/Aerialist.git`
2. Experiment first with the local agent provided by the Docker image; make sure you can run test cases inside the container using the helper scripts at the repo root.
3. Check [`TestCase`](testcase.py) for a simple wrapper that defines and executes a test case.
4. Check [`RandomGenerator`](random_generator.py) for a baseline generator that drops obstacles with random size and position into a case-study mission.
5. Check [`cli.py`](cli.py) for the entry point; the active generator is chosen via the `GENERATOR` constant at the top of the file.
6. Check [`Dockerfile`](Dockerfile) for how to dockerise your own code.
7. Check [`geneticAlgorithm/`](geneticAlgorithm) for the genetic generator; `python3 -m pytest -q` runs its unit tests without the simulator.

## Our test generator

`geneticAlgorithm/` is a genetic algorithm over path-relative obstacle
layouts: each test is one thin wall or a two-wall gate placed across a
straight stretch of the mission, described by where it crosses the path, at
what angle and how far it reaches past it. Layouts are checked before they
are simulated, one- and two-obstacle layouts evolve in separate niches, the
best layouts are re-run to confirm them, and the returned suite is ranked by
the official score and filtered by trajectory similarity.

```bash
cd /src/generator
python3 cli.py generate case_studies/mission3.yaml 100
```

## Authors

| Name | Email | Affiliation |
|---|---|---|
| Gabriele Beltrami | gabriele1.beltrami@mail.polimi.it | Politecnico di Milano |
| Michael Agazzi | michael.agazzi@mail.polimi.it | Politecnico di Milano |
| Samuele Calosso | samuele.calosso@mail.polimi.it | Politecnico di Milano |

## Usage

Select the generator by editing the `GENERATOR` constant at the top of [`cli.py`](cli.py) (`"ga"` or `"random"`), then run:

```bash
python3 cli.py generate <test> <budget>
```

Example:

```bash
python3 cli.py generate case_studies/mission1.yaml 5
```

| Parameter | Required | Description |
|---|---|---|
| `test` | Yes | Path to the case-study YAML, e.g. `case_studies/mission1.yaml`. |
| `budget` | Yes | Total number of simulations the generator may run. |

## Environment

| Variable | Default | Description |
|---|---|---|
| `AGENT` | `local` | Aerialist execution backend (`local`, `docker`, or `k8s`); the Linux compose files set `docker`. |
| `GA_SEED` | unset | Random seed of the genetic generator; the same seed repeats a run exactly. |
| `GA_WORKERS` | `1` | Simulations run in parallel, one worker container each; only used when the agent is isolated. The compose files set 4 (GPU) and 2 (CPU). |
| `GA_EVOLVE` | `1` | `0` replaces breeding with fresh seeds every generation, the control arm of the A/B measurement. |
| `DOCKER_IMG` | `docker-uav-testing` | Image of the worker containers. |
| `DOCKER_TIMEOUT` | unset | Per-simulation timeout in seconds inside a worker; the compose files set 300. |
| `SIM_WORKER_GPU` | `true` | Give workers the GPU and host X11; `false` uses `xvfb-run` with software OpenGL. |
| `SIM_WORKER_SECURITY_OPT` | unset | Passed to `docker run --security-opt` for the workers. Set it to `apparmor=unconfined` on hosts whose Docker cannot apply its default AppArmor profile, where every worker otherwise fails to start. |
| `SIM_WORKER_SPEED` | unset | Upper bound on the simulation speed factor per test when workers share one host. |
| `LOGS_COPY_DIR` | `results/logs/` | Where worker logs are copied back. |
| `TESTS_FOLDER` | `./generated_tests/` | Output folder of the generated tests. |
