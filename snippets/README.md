# Developing Your Own Test Generator

You can integrate Aerialist's python package in your own code and directly define and execute UAV test cases with it. This is useful when working on test-generation approaches for UAVs; an example of such usage can be found in [Surrealist](https://github.com/skhatiri/Surrealist).

1. `pip3 install git+https://github.com/skhatiri/Aerialist.git`
2. Experiment first with the local agent provided by the Docker image — make sure you can run test cases inside the container using the helper scripts at the repo root.
3. Check [`TestCase`](testcase.py) for a simple wrapper that defines and executes a test case.
4. Check [`RandomGenerator`](random_generator.py) for a baseline generator that drops obstacles with random size and position into a case-study mission.
5. Check [`cli.py`](cli.py) for the entry point that wires the generator into Aerialist.
6. Check [`Dockerfile`](Dockerfile) for how to dockerise your own code.

## Usage

Invoke the entry point with the `generate` subcommand:

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
| `budget` | Yes | Total simulation budget for the run. |

## Environment

| Variable | Default | Description |
|---|---|---|
| `AGENT` | `local` | Aerialist execution backend (`local`, `docker`, or `k8s`). |
