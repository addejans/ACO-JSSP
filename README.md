# ACO-JSSP

Job Shop Scheduling Problem via Ant Colony Optimization.

This repository began as a 2017 research/prototype implementation of Ant Colony Optimization (ACO) for a small Job Shop Scheduling Problem (JSSP). It now includes a cleaned Python 3 solver, tests, and a dependency-free browser visualizer for the demo instance.

## What this solves

In a job shop scheduling problem, each job has a fixed sequence of operations. Each operation must run on a specific machine for a given duration. The goal is to sequence operations on machines while respecting job precedence and machine capacity, usually minimizing the final completion time, or **makespan**.

ACO is a metaheuristic inspired by ants depositing pheromone on paths. In this project, an ant constructs a complete job-order sequence. Better schedules reinforce the job transitions that produced them.

## Quick start

```bash
python -m unittest discover -s tests -v
python -m src.aco_jssp --ants 30 --iterations 80 --seed 7
python -m src.aco_jssp --output docs/sample-schedule.json
```

The cleaned implementation is dependency-free and uses only the Python standard library.

## Browser visualizer

The visualizer lives in [`docs/index.html`](docs/index.html). It runs entirely in the browser and renders a Gantt chart for the demo JSSP instance.

After this PR is merged, GitHub Pages can be deployed from the included workflow:

```text
.github/workflows/pages.yml
```

The Pages workflow publishes the `docs/` directory.

## Repository structure

```text
src/aco_jssp.py          # Modern Python 3 ACO demo solver
tests/test_aco_jssp.py   # Unit tests for schedule validity and determinism
docs/index.html          # Static browser visualizer / Gantt UI
ERRORS_AND_FIXES.md      # Review notes and corrected issues
```

The original scripts are still present as historical artifacts. New work should start from `src/aco_jssp.py`.

## Demo instance

The included demo mirrors the original 4-job, 3-machine setup:

| Job | Operation 1 | Operation 2 | Operation 3 |
| --- | --- | --- | --- |
| J1 | M1 / 1 | M2 / 2 | M3 / 3 |
| J2 | M2 / 4 | M1 / 5 | M3 / 6 |
| J3 | M1 / 7 | M3 / 8 | M2 / 9 |
| J4 | M1 / 10 | M2 / 11 | M3 / 12 |

Machine labels are shown one-based in the README/UI and zero-based in the implementation.

## Review notes

See [`ERRORS_AND_FIXES.md`](ERRORS_AND_FIXES.md) for the code review notes, bugs found, and modernization changes.
