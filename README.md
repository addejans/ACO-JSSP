# ACO-JSSP

Job Shop Scheduling Problem via Ant Colony Optimization.

This repository began as a 2017 research/prototype implementation of Ant Colony Optimization (ACO) for a small Job Shop Scheduling Problem (JSSP). It now includes a cleaned Python 3 solver, tests, a HiGHS MILP baseline, and a dependency-free browser visualizer for generated job-shop instances.

## Live demo

Try the browser visualizer here:

https://addejans.github.io/ACO-JSSP/

The demo runs entirely in the browser and lets you choose the number of jobs and machines, tune ACO parameters, inspect the resulting Gantt chart, and compare the heuristic result against a capped exact browser search for small instances.

Updated LaTeX report: [ACO_JSSP_Updated_Report.pdf](paper/ACO_JSSP_Updated_Report.pdf)

## What this solves

In a job shop scheduling problem, each job has a fixed sequence of operations. Each operation must run on a specific machine for a given duration. The goal is to sequence operations on machines while respecting job precedence and machine capacity, usually minimizing the final completion time, or **makespan**.

ACO is a metaheuristic inspired by ants depositing pheromone on paths. In this project, an ant constructs a complete job-order sequence. Better schedules reinforce the job transitions that produced them.

## Quick start

```bash
python -m unittest discover -s tests -v
python -m src.aco_jssp --demo --ants 40 --iterations 120 --seed 7
python -m src.aco_jssp --jobs 8 --machines 5 --ants 60 --iterations 150 --seed 11
python -m src.aco_jssp --jobs 8 --machines 5 --output docs/sample-schedule.json
```

The cleaned ACO implementation is dependency-free and uses only the Python standard library.

## ACO vs HiGHS MILP comparison

The exact MILP baseline uses SciPy's HiGHS backend. Install the optional dependency first:

```bash
python -m pip install -r requirements-optional.txt
```

Then compare ACO against the MILP baseline:

```bash
python -m src.compare_solvers --jobs 4 --machines 3 --ants 40 --iterations 120 --milp-time-limit 10
python -m src.compare_solvers --demo --milp-time-limit 10 --output results/demo-comparison.json
```

The comparison output reports:

- ACO makespan and runtime;
- HiGHS MILP objective, runtime, solver status, and MIP gap;
- ACO optimality gap when HiGHS proves an optimal solution;
- both schedules in JSON-friendly form.

The browser UI also includes a small-instance exact comparison panel. The browser comparison is not SciPy/HiGHS because this is a static GitHub Pages site, but it gives a useful interactive gap check for small instances.

## Scalable generated instances

The solver can generate reproducible random JSSP instances. Each job visits each machine exactly once in a random route order.

Important CLI options:

```text
--jobs                 number of jobs for generated instances
--machines             number of machines for generated instances
--min-duration         minimum operation duration
--max-duration         maximum operation duration
--ants                 number of ants per iteration
--iterations           number of ACO iterations
--local-search-steps   pair-swap local improvement budget
--exploitation         probability of greedily selecting the best weighted move
--elite-weight         global-best pheromone reinforcement weight
```

## Browser visualizer

The visualizer lives in [`docs/index.html`](docs/index.html). It runs entirely in the browser and renders a Gantt chart from the controls in the sidebar:

- number of jobs;
- number of machines;
- operation duration range;
- ant count;
- iteration count;
- random seed;
- local-search budget;
- capped exact-comparison time limit.

The GitHub Pages workflow publishes the `docs/` directory:

```text
.github/workflows/pages.yml
```

For the live site, GitHub Pages should be configured to deploy from GitHub Actions.

## Algorithm improvements

The modern solver is still intentionally compact, but it is no longer a direct mechanical port of the 2017 prototype. It now includes:

- seeded reproducibility;
- generated instances with variable job/machine counts;
- a remaining-work-aware transition heuristic;
- a mixed exploration/exploitation construction rule;
- pheromone evaporation with numerical floor;
- iteration-best pheromone reinforcement;
- global-best elite reinforcement;
- pair-swap local search on constructed job-order sequences;
- tests for precedence feasibility, machine non-overlap, determinism, generated larger instances, and the optional HiGHS MILP baseline.

## Repository structure

```text
src/aco_jssp.py          # Modern Python 3 ACO solver
src/milp_jssp.py         # Exact MILP baseline using SciPy/HiGHS
src/compare_solvers.py   # ACO vs HiGHS comparison runner
tests/test_aco_jssp.py   # Unit tests for schedule validity, determinism, and MILP smoke tests
docs/index.html          # Static browser visualizer / Gantt UI
paper/                   # LaTeX report source and compiled PDF
legacy/                  # Original 2017 prototype scripts
ERRORS_AND_FIXES.md      # Review notes and corrected issues
```

The original scripts have been moved to `legacy/` as historical artifacts. New work should start from `src/aco_jssp.py`.

## Original demo instance

The CLI can still reproduce the original 4-job, 3-machine setup using `--demo`:

| Job | Operation 1 | Operation 2 | Operation 3 |
| --- | --- | --- | --- |
| J1 | M1 / 1 | M2 / 2 | M3 / 3 |
| J2 | M2 / 4 | M1 / 5 | M3 / 6 |
| J3 | M1 / 7 | M3 / 8 | M2 / 9 |
| J4 | M1 / 10 | M2 / 11 | M3 / 12 |

Machine labels are shown one-based in the README/UI and zero-based in the implementation.

## Review notes

See [`ERRORS_AND_FIXES.md`](ERRORS_AND_FIXES.md) for the code review notes, bugs found, and modernization changes.
