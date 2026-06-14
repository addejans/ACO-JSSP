# Errors and Fixes

This repository started as a 2017 research/prototype implementation. The PR modernization keeps the historical scripts, but adds a clean Python 3 implementation and a browser visualizer. During review, the following issues were found.

## Code issues found

1. **Python 2 syntax prevents modern execution.**
   The legacy scripts use `print` statements and the removed `<>` inequality operator. These fail under Python 3.

2. **Deprecated Plotly API usage.**
   The old code imports `plotly.plotly`, which was removed from modern Plotly releases.

3. **Hard-coded Plotly credentials.**
   The historical script includes hard-coded Plotly usernames/API keys. Those credentials should be considered compromised and rotated if still active. The new implementation does not require external plotting services or credentials.

4. **Heavy global mutable state.**
   The original implementation stores jobs, nodes, ants, pheromones, and schedules in globals. That makes resets brittle and makes unit testing difficult.

5. **Shared list aliasing risk.**
   `Node.dependents = [dependents for i in range(K)]` reuses the same list object for every ant unless later overwritten. The modern solver avoids this class of bug by using immutable problem data and local schedule state.

6. **Pheromone accumulator reset and probability edge cases.**
   The old code has unclear accumulator reset semantics and can fall back to arbitrary choices when transition probabilities are empty or numerically unstable. The modern implementation guards weighted choice and evaporates/deposits pheromone deterministically.

7. **Fixed-size demo limitation.**
   The first visualizer only covered the original 4-job, 3-machine instance. The UI now supports generated instances where the user can choose job and machine counts from bounded ranges.

8. **Weak improvement loop.**
   The first cleaned ACO version constructed schedules but did not intensify much after construction. The solver now adds remaining-work-aware move scoring, a mixed exploration/exploitation rule, global-best elite reinforcement, and pair-swap local search.

9. **No automated tests.**
   There were no tests for job precedence, machine capacity, or deterministic behavior. The PR adds tests for the original demo and larger generated instances.

10. **No easy visual inspection.**
   The original Gantt chart depended on online Plotly credentials. The PR adds a dependency-free browser Gantt chart under `docs/`.

## Fixes added

- Added `src/aco_jssp.py`, a Python 3, dependency-free ACO/JSSP solver.
- Added random instance generation with selectable jobs, machines, and duration ranges.
- Added local search, exploitation probability, elite reinforcement, and remaining-work heuristic scoring.
- Added unit tests in `tests/test_aco_jssp.py`.
- Added a static browser visualizer in `docs/index.html` with job/machine controls.
- Added GitHub Actions CI for tests.
- Added a GitHub Pages deployment workflow for the visualizer.
- Updated the README with usage, deployment, scaling, and maintenance notes.

## Not changed intentionally

The original 2017 scripts remain in place for historical reference. New users should start with `src/aco_jssp.py` and the visualizer rather than the legacy scripts.
