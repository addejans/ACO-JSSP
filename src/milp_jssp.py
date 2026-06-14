"""MILP baseline for the Job Shop Scheduling Problem using SciPy/HiGHS.

The ACO solver is intentionally heuristic. This module provides an exact mixed-
integer linear programming baseline for small and medium generated instances so
solution quality can be compared against a solver-backed objective value.

SciPy is an optional dependency. Install it with:

    python -m pip install scipy
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from src.aco_jssp import JobShopProblem, ScheduledOperation


@dataclass(frozen=True)
class MilpSolveResult:
    status: int
    message: str
    objective: float | None
    optimal: bool
    mip_gap: float | None
    runtime_seconds: float
    schedule: List[ScheduledOperation]


def _operations(problem: JobShopProblem) -> List[Tuple[int, int, int, int]]:
    """Return operations as (job, op_index, machine, duration)."""

    ops: List[Tuple[int, int, int, int]] = []
    for job, route in enumerate(problem.jobs):
        for index, (machine, duration) in enumerate(route):
            ops.append((job, index, machine, duration))
    return ops


def solve_milp_highs(problem: JobShopProblem, time_limit: float | None = 10.0) -> MilpSolveResult:
    """Solve a JSSP instance as a disjunctive MILP with SciPy's HiGHS backend.

    Variables:
    - start time for every operation;
    - one makespan variable;
    - one binary ordering variable for every pair of operations sharing a machine.

    The formulation minimizes makespan subject to job precedence and machine
    non-overlap constraints.
    """

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception as exc:  # pragma: no cover - depends on optional SciPy install
        raise RuntimeError("SciPy is required for the HiGHS MILP baseline. Install with `python -m pip install scipy`." ) from exc

    start_clock = time.perf_counter()
    ops = _operations(problem)
    n_ops = len(ops)
    cmax_idx = n_ops
    horizon = float(sum(duration for *_prefix, duration in ops))

    by_job: Dict[Tuple[int, int], int] = {(job, index): i for i, (job, index, _m, _d) in enumerate(ops)}
    machine_pairs: List[Tuple[int, int]] = []
    for machine in range(problem.num_machines):
        machine_ops = [i for i, (_job, _index, m, _duration) in enumerate(ops) if m == machine]
        for pos, first in enumerate(machine_ops):
            for second in machine_ops[pos + 1:]:
                machine_pairs.append((first, second))

    first_binary_idx = n_ops + 1
    n_vars = first_binary_idx + len(machine_pairs)
    c = np.zeros(n_vars)
    c[cmax_idx] = 1.0

    lb = np.zeros(n_vars)
    ub = np.full(n_vars, horizon)
    lb[first_binary_idx:] = 0.0
    ub[first_binary_idx:] = 1.0
    integrality = np.zeros(n_vars)
    integrality[first_binary_idx:] = 1

    rows: List[Tuple[Dict[int, float], float, float]] = []

    # Job precedence: s_{j,k+1} - s_{j,k} >= p_{j,k}
    for job in range(problem.num_jobs):
        for index in range(problem.operations_per_job - 1):
            prev_idx = by_job[(job, index)]
            next_idx = by_job[(job, index + 1)]
            duration = ops[prev_idx][3]
            rows.append(({next_idx: 1.0, prev_idx: -1.0}, float(duration), np.inf))

    # Makespan: Cmax - s_i >= p_i
    for op_idx, (_job, _index, _machine, duration) in enumerate(ops):
        rows.append(({cmax_idx: 1.0, op_idx: -1.0}, float(duration), np.inf))

    # Machine disjunctions with y=1 meaning first before second.
    for pair_idx, (first, second) in enumerate(machine_pairs):
        y_idx = first_binary_idx + pair_idx
        first_duration = ops[first][3]
        second_duration = ops[second][3]
        rows.append(({second: 1.0, first: -1.0, y_idx: -horizon}, float(first_duration) - horizon, np.inf))
        rows.append(({first: 1.0, second: -1.0, y_idx: horizon}, float(second_duration), np.inf))

    a = lil_matrix((len(rows), n_vars), dtype=float)
    lower = np.empty(len(rows), dtype=float)
    upper = np.empty(len(rows), dtype=float)
    for row_idx, (coefficients, lo, hi) in enumerate(rows):
        for col, value in coefficients.items():
            a[row_idx, col] = value
        lower[row_idx] = lo
        upper[row_idx] = hi

    options = {"disp": False}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(a.tocsr(), lower, upper),
        options=options,
    )
    runtime = time.perf_counter() - start_clock

    schedule: List[ScheduledOperation] = []
    if result.x is not None:
        for op_idx, (job, index, machine, duration) in enumerate(ops):
            start = int(round(float(result.x[op_idx])))
            schedule.append(
                ScheduledOperation(
                    job=job,
                    index=index,
                    machine=machine,
                    duration=duration,
                    start=start,
                    end=start + duration,
                )
            )
        schedule.sort(key=lambda op: (op.start, op.machine, op.job, op.index))

    return MilpSolveResult(
        status=int(result.status),
        message=str(result.message),
        objective=None if result.fun is None else float(result.fun),
        optimal=bool(result.success),
        mip_gap=None if not hasattr(result, "mip_gap") else float(result.mip_gap),
        runtime_seconds=runtime,
        schedule=schedule,
    )
