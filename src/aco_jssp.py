"""Ant Colony Optimization demo solver for the Job Shop Scheduling Problem.

This module is intentionally small and dependency-free. It provides a modern,
Python 3 implementation that can be used from tests, scripts, or a UI layer while
leaving the original 2017 research scripts in the repository for historical
reference.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Operation:
    """A single operation in a job route."""

    job: int
    index: int
    machine: int
    duration: int

    @property
    def id(self) -> str:
        return f"J{self.job + 1}-O{self.index + 1}"


@dataclass(frozen=True)
class ScheduledOperation(Operation):
    """An operation with concrete start and end times."""

    start: int
    end: int


@dataclass(frozen=True)
class JobShopProblem:
    """A compact JSSP instance.

    jobs[j][k] is the kth operation of job j and is represented as
    (machine_id, processing_time). Machine ids are zero-based.
    """

    jobs: Tuple[Tuple[Tuple[int, int], ...], ...]

    def __post_init__(self) -> None:
        if not self.jobs:
            raise ValueError("A job shop instance must include at least one job.")
        route_lengths = {len(job) for job in self.jobs}
        if 0 in route_lengths:
            raise ValueError("Every job must contain at least one operation.")
        if len(route_lengths) != 1:
            raise ValueError("This demo solver expects each job to have the same number of operations.")
        for job in self.jobs:
            for machine, duration in job:
                if machine < 0:
                    raise ValueError("Machine ids must be non-negative.")
                if duration <= 0:
                    raise ValueError("Operation durations must be positive.")

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    @property
    def num_machines(self) -> int:
        return len(self.jobs[0])

    @property
    def num_operations(self) -> int:
        return self.num_jobs * self.num_machines


def demo_problem() -> JobShopProblem:
    """Return the 4-job, 3-machine instance from the original repository scripts."""

    return JobShopProblem(
        jobs=(
            ((0, 1), (1, 2), (2, 3)),
            ((1, 4), (0, 5), (2, 6)),
            ((0, 7), (2, 8), (1, 9)),
            ((0, 10), (1, 11), (2, 12)),
        )
    )


def schedule_from_job_order(problem: JobShopProblem, job_order: Sequence[int]) -> Tuple[int, List[ScheduledOperation]]:
    """Build an active schedule from a job-order representation."""

    if len(job_order) != problem.num_operations:
        raise ValueError(f"Expected {problem.num_operations} job selections, got {len(job_order)}.")

    remaining_counts = {j: job_order.count(j) for j in range(problem.num_jobs)}
    expected = problem.num_machines
    bad_counts = {j: count for j, count in remaining_counts.items() if count != expected}
    if bad_counts:
        raise ValueError(f"Each job must appear {expected} times. Bad counts: {bad_counts}.")

    next_operation = [0] * problem.num_jobs
    job_ready = [0] * problem.num_jobs
    max_machine_id = max(machine for job in problem.jobs for machine, _ in job)
    machine_ready = [0] * (max_machine_id + 1)
    scheduled: List[ScheduledOperation] = []

    for job in job_order:
        if job < 0 or job >= problem.num_jobs:
            raise ValueError(f"Unknown job id in order: {job}")
        op_index = next_operation[job]
        if op_index >= problem.num_machines:
            raise ValueError(f"Job {job} appears too many times in the order.")
        machine, duration = problem.jobs[job][op_index]
        start = max(job_ready[job], machine_ready[machine])
        end = start + duration
        scheduled.append(
            ScheduledOperation(
                job=job,
                index=op_index,
                machine=machine,
                duration=duration,
                start=start,
                end=end,
            )
        )
        next_operation[job] += 1
        job_ready[job] = end
        machine_ready[machine] = end

    return max(machine_ready), scheduled


def _weighted_choice(rng: random.Random, choices: Sequence[int], weights: Sequence[float]) -> int:
    total = sum(weights)
    if total <= 0 or any(math.isnan(w) or w < 0 for w in weights):
        return rng.choice(list(choices))
    threshold = rng.random() * total
    cumulative = 0.0
    for choice, weight in zip(choices, weights):
        cumulative += weight
        if cumulative >= threshold:
            return choice
    return choices[-1]


def _construct_order(
    problem: JobShopProblem,
    pheromone: Dict[Tuple[int, int], float],
    rng: random.Random,
    alpha: float,
    beta: float,
) -> List[int]:
    remaining = [problem.num_machines] * problem.num_jobs
    next_operation = [0] * problem.num_jobs
    job_ready = [0] * problem.num_jobs
    max_machine_id = max(machine for job in problem.jobs for machine, _ in job)
    machine_ready = [0] * (max_machine_id + 1)
    order: List[int] = []
    previous_job = -1

    for _ in range(problem.num_operations):
        feasible = [j for j in range(problem.num_jobs) if remaining[j] > 0]
        weights: List[float] = []
        for job in feasible:
            op_index = next_operation[job]
            machine, duration = problem.jobs[job][op_index]
            earliest_start = max(job_ready[job], machine_ready[machine])
            earliest_finish = earliest_start + duration
            tau = pheromone.get((previous_job, job), 1.0)
            eta = 1.0 / max(earliest_finish, 1)
            weights.append((tau ** alpha) * (eta ** beta))

        chosen = _weighted_choice(rng, feasible, weights)
        order.append(chosen)

        op_index = next_operation[chosen]
        machine, duration = problem.jobs[chosen][op_index]
        start = max(job_ready[chosen], machine_ready[machine])
        finish = start + duration
        job_ready[chosen] = finish
        machine_ready[machine] = finish
        next_operation[chosen] += 1
        remaining[chosen] -= 1
        previous_job = chosen

    return order


def solve_aco(
    problem: JobShopProblem,
    ants: int = 30,
    iterations: int = 80,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.35,
    q: float = 100.0,
    seed: int | None = 7,
) -> Tuple[int, List[ScheduledOperation], List[int]]:
    """Run a compact ACO heuristic and return the best schedule found."""

    if ants <= 0:
        raise ValueError("ants must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 <= evaporation < 1:
        raise ValueError("evaporation must be in [0, 1)")

    rng = random.Random(seed)
    pheromone: Dict[Tuple[int, int], float] = {(-1, j): 1.0 for j in range(problem.num_jobs)}
    for i in range(problem.num_jobs):
        for j in range(problem.num_jobs):
            pheromone[(i, j)] = 1.0

    best_makespan = math.inf
    best_schedule: List[ScheduledOperation] = []
    best_order: List[int] = []

    for _ in range(iterations):
        iteration_best_makespan = math.inf
        iteration_best_order: List[int] = []

        for _ant in range(ants):
            order = _construct_order(problem, pheromone, rng, alpha, beta)
            makespan, schedule = schedule_from_job_order(problem, order)
            if makespan < iteration_best_makespan:
                iteration_best_makespan = makespan
                iteration_best_order = order
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = schedule
                best_order = order

        for edge in list(pheromone):
            pheromone[edge] *= 1.0 - evaporation
            pheromone[edge] = max(pheromone[edge], 1e-6)

        previous = -1
        deposit = q / max(iteration_best_makespan, 1)
        for job in iteration_best_order:
            pheromone[(previous, job)] = pheromone.get((previous, job), 1.0) + deposit
            previous = job

    return int(best_makespan), best_schedule, best_order


def schedule_to_dict(makespan: int, schedule: Iterable[ScheduledOperation], order: Sequence[int]) -> dict:
    return {
        "makespan": makespan,
        "order": list(order),
        "operations": [asdict(operation) | {"label": operation.id} for operation in schedule],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ACO-JSSP demo solver.")
    parser.add_argument("--ants", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    problem = demo_problem()
    makespan, schedule, order = solve_aco(problem, ants=args.ants, iterations=args.iterations, seed=args.seed)
    payload = schedule_to_dict(makespan, schedule, order)
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
