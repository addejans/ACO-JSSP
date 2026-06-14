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
    def operations_per_job(self) -> int:
        return len(self.jobs[0])

    @property
    def num_machines(self) -> int:
        return max(machine for job in self.jobs for machine, _ in job) + 1

    @property
    def num_operations(self) -> int:
        return self.num_jobs * self.operations_per_job


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


def random_problem(
    num_jobs: int,
    num_machines: int,
    seed: int | None = 7,
    min_duration: int = 1,
    max_duration: int = 20,
) -> JobShopProblem:
    """Generate a reproducible JSSP instance.

    Each job visits each machine exactly once in a random route order.
    """

    if not 2 <= num_jobs <= 30:
        raise ValueError("num_jobs must be in [2, 30].")
    if not 2 <= num_machines <= 20:
        raise ValueError("num_machines must be in [2, 20].")
    if min_duration <= 0 or max_duration < min_duration:
        raise ValueError("Duration bounds must be positive and ordered.")

    rng = random.Random(seed)
    jobs: List[Tuple[Tuple[int, int], ...]] = []
    for _ in range(num_jobs):
        route = list(range(num_machines))
        rng.shuffle(route)
        jobs.append(tuple((machine, rng.randint(min_duration, max_duration)) for machine in route))
    return JobShopProblem(tuple(jobs))


def schedule_from_job_order(problem: JobShopProblem, job_order: Sequence[int]) -> Tuple[int, List[ScheduledOperation]]:
    """Build an active schedule from a job-order representation."""

    if len(job_order) != problem.num_operations:
        raise ValueError(f"Expected {problem.num_operations} job selections, got {len(job_order)}.")

    remaining_counts = {j: job_order.count(j) for j in range(problem.num_jobs)}
    expected = problem.operations_per_job
    bad_counts = {j: count for j, count in remaining_counts.items() if count != expected}
    if bad_counts:
        raise ValueError(f"Each job must appear {expected} times. Bad counts: {bad_counts}.")

    next_operation = [0] * problem.num_jobs
    job_ready = [0] * problem.num_jobs
    machine_ready = [0] * problem.num_machines
    scheduled: List[ScheduledOperation] = []

    for job in job_order:
        if job < 0 or job >= problem.num_jobs:
            raise ValueError(f"Unknown job id in order: {job}")
        op_index = next_operation[job]
        if op_index >= problem.operations_per_job:
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


def _remaining_work(problem: JobShopProblem, job: int, next_operation: int) -> int:
    return sum(duration for _, duration in problem.jobs[job][next_operation:])


def _construct_order(
    problem: JobShopProblem,
    pheromone: Dict[Tuple[int, int], float],
    rng: random.Random,
    alpha: float,
    beta: float,
    exploitation: float,
) -> List[int]:
    remaining = [problem.operations_per_job] * problem.num_jobs
    next_operation = [0] * problem.num_jobs
    job_ready = [0] * problem.num_jobs
    machine_ready = [0] * problem.num_machines
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
            remaining_work = _remaining_work(problem, job, op_index)
            tau = pheromone.get((previous_job, job), 1.0)
            eta = 1.0 / max(earliest_finish + 0.15 * remaining_work, 1.0)
            weights.append((tau ** alpha) * (eta ** beta))

        if rng.random() < exploitation:
            chosen = feasible[max(range(len(feasible)), key=lambda i: weights[i])]
        else:
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


def improve_order(
    problem: JobShopProblem,
    order: Sequence[int],
    rng: random.Random,
    steps: int = 100,
) -> Tuple[int, List[ScheduledOperation], List[int]]:
    """Improve a job-order sequence with random pair-swap local search."""

    best_order = list(order)
    best_makespan, best_schedule = schedule_from_job_order(problem, best_order)
    if steps <= 0 or len(best_order) < 2:
        return best_makespan, best_schedule, best_order

    for _ in range(steps):
        i, j = sorted(rng.sample(range(len(best_order)), 2))
        if best_order[i] == best_order[j]:
            continue
        candidate = best_order[:]
        candidate[i], candidate[j] = candidate[j], candidate[i]
        makespan, schedule = schedule_from_job_order(problem, candidate)
        if makespan <= best_makespan:
            best_order = candidate
            best_makespan = makespan
            best_schedule = schedule

    return best_makespan, best_schedule, best_order


def _deposit_order(
    pheromone: Dict[Tuple[int, int], float],
    order: Sequence[int],
    amount: float,
) -> None:
    previous = -1
    for job in order:
        pheromone[(previous, job)] = pheromone.get((previous, job), 1.0) + amount
        previous = job


def solve_aco(
    problem: JobShopProblem,
    ants: int = 30,
    iterations: int = 80,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.35,
    q: float = 100.0,
    seed: int | None = 7,
    exploitation: float = 0.15,
    local_search_steps: int = 80,
    elite_weight: float = 0.35,
) -> Tuple[int, List[ScheduledOperation], List[int]]:
    """Run an ACO heuristic and return the best schedule found.

    Improvements over the legacy prototype:
    - seeded reproducibility;
    - probabilistic/exploitative construction mix;
    - remaining-work heuristic term;
    - pair-swap local search;
    - iteration-best and global-elite pheromone reinforcement.
    """

    if ants <= 0:
        raise ValueError("ants must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 <= evaporation < 1:
        raise ValueError("evaporation must be in [0, 1)")
    if not 0 <= exploitation <= 1:
        raise ValueError("exploitation must be in [0, 1)")
    if local_search_steps < 0:
        raise ValueError("local_search_steps must be non-negative")
    if elite_weight < 0:
        raise ValueError("elite_weight must be non-negative")

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
            order = _construct_order(problem, pheromone, rng, alpha, beta, exploitation)
            makespan, schedule, improved_order = improve_order(problem, order, rng, local_search_steps)

            if makespan < iteration_best_makespan:
                iteration_best_makespan = makespan
                iteration_best_order = improved_order
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = schedule
                best_order = improved_order

        for edge in list(pheromone):
            pheromone[edge] *= 1.0 - evaporation
            pheromone[edge] = max(pheromone[edge], 1e-6)

        _deposit_order(pheromone, iteration_best_order, q / max(iteration_best_makespan, 1))
        if best_order and elite_weight:
            _deposit_order(pheromone, best_order, elite_weight * q / max(best_makespan, 1))

    return int(best_makespan), best_schedule, best_order


def problem_to_dict(problem: JobShopProblem) -> dict:
    return {
        "num_jobs": problem.num_jobs,
        "num_machines": problem.num_machines,
        "jobs": [
            [
                {"machine": machine, "duration": duration}
                for machine, duration in job
            ]
            for job in problem.jobs
        ],
    }


def schedule_to_dict(makespan: int, schedule: Iterable[ScheduledOperation], order: Sequence[int], problem: JobShopProblem | None = None) -> dict:
    payload = {
        "makespan": makespan,
        "order": list(order),
        "operations": [asdict(operation) | {"label": operation.id} for operation in schedule],
    }
    if problem is not None:
        payload["problem"] = problem_to_dict(problem)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ACO-JSSP demo solver.")
    parser.add_argument("--ants", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jobs", type=int, default=4, help="Number of jobs for generated instances.")
    parser.add_argument("--machines", type=int, default=3, help="Number of machines for generated instances.")
    parser.add_argument("--min-duration", type=int, default=1)
    parser.add_argument("--max-duration", type=int, default=20)
    parser.add_argument("--demo", action="store_true", help="Use the original 4x3 demo instance.")
    parser.add_argument("--local-search-steps", type=int, default=80)
    parser.add_argument("--exploitation", type=float, default=0.15)
    parser.add_argument("--elite-weight", type=float, default=0.35)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    problem = demo_problem() if args.demo else random_problem(
        num_jobs=args.jobs,
        num_machines=args.machines,
        seed=args.seed,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    makespan, schedule, order = solve_aco(
        problem,
        ants=args.ants,
        iterations=args.iterations,
        seed=args.seed,
        local_search_steps=args.local_search_steps,
        exploitation=args.exploitation,
        elite_weight=args.elite_weight,
    )
    payload = schedule_to_dict(makespan, schedule, order, problem)
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
