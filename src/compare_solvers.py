"""Compare the ACO heuristic against the HiGHS MILP baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from src.aco_jssp import demo_problem, problem_to_dict, random_problem, schedule_to_dict, solve_aco
from src.milp_jssp import solve_milp_highs


def compare(
    *,
    demo: bool = False,
    jobs: int = 4,
    machines: int = 3,
    min_duration: int = 1,
    max_duration: int = 20,
    instance_seed: int = 7,
    algorithm_seed: int = 11,
    ants: int = 40,
    iterations: int = 120,
    local_search_steps: int = 80,
    milp_time_limit: float = 10.0,
) -> dict:
    problem = demo_problem() if demo else random_problem(
        num_jobs=jobs,
        num_machines=machines,
        seed=instance_seed,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    aco_start = time.perf_counter()
    aco_makespan, aco_schedule, aco_order = solve_aco(
        problem,
        ants=ants,
        iterations=iterations,
        seed=algorithm_seed,
        local_search_steps=local_search_steps,
    )
    aco_runtime = time.perf_counter() - aco_start

    milp = solve_milp_highs(problem, time_limit=milp_time_limit)
    best_known = milp.objective
    aco_gap = None
    if best_known and best_known > 0:
        aco_gap = (aco_makespan - best_known) / best_known

    return {
        "problem": problem_to_dict(problem),
        "settings": {
            "demo": demo,
            "instance_seed": instance_seed,
            "algorithm_seed": algorithm_seed,
            "ants": ants,
            "iterations": iterations,
            "local_search_steps": local_search_steps,
            "milp_time_limit": milp_time_limit,
        },
        "aco": schedule_to_dict(aco_makespan, aco_schedule, aco_order),
        "aco_runtime_seconds": aco_runtime,
        "milp_highs": {
            "status": milp.status,
            "message": milp.message,
            "objective": milp.objective,
            "optimal": milp.optimal,
            "mip_gap": milp.mip_gap,
            "runtime_seconds": milp.runtime_seconds,
            "operations": [op.__dict__ | {"label": op.id} for op in milp.schedule],
        },
        "comparison": {
            "aco_makespan": aco_makespan,
            "milp_makespan_or_best_bound": best_known,
            "aco_gap_vs_milp": aco_gap,
            "aco_gap_percent": None if aco_gap is None else 100.0 * aco_gap,
            "milp_proved_optimal": milp.optimal,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare ACO against a HiGHS MILP baseline.")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--machines", type=int, default=3)
    parser.add_argument("--min-duration", type=int, default=1)
    parser.add_argument("--max-duration", type=int, default=20)
    parser.add_argument("--instance-seed", type=int, default=7)
    parser.add_argument("--algorithm-seed", type=int, default=11)
    parser.add_argument("--ants", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--local-search-steps", type=int, default=80)
    parser.add_argument("--milp-time-limit", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    payload = compare(
        demo=args.demo,
        jobs=args.jobs,
        machines=args.machines,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        instance_seed=args.instance_seed,
        algorithm_seed=args.algorithm_seed,
        ants=args.ants,
        iterations=args.iterations,
        local_search_steps=args.local_search_steps,
        milp_time_limit=args.milp_time_limit,
    )
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
