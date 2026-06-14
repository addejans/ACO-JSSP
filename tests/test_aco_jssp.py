import unittest

from src.aco_jssp import demo_problem, schedule_from_job_order, solve_aco


class AcoJsspTests(unittest.TestCase):
    def test_schedule_respects_job_precedence_and_machine_capacity(self):
        problem = demo_problem()
        order = [0, 1, 2, 3] * problem.num_machines
        makespan, schedule = schedule_from_job_order(problem, order)
        self.assertGreater(makespan, 0)
        by_job = {}
        by_machine = {}
        for op in schedule:
            by_job.setdefault(op.job, []).append(op)
            by_machine.setdefault(op.machine, []).append(op)
        for ops in by_job.values():
            ops.sort(key=lambda op: op.index)
            for prev, nxt in zip(ops, ops[1:]):
                self.assertLessEqual(prev.end, nxt.start)
        for ops in by_machine.values():
            ops.sort(key=lambda op: op.start)
            for prev, nxt in zip(ops, ops[1:]):
                self.assertLessEqual(prev.end, nxt.start)

    def test_solver_is_deterministic_for_fixed_seed(self):
        problem = demo_problem()
        first = solve_aco(problem, ants=8, iterations=10, seed=11)
        second = solve_aco(problem, ants=8, iterations=10, seed=11)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[2], second[2])

    def test_solver_returns_complete_schedule(self):
        problem = demo_problem()
        makespan, schedule, order = solve_aco(problem, ants=8, iterations=10, seed=1)
        self.assertGreater(makespan, 0)
        self.assertEqual(len(schedule), problem.num_operations)
        self.assertEqual(len(order), problem.num_operations)


if __name__ == "__main__":
    unittest.main()
