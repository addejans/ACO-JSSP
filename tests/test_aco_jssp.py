import unittest

from src.aco_jssp import demo_problem, random_problem, schedule_from_job_order, solve_aco


def assert_valid_schedule(testcase, problem, makespan, schedule):
    testcase.assertGreater(makespan, 0)
    testcase.assertEqual(len(schedule), problem.num_operations)

    by_job = {}
    by_machine = {}
    for op in schedule:
        by_job.setdefault(op.job, []).append(op)
        by_machine.setdefault(op.machine, []).append(op)
        testcase.assertEqual(op.end - op.start, op.duration)

    for ops in by_job.values():
        ops.sort(key=lambda op: op.index)
        for prev, nxt in zip(ops, ops[1:]):
            testcase.assertLessEqual(prev.end, nxt.start)

    for ops in by_machine.values():
        ops.sort(key=lambda op: op.start)
        for prev, nxt in zip(ops, ops[1:]):
            testcase.assertLessEqual(prev.end, nxt.start)


class AcoJsspTests(unittest.TestCase):
    def test_schedule_respects_job_precedence_and_machine_capacity(self):
        problem = demo_problem()
        order = [0, 1, 2, 3] * problem.operations_per_job
        makespan, schedule = schedule_from_job_order(problem, order)
        assert_valid_schedule(self, problem, makespan, schedule)

    def test_solver_is_deterministic_for_fixed_seed(self):
        problem = demo_problem()
        first = solve_aco(problem, ants=8, iterations=10, seed=11)
        second = solve_aco(problem, ants=8, iterations=10, seed=11)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[2], second[2])

    def test_solver_returns_complete_schedule(self):
        problem = demo_problem()
        makespan, schedule, order = solve_aco(problem, ants=8, iterations=10, seed=1)
        self.assertEqual(len(order), problem.num_operations)
        assert_valid_schedule(self, problem, makespan, schedule)

    def test_random_problem_generation_is_reproducible(self):
        a = random_problem(num_jobs=6, num_machines=5, seed=123)
        b = random_problem(num_jobs=6, num_machines=5, seed=123)
        self.assertEqual(a, b)
        self.assertEqual(a.num_jobs, 6)
        self.assertEqual(a.num_machines, 5)
        self.assertEqual(a.num_operations, 30)

    def test_solver_handles_larger_generated_problem(self):
        problem = random_problem(num_jobs=7, num_machines=5, seed=4, min_duration=1, max_duration=12)
        makespan, schedule, order = solve_aco(problem, ants=10, iterations=12, seed=4, local_search_steps=20)
        self.assertEqual(len(order), problem.num_operations)
        assert_valid_schedule(self, problem, makespan, schedule)


if __name__ == "__main__":
    unittest.main()
